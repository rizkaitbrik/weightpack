// Package serving runs images: wpk run, ps, stop and logs, and the gateway
// that routes OpenAI requests to them.
//
// An image is served as a dense serving base plus the LoRA adapters on top
// of it (see plan). The base is exported once, as a Hugging Face directory,
// under <root>/run; its adapters are exported beside it as PEFT adapters.
// One engine process serves each base on a device, and every container on
// that base shares it: the base's weights load once, and each container's
// adapter loads into the running engine. The gateway sends each request to
// the engine that serves its model, under the engine's own name for it.
//
// The engines are stock: vLLM on NVIDIA GPUs and weightpack-engine, the
// reference engine, on NVIDIA GPUs, Apple Silicon GPUs and CPUs. Both
// speak the OpenAI API and load adapters at runtime.
package serving

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	goruntime "runtime"
	"sort"
	"strings"
	"sync"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/export"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/squash"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// Config sets up a Runtime.
type Config struct {
	Store *store.Store
	// Worker bakes serving bases (squash) for images with XOR layers or
	// stacked LoRA layers. Without one, only images whose base is their
	// base layer can run.
	Worker workerpb.WorkerClient
	// Engines maps an engine kind to the command that starts it; kinds it
	// leaves out use DefaultEngines.
	Engines map[string][]string
	// Device is the default device; "" detects one (see DetectDevice).
	Device string
	// Endpoint is the gateway's OpenAI base URL, shown on each container.
	Endpoint string
	// StartTimeout bounds an engine's start, model load included. 0 means
	// 20 minutes.
	StartTimeout time.Duration
	// APIKey, when set, is the bearer token the gateway requires, as the
	// OpenAI API does. Set it whenever the gateway listens beyond loopback.
	APIKey string
	Logger *slog.Logger
}

// Runtime runs containers and the engines behind them.
type Runtime struct {
	cfg       Config
	dir       string
	storeLock sync.Locker

	mu         sync.Mutex
	engines    map[string]*engine    // by base, kind and device
	containers map[string]*container // by name
	routes     map[string]route      // by model name
	reserved   map[string]bool       // names of containers being started
	pins       map[string]int        // serving files runs are preparing

	prepMu sync.Mutex
	prep   map[string]*sync.Mutex // per serving file being prepared

	detectOnce sync.Once
	detected   string
}

type container struct {
	info     api.Container
	engine   *engine
	template string   // the image's chat template, sent with chat requests
	adapters []string // engine-side adapter names to unload on stop
	files    []string // its adapter directories, relative to run/
}

// route is where the gateway sends a model's requests.
type route struct {
	c        *container
	upstream string // the model's name inside the engine
}

// New returns a runtime that keeps its files under the store's run/.
func New(cfg Config) *Runtime {
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.DiscardHandler)
	}
	if cfg.StartTimeout <= 0 {
		cfg.StartTimeout = 20 * time.Minute
	}
	engines := map[string][]string{}
	for k, v := range DefaultEngines {
		engines[k] = v
	}
	for k, v := range cfg.Engines {
		if len(v) > 0 {
			engines[k] = v
		}
	}
	cfg.Engines = engines
	return &Runtime{
		cfg: cfg, dir: filepath.Join(cfg.Store.Root(), "run"), storeLock: noLock{},
		engines: map[string]*engine{}, containers: map[string]*container{},
		routes: map[string]route{}, reserved: map[string]bool{}, pins: map[string]int{}, prep: map[string]*sync.Mutex{},
	}
}

// SetStoreLock sets the lock the runtime holds while it reads the store or
// writes to it (a squash, an export), so a prune never runs underneath.
// Call it before the first Run.
func (rt *Runtime) SetStoreLock(l sync.Locker) { rt.storeLock = l }

type noLock struct{}

func (noLock) Lock()   {}
func (noLock) Unlock() {}

// nameForm is what a container name, and so a model name, may look like:
// an image reference fits it.
var nameForm = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,254}$`)

// DetectDevice picks the best device here: cuda when nvidia-smi lists a
// GPU, mps on Apple Silicon, cpu otherwise.
func DetectDevice() string {
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		out, err := exec.Command("nvidia-smi", "-L").Output()
		if err == nil && strings.Contains(string(out), "GPU ") {
			return "cuda"
		}
	}
	if goruntime.GOOS == "darwin" && goruntime.GOARCH == "arm64" {
		return "mps"
	}
	return "cpu"
}

// pick settles the device and engine for a run.
func (rt *Runtime) pick(engineKind, device string) (string, string, error) {
	if device == "" {
		device = rt.cfg.Device
	}
	if device == "" {
		rt.detectOnce.Do(func() { rt.detected = DetectDevice() })
		device = rt.detected
	}
	switch device {
	case "cuda", "mps", "cpu":
	default:
		return "", "", fmt.Errorf("unknown device %q; want cuda, mps or cpu", device)
	}
	switch engineKind {
	case "":
		engineKind = EngineReference
		if device == "cuda" {
			if _, err := exec.LookPath(rt.cfg.Engines[EngineVLLM][0]); err == nil {
				engineKind = EngineVLLM
			}
		}
	case EngineReference:
	case EngineVLLM:
		if device != "cuda" {
			return "", "", fmt.Errorf("vllm serves on NVIDIA GPUs (cuda) here; on %s use the reference engine", device)
		}
	default:
		return "", "", fmt.Errorf("unknown engine %q; want reference or vllm", engineKind)
	}
	return engineKind, device, nil
}

// Run serves the image req.Ref names and returns its container once every
// model it answers to is ready.
func (rt *Runtime) Run(ctx context.Context, req api.RunRequest, progress func(api.Event)) (*api.Container, error) {
	if progress == nil {
		progress = func(api.Event) {}
	}
	name := req.Name
	if name == "" {
		name = req.Ref
	}
	if !nameForm.MatchString(name) {
		return nil, fmt.Errorf("bad container name %q: use letters, digits and . _ : / @ + -, starting with a letter or digit", name)
	}
	kind, device, err := rt.pick(req.Engine, req.Device)
	if err != nil {
		return nil, err
	}
	rt.mu.Lock()
	if rt.reserved[name] || rt.containers[name] != nil {
		rt.mu.Unlock()
		return nil, fmt.Errorf("a container named %q is already running; wpk stop it, or pick another name with --name", name)
	}
	rt.reserved[name] = true
	rt.mu.Unlock()
	defer func() {
		rt.mu.Lock()
		delete(rt.reserved, name)
		rt.mu.Unlock()
	}()

	step := "run " + req.Ref
	digest, cfg, template, err := rt.read(req.Ref)
	if err != nil {
		return nil, err
	}
	p, err := planFor(cfg)
	if err != nil {
		return nil, fmt.Errorf("run %s: %w", req.Ref, err)
	}
	maxRank := 0
	for _, a := range p.adapters {
		maxRank = max(maxRank, a.Rank)
	}
	if kind == EngineVLLM {
		if _, err := vllmRank(maxRank); err != nil {
			return nil, err
		}
	}
	progress(api.Event{Step: step, Message: fmt.Sprintf("serving %s on the %s engine (%s)", p.describe(), kind, device)})

	// Pin the serving files so a prune leaves them alone until the
	// container (or its engine) holds them.
	baseRel := filepath.Join("bases", hexOf(p.base.Chain))
	adapterRels := make([]string, len(p.adapters))
	for i, a := range p.adapters {
		if a.Chain == "" {
			return nil, fmt.Errorf("layer %q has no chain ID; rebuild the image", a.ID)
		}
		adapterRels[i] = filepath.Join("adapters", hexOf(a.Chain)+"-"+a.ID)
	}
	rels := append([]string{baseRel}, adapterRels...)
	rt.pin(rels, 1)
	defer rt.pin(rels, -1)

	if err := rt.prepareBase(ctx, digest, p, baseRel, step, progress); err != nil {
		return nil, err
	}
	for i, a := range p.adapters {
		if err := rt.prepareAdapter(ctx, digest, a, adapterRels[i], progress); err != nil {
			return nil, err
		}
	}

	e, started, err := rt.acquire(p, kind, device, maxRank, filepath.Join(rt.dir, baseRel, "base"))
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			rt.release(e)
		}
	}()
	if started {
		progress(api.Event{Step: step, Message: fmt.Sprintf("starting the %s engine %s; its log is %s", kind, e.id, e.logPath)})
	}
	if err := waitReady(ctx, e, step, progress); err != nil {
		return nil, err
	}
	if e.hasExited() {
		return nil, fmt.Errorf("the %s engine for this base has exited; wpk stop its containers (wpk ps) and run again", kind)
	}
	if kind == EngineVLLM && maxRank > e.maxRank {
		return nil, fmt.Errorf("the vllm engine on this base was started with --max-lora-rank %d, below this image's rank %d; stop its containers (wpk ps) and run again", e.maxRank, maxRank)
	}

	c := &container{engine: e, template: template, files: adapterRels}
	c.info = api.Container{
		Name: name, Image: req.Ref, Digest: digest, Engine: kind, Device: device, EngineID: e.id,
		Base: p.base.Chain, Created: time.Now().UTC().Truncate(time.Second), Endpoint: rt.cfg.Endpoint,
	}
	upstream := map[string]string{}
	if len(p.adapters) == 0 {
		c.info.Models = []string{name}
		upstream[name] = e.served
	}
	for i, a := range p.adapters {
		an := adapterName(a)
		if err := e.loadAdapter(ctx, an, filepath.Join(rt.dir, adapterRels[i], "adapters", a.ID)); err != nil {
			for _, loaded := range c.adapters {
				e.unloadAdapter(loaded)
			}
			return nil, fmt.Errorf("run %s: adapter %s: %w", req.Ref, a.ID, err)
		}
		c.adapters = append(c.adapters, an)
		c.info.Adapters = append(c.info.Adapters, a.ID)
		model := name
		if p.mix != nil {
			model = name + "/" + a.ID
		}
		c.info.Models = append(c.info.Models, model)
		upstream[model] = an
	}

	rt.mu.Lock()
	for _, m := range c.info.Models {
		if r, taken := rt.routes[m]; taken {
			rt.mu.Unlock()
			for _, an := range c.adapters {
				e.unloadAdapter(an)
			}
			return nil, fmt.Errorf("model name %q is taken by container %q; pick another name with --name", m, r.c.info.Name)
		}
	}
	for m, up := range upstream {
		rt.routes[m] = route{c: c, upstream: up}
	}
	rt.containers[name] = c
	rt.mu.Unlock()
	ok = true
	rt.cfg.Logger.Info("container running", "name", name, "image", req.Ref, "engine", e.id, "models", c.info.Models)
	info := c.info
	info.Status = "running"
	return &info, nil
}

// read resolves an image and loads its config and chat template.
func (rt *Runtime) read(ref string) (string, *spec.Config, string, error) {
	rt.storeLock.Lock()
	defer rt.storeLock.Unlock()
	s := rt.cfg.Store
	d, err := s.Resolve(ref)
	if err != nil {
		return "", nil, "", err
	}
	var m ocispec.Manifest
	var cfg spec.Config
	if err := s.ReadJSON(d.Digest.String(), &m); err != nil {
		return "", nil, "", err
	}
	if err := s.ReadJSON(m.Config.Digest.String(), &cfg); err != nil {
		return "", nil, "", err
	}
	if err := spec.CheckImage(&m, &cfg); err != nil {
		return "", nil, "", fmt.Errorf("%s is inconsistent, not running it: %w", ref, err)
	}
	template := ""
	for _, l := range m.Layers {
		if l.Annotations[spec.AnnotationLayer] == "" && l.Annotations[spec.AnnotationTitle] == "chat_template.jinja" {
			raw, err := s.ReadBlob(l.Digest.String())
			if err != nil {
				return "", nil, "", err
			}
			template = string(raw)
		}
	}
	return d.Digest.String(), &cfg, template, nil
}

// lockPrep serializes the preparation of one serving file.
func (rt *Runtime) lockPrep(key string) func() {
	rt.prepMu.Lock()
	mu := rt.prep[key]
	if mu == nil {
		mu = &sync.Mutex{}
		rt.prep[key] = mu
	}
	rt.prepMu.Unlock()
	mu.Lock()
	return mu.Unlock
}

// prepareBase exports the serving base, baking it first if the plan says
// so, unless an earlier run did. Export renames a finished directory into
// place, so one that exists is complete.
func (rt *Runtime) prepareBase(ctx context.Context, digest string, p *plan, rel, step string, progress func(api.Event)) error {
	dir := filepath.Join(rt.dir, rel)
	defer rt.lockPrep(rel)()
	if _, err := os.Stat(dir); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return err
	}
	rt.storeLock.Lock()
	defer rt.storeLock.Unlock()
	src := digest
	if p.through != "" {
		if rt.cfg.Worker == nil {
			return errors.New("this image's serving base has layers baked in, which needs wpkd's build worker; restart wpkd with --worker")
		}
		progress(api.Event{Step: step, Message: fmt.Sprintf("baking the layers through %s into a serving base (once per base)", p.through)})
		res, err := squash.Squash(ctx, rt.cfg.Store, rt.cfg.Worker, digest, squash.Options{Through: p.through}, progress)
		if err != nil {
			return err
		}
		src = res.Manifest.Digest.String()
	}
	progress(api.Event{Step: step, Message: "writing the serving base (once per base)"})
	_, err := export.Export(ctx, rt.cfg.Store, src, export.Options{Dir: dir, Adapters: []string{}}, progress)
	return err
}

// prepareAdapter exports one LoRA layer as a PEFT adapter, unless an
// earlier run did. A layer's chain ID pins its content and parents; its
// id names the directory export writes.
func (rt *Runtime) prepareAdapter(ctx context.Context, digest string, a spec.Layer, rel string, progress func(api.Event)) error {
	dir := filepath.Join(rt.dir, rel)
	defer rt.lockPrep(rel)()
	if _, err := os.Stat(dir); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return err
	}
	rt.storeLock.Lock()
	defer rt.storeLock.Unlock()
	_, err := export.Export(ctx, rt.cfg.Store, digest, export.Options{Dir: dir, NoBase: true, Adapters: []string{a.ID}}, progress)
	return err
}

// pin counts runs that need serving files (delta 1) or no longer do (-1).
func (rt *Runtime) pin(rels []string, delta int) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	for _, r := range rels {
		if rt.pins[r] += delta; rt.pins[r] <= 0 {
			delete(rt.pins, r)
		}
	}
}

// acquire returns the engine for a base on a device, starting one if none
// runs, and counts the caller as one of its users. started says whether
// this call started it.
func (rt *Runtime) acquire(p *plan, kind, device string, maxRank int, modelDir string) (*engine, bool, error) {
	key := p.base.Chain + " " + kind + " " + device
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if e := rt.engines[key]; e != nil && !e.hasExited() {
		e.refs++
		return e, false, nil
	}
	id, err := newID()
	if err != nil {
		return nil, false, err
	}
	if err := os.MkdirAll(filepath.Join(rt.dir, "logs"), 0o755); err != nil {
		return nil, false, err
	}
	e := &engine{
		id: id, key: key, kind: kind, device: device, base: p.base.Chain,
		served: "base-" + short(p.base.Chain), logPath: filepath.Join(rt.dir, "logs", id+".log"),
		ready: make(chan struct{}), exited: make(chan struct{}), abort: make(chan struct{}), adapters: map[string]int{}, refs: 1,
	}
	if kind == EngineVLLM {
		// Headroom for adapters that later images on this base bring.
		e.maxRank, _ = vllmRank(max(maxRank, 64))
	}
	rt.engines[key] = e
	go e.start(rt.cfg.Engines[kind], modelDir, rt.cfg.StartTimeout)
	rt.cfg.Logger.Info("engine starting", "engine", id, "kind", kind, "device", device, "base", p.base.Chain)
	return e, true, nil
}

// release drops one user of an engine and stops it after the last.
func (rt *Runtime) release(e *engine) {
	rt.mu.Lock()
	e.refs--
	last := e.refs == 0
	if last && rt.engines[e.key] == e {
		delete(rt.engines, e.key)
	}
	rt.mu.Unlock()
	if last {
		rt.cfg.Logger.Info("engine stopping", "engine", e.id)
		e.stop()
	}
}

// waitReady waits for an engine to start, telling the caller it is still
// loading every so often.
func waitReady(ctx context.Context, e *engine, step string, progress func(api.Event)) error {
	began := time.Now()
	tick := time.NewTicker(15 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-e.ready:
			return e.startErr
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
			progress(api.Event{Step: step, Message: fmt.Sprintf("the engine is loading the base (%s so far)", time.Since(began).Round(time.Second))})
		}
	}
}

// Stop stops containers by name. An engine stops with its last container.
func (rt *Runtime) Stop(names []string) ([]string, error) {
	rt.mu.Lock()
	var cs []*container
	seen := map[string]bool{}
	for _, n := range names {
		c := rt.containers[n]
		if c == nil {
			rt.mu.Unlock()
			return nil, fmt.Errorf("%w: %s", ErrNoContainer, n)
		}
		// A name given twice stops its container once: releasing it twice
		// would stop an engine other containers still use.
		if !seen[n] {
			seen[n] = true
			cs = append(cs, c)
		}
	}
	for _, c := range cs {
		delete(rt.containers, c.info.Name)
		for _, m := range c.info.Models {
			delete(rt.routes, m)
		}
	}
	rt.mu.Unlock()
	var stopped []string
	for _, c := range cs {
		for _, an := range c.adapters {
			c.engine.unloadAdapter(an)
		}
		rt.release(c.engine)
		stopped = append(stopped, c.info.Name)
		rt.cfg.Logger.Info("container stopped", "name", c.info.Name)
	}
	return stopped, nil
}

// ErrNoContainer is returned for a name no container has.
var ErrNoContainer = errors.New("no such container")

// List returns the containers, sorted by name.
func (rt *Runtime) List() []api.Container {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	out := make([]api.Container, 0, len(rt.containers))
	for _, c := range rt.containers {
		out = append(out, c.status())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (c *container) status() api.Container {
	info := c.info
	info.Status = "running"
	if msg := c.engine.exitError(); msg != "" {
		info.Status, info.Error = "exited", msg
	}
	return info
}

// Logs writes a container's engine log to w. With follow it keeps writing
// new output until ctx is done or the engine exits.
func (rt *Runtime) Logs(ctx context.Context, name string, follow bool, w io.Writer, flush func()) error {
	rt.mu.Lock()
	c := rt.containers[name]
	rt.mu.Unlock()
	if c == nil {
		return fmt.Errorf("%w: %s", ErrNoContainer, name)
	}
	f, err := os.Open(c.engine.logPath)
	if err != nil {
		return err
	}
	defer f.Close()
	for {
		if _, err := io.Copy(w, f); err != nil {
			return err
		}
		flush()
		if !follow {
			return nil
		}
		select {
		case <-ctx.Done():
			return nil
		case <-c.engine.exited:
			_, err := io.Copy(w, f)
			flush()
			return err
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// Close stops every engine, as wpkd exits.
func (rt *Runtime) Close() {
	rt.mu.Lock()
	engines := make([]*engine, 0, len(rt.engines))
	seen := map[*engine]bool{}
	for _, e := range rt.engines {
		engines = append(engines, e)
		seen[e] = true
	}
	for _, c := range rt.containers {
		if !seen[c.engine] {
			engines = append(engines, c.engine)
			seen[c.engine] = true
		}
	}
	rt.engines, rt.containers, rt.routes = map[string]*engine{}, map[string]*container{}, map[string]route{}
	rt.mu.Unlock()
	var wg sync.WaitGroup
	for _, e := range engines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			e.stop()
		}()
	}
	wg.Wait()
}

// PruneFiles removes serving files (bases, adapters, and the logs of
// engines that are gone) that no container, engine or run in progress
// uses, and returns the bytes freed, or that would be with dryRun.
func (rt *Runtime) PruneFiles(dryRun bool) (int64, error) {
	rt.mu.Lock()
	inUse := map[string]bool{}
	for r := range rt.pins {
		inUse[r] = true
	}
	engineFiles := func(e *engine) {
		inUse[filepath.Join("bases", hexOf(e.base))] = true
		inUse[filepath.Join("logs", e.id+".log")] = true
	}
	for _, e := range rt.engines {
		engineFiles(e)
	}
	for _, c := range rt.containers {
		engineFiles(c.engine)
		for _, r := range c.files {
			inUse[r] = true
		}
	}
	rt.mu.Unlock()
	var freed int64
	for _, sub := range []string{"bases", "adapters", "logs"} {
		entries, err := os.ReadDir(filepath.Join(rt.dir, sub))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return freed, err
		}
		for _, ent := range entries {
			rel := filepath.Join(sub, ent.Name())
			// Export stages a directory under a dot name and renames it
			// into place when done; a staging directory belongs to a run.
			if strings.HasPrefix(ent.Name(), ".") || inUse[rel] {
				continue
			}
			p := filepath.Join(rt.dir, rel)
			freed += dirSize(p)
			if !dryRun {
				if err := os.RemoveAll(p); err != nil {
					return freed, err
				}
			}
		}
	}
	return freed, nil
}

// adapterName is a LoRA layer's name inside an engine. It is the same for
// the same layer on the same chain, so containers of one image share it.
func adapterName(a spec.Layer) string { return "lora-" + short(a.Chain) + "-" + a.ID }

func hexOf(d string) string {
	_, h, ok := strings.Cut(d, ":")
	if !ok {
		return d
	}
	return h
}

func newID() (string, error) {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func dirSize(p string) int64 {
	var n int64
	_ = filepath.WalkDir(p, func(_ string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}

// Has reports whether a container has this name.
func (rt *Runtime) Has(name string) bool {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.containers[name] != nil
}

// Endpoint is the gateway's OpenAI base URL.
func (rt *Runtime) Endpoint() string { return rt.cfg.Endpoint }
