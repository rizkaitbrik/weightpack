package daemon

import (
	"bytes"
	"context"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ggcr "github.com/google/go-containerregistry/pkg/registry"
	"github.com/opencontainers/go-digest"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/serving"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/trust"
)

// serveUnix starts the handler on a Unix socket and returns a client for it.
func serveUnix(t *testing.T, s *store.Store, opts ...Option) *api.Client {
	t.Helper()
	dir, err := os.MkdirTemp("", "wpkd") // short path: socket paths are limited
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	socket := filepath.Join(dir, "s")
	ln, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(NewHandler(s, "test", nil, opts...))
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)
	return api.NewClient(socket)
}

func TestBuildImagesInspect(t *testing.T) {
	dir := t.TempDir()
	wf := buildtest.Project(t, dir)
	s, err := store.Open(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	c := serveUnix(t, s)
	ctx := context.Background()

	var events []api.Event
	res, err := c.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"assistant:v1"}}, func(e api.Event) {
		events = append(events, e)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) == 0 || res.Manifest.Digest == "" || len(res.Tags) != 1 {
		t.Fatalf("result %+v after %d events", res, len(events))
	}

	images, err := c.Images(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(images) != 1 || images[0].Ref != "assistant:v1" || images[0].Digest != res.Manifest.Digest.String() ||
		images[0].Size != res.Size || images[0].Layers != 5 || images[0].Arch != "llama" {
		t.Errorf("images = %+v, build result size %d", images, res.Size)
	}

	detail, err := c.Inspect(ctx, "assistant:v1")
	if err != nil {
		t.Fatal(err)
	}
	if err := spec.CheckImage(&detail.Manifest, &detail.Config); err != nil {
		t.Errorf("inspected image fails CheckImage: %v", err)
	}
	if _, err := c.Inspect(ctx, "nothing:here"); err == nil || !strings.Contains(err.Error(), "no such image") {
		t.Errorf("inspect unknown = %v", err)
	}
}

func TestBuildErrors(t *testing.T) {
	dir := t.TempDir()
	s, _ := store.Open(filepath.Join(dir, "store"))
	c := serveUnix(t, s)
	ctx := context.Background()

	if _, err := c.Build(ctx, api.BuildRequest{Weightfile: "Weightfile"}, nil); err == nil || !strings.Contains(err.Error(), "absolute") {
		t.Errorf("relative path: %v", err)
	}
	// A build that fails after the stream started reports its error as the
	// last event.
	wf := filepath.Join(dir, "Weightfile")
	_ = os.WriteFile(wf, []byte("FROM ./missing\n"), 0o644)
	if _, err := c.Build(ctx, api.BuildRequest{Weightfile: wf}, nil); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Errorf("failing build: %v", err)
	}
}

func TestBlobLookup(t *testing.T) {
	s, _ := store.Open(t.TempDir())
	d, _, err := s.Put(bytes.NewReader([]byte("blob")), "")
	if err != nil {
		t.Fatal(err)
	}
	h := NewHandler(s, "test", nil)
	for path, want := range map[string]int{
		"/v1/blobs/" + d: http.StatusNoContent,
		"/v1/blobs/sha256:" + strings.Repeat("0", 64): http.StatusNotFound,
		"/v1/blobs/bogus": http.StatusBadRequest,
		"/healthz":        http.StatusOK,
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != want {
			t.Errorf("GET %s: status %d, want %d", path, rec.Code, want)
		}
	}
}

func TestClientWithoutDaemon(t *testing.T) {
	c := api.NewClient(filepath.Join(t.TempDir(), "absent.sock"))
	if _, err := c.Images(context.Background()); err == nil || !strings.Contains(err.Error(), "is the daemon running") {
		t.Errorf("err = %v", err)
	}
}

func TestTagPushPull(t *testing.T) {
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	reg := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(reg.Close)
	host := strings.TrimPrefix(reg.URL, "http://")

	dir := t.TempDir()
	wf := buildtest.Project(t, dir)
	a, _ := store.Open(filepath.Join(dir, "a"))
	b, _ := store.Open(filepath.Join(dir, "b"))
	ca, cb := serveUnix(t, a), serveUnix(t, b)
	ctx := context.Background()

	built, err := ca.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"assistant"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	remoteRef := host + "/team/assistant:v1"
	tagged, err := ca.Tag(ctx, api.TagRequest{Source: "assistant", Target: remoteRef})
	if err != nil || tagged.Ref != remoteRef || tagged.Digest != built.Manifest.Digest.String() {
		t.Fatalf("tag = %+v, %v", tagged, err)
	}
	if _, err := ca.Tag(ctx, api.TagRequest{Source: "absent", Target: "x"}); err == nil || !strings.Contains(err.Error(), "no such image") {
		t.Errorf("tag of a missing image: %v", err)
	}

	var events []api.Event
	pushed, err := ca.Push(ctx, api.PushRequest{Ref: remoteRef}, func(e api.Event) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	if pushed.Moved != pushed.Blobs || len(events) == 0 {
		t.Errorf("push = %+v after %d events", pushed, len(events))
	}
	if _, err := ca.Push(ctx, api.PushRequest{Ref: "assistant"}, nil); err == nil || !strings.Contains(err.Error(), "names no registry") {
		t.Errorf("push without a registry: %v", err)
	}

	pulled, err := cb.Pull(ctx, api.PullRequest{Ref: remoteRef, Exact: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pulled.Digest != built.Manifest.Digest.String() || pulled.Tag != remoteRef {
		t.Errorf("pull = %+v", pulled)
	}
	detail, err := cb.Inspect(ctx, remoteRef)
	if err != nil {
		t.Fatal(err)
	}
	if err := spec.CheckImage(&detail.Manifest, &detail.Config); err != nil {
		t.Errorf("pulled image: %v", err)
	}
}

func TestBuildPullsImagesItLacks(t *testing.T) {
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	reg := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(reg.Close)
	host := strings.TrimPrefix(reg.URL, "http://")

	dir := t.TempDir()
	buildtest.Project(t, dir)
	a, _ := store.Open(filepath.Join(dir, "a"))
	b, _ := store.Open(filepath.Join(dir, "b"))
	ca, cb := serveUnix(t, a), serveUnix(t, b)
	ctx := context.Background()

	baseWF := filepath.Join(dir, "Base.Weightfile")
	_ = os.WriteFile(baseWF, []byte("FROM ./base\n"), 0o644)
	ref := host + "/team/base:v1"
	if _, err := ca.Build(ctx, api.BuildRequest{Weightfile: baseWF, Tags: []string{ref}}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := ca.Push(ctx, api.PushRequest{Ref: ref}, nil); err != nil {
		t.Fatal(err)
	}

	wf := filepath.Join(dir, "Child.Weightfile")
	_ = os.WriteFile(wf, []byte("FROM "+ref+"\nLAYER french ./adapters/french\n"), 0o644)
	var events []api.Event
	res, err := cb.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"child"}}, func(e api.Event) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Config.Layers) != 2 {
		t.Errorf("child layers = %+v", res.Config.Layers)
	}
	var pulled bool
	for _, e := range events {
		pulled = pulled || strings.HasPrefix(e.Step, "pull ")
	}
	if !pulled {
		t.Errorf("no pull progress in the build stream: %+v", events)
	}
}

func TestServingEndpoints(t *testing.T) {
	dir := t.TempDir()
	wf := buildtest.Project(t, dir)
	s, err := store.Open(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Without a runtime, serving is off.
	off := serveUnix(t, s)
	if _, err := off.Containers(ctx); err == nil || !strings.Contains(err.Error(), "serving is off") {
		t.Errorf("containers without a runtime: %v", err)
	}

	// An engine command that exits at once: the run fails with its output.
	rt := serving.New(serving.Config{
		Store: s, Device: "cpu",
		Engines: map[string][]string{serving.EngineReference: {"/bin/sh", "-c", "echo engine broke >&2; exit 3", "sh"}},
	})
	t.Cleanup(rt.Close)
	c := serveUnix(t, s, WithRuntime(rt))
	if err := os.WriteFile(wf, []byte("FROM ./base\nLAYER french ./adapters/french\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"french"}}, nil); err != nil {
		t.Fatal(err)
	}
	var steps []string
	_, err = c.Run(ctx, api.RunRequest{Ref: "french"}, func(e api.Event) { steps = append(steps, e.Message) })
	if err == nil || !strings.Contains(err.Error(), "engine broke") {
		t.Errorf("run with a broken engine: %v", err)
	}
	if len(steps) == 0 || !strings.Contains(steps[0], "french as an adapter") {
		t.Errorf("progress = %q", steps)
	}
	if _, err := c.Run(ctx, api.RunRequest{Ref: "missing"}, nil); err == nil {
		t.Error("running an unknown image succeeded")
	}
	if _, err := c.Run(ctx, api.RunRequest{Ref: "french", Device: "tpu"}, nil); err == nil || !strings.Contains(err.Error(), "unknown device") {
		t.Errorf("bad device: %v", err)
	}
	if cs, err := c.Containers(ctx); err != nil || len(cs) != 0 {
		t.Errorf("containers = %v, %v", cs, err)
	}
	if _, err := c.Stop(ctx, api.StopRequest{Names: []string{"nope"}}); err == nil || !strings.Contains(err.Error(), "no such container") {
		t.Errorf("stop unknown: %v", err)
	}
	if _, err := c.Logs(ctx, "nope", false); err == nil || !strings.Contains(err.Error(), "no such container") {
		t.Errorf("logs unknown: %v", err)
	}
	// The failed run left a serving base behind, which prune removes.
	res, err := c.Prune(ctx, api.PruneRequest{})
	if err != nil || res.ServingBytes == 0 {
		t.Errorf("prune: %+v, %v", res, err)
	}
}

func TestDoctor(t *testing.T) {
	s, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	rt := serving.New(serving.Config{
		Store: s, Device: "cpu", Endpoint: "http://127.0.0.1:8090/v1",
		Engines: map[string][]string{serving.EngineReference: {"no-such-engine-binary"}},
	})
	c := serveUnix(t, s, WithRuntime(rt))
	report, err := c.Doctor(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]api.Check{}
	for _, ch := range report.Checks {
		got[ch.Name] = ch
	}
	for name, want := range map[string]string{
		"wpkd": api.CheckOK, "build worker": api.CheckFail, "device": api.CheckOK,
		"reference engine": api.CheckFail, "vllm": api.CheckSkip, "gateway": api.CheckOK,
		"store": "", "trust policy": api.CheckOK, "hugging face": api.CheckSkip,
	} {
		ch, ok := got[name]
		if !ok {
			t.Errorf("no %s check in %+v", name, report.Checks)
			continue
		}
		if want != "" && ch.Status != want {
			t.Errorf("%s: %s (%s), want %s", name, ch.Status, ch.Detail, want)
		}
		if (ch.Status == api.CheckFail || ch.Status == api.CheckWarn) && ch.Fix == "" {
			t.Errorf("%s fails without a fix", name)
		}
	}
	if !strings.Contains(got["device"].Detail, "cpu (set with wpkd --device)") {
		t.Errorf("device: %s", got["device"].Detail)
	}

	// A policy file wpkd cannot read fails every pull, so doctor says so.
	_ = os.WriteFile(filepath.Join(s.Root(), "policy.json"), []byte(`{"rules": [{"scope": "ghcr.io", "action": "maybe"}]}`), 0o644)
	report, err = c.Doctor(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, ch := range report.Checks {
		if ch.Name == "trust policy" && (ch.Status != api.CheckFail || !strings.Contains(ch.Detail, "action must be")) {
			t.Errorf("broken policy: %+v", ch)
		}
	}
}

func TestTrustPolicy(t *testing.T) {
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	reg := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(reg.Close)
	host := strings.TrimPrefix(reg.URL, "http://")

	dir := t.TempDir()
	buildtest.Project(t, dir)
	a, _ := store.Open(filepath.Join(dir, "a"))
	b, _ := store.Open(filepath.Join(dir, "b"))
	ca, cb := serveUnix(t, a), serveUnix(t, b)
	ctx := context.Background()

	baseWF := filepath.Join(dir, "Base.Weightfile")
	_ = os.WriteFile(baseWF, []byte("FROM ./base\n"), 0o644)
	ref := host + "/team/base:v1"
	if _, err := ca.Build(ctx, api.BuildRequest{Weightfile: baseWF, Tags: []string{ref}}, nil); err != nil {
		t.Fatal(err)
	}
	pushed, err := ca.Push(ctx, api.PushRequest{Ref: ref}, nil)
	if err != nil {
		t.Fatal(err)
	}

	priv, pub, err := trust.GenerateKeyPair(nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cb.SetTrust(ctx, api.TrustPolicy{Rules: []api.TrustRule{{Scope: host, Action: api.TrustSigned}}}); err == nil || !strings.Contains(err.Error(), "at least one trusted key") {
		t.Errorf("a signed rule without signers: %v", err)
	}
	stored, err := cb.SetTrust(ctx, api.TrustPolicy{Rules: []api.TrustRule{{Scope: host + "/team/", Action: api.TrustSigned, Keys: []string{string(pub)}}}})
	if err != nil || stored.Rules[0].Scope != host+"/team" {
		t.Fatalf("set trust = %+v, %v", stored, err)
	}
	if got, err := cb.Trust(ctx); err != nil || len(got.Rules) != 1 {
		t.Errorf("trust = %+v, %v", got, err)
	}

	// A build FROM the unsigned image is refused before any tensor moves.
	wf := filepath.Join(dir, "Child.Weightfile")
	_ = os.WriteFile(wf, []byte("FROM "+ref+"\nLAYER french ./adapters/french\n"), 0o644)
	if _, err := cb.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"child"}}, nil); err == nil || !strings.Contains(err.Error(), "no Sigstore signature") {
		t.Fatalf("build from an unsigned image: %v", err)
	}
	if images, _ := cb.Images(ctx); len(images) != 0 {
		t.Errorf("the refused pull stored %+v", images)
	}

	signer, err := trust.NewKeySigner(priv, nil)
	if err != nil {
		t.Fatal(err)
	}
	other, _ := signer.Sign(ctx, digest.FromString("another manifest"))
	if _, err := cb.Attach(ctx, api.AttachRequest{Ref: ref, Digest: pushed.Digest, Bundle: other}); err == nil || !strings.Contains(err.Error(), "does not sign") {
		t.Errorf("attach of a bundle for another manifest: %v", err)
	}
	bundle, _ := signer.Sign(ctx, digest.Digest(pushed.Digest))
	if _, err := cb.Attach(ctx, api.AttachRequest{Ref: ref, Digest: pushed.Digest, Bundle: bundle}); err != nil {
		t.Fatal(err)
	}
	if _, err := cb.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"child"}}, nil); err != nil {
		t.Fatalf("build from a signed image: %v", err)
	}
	res, err := cb.Verify(ctx, api.VerifyRequest{Ref: ref})
	if err != nil || res.Rule != host+"/team" || res.Signatures != 1 || !strings.HasPrefix(res.Signer, "key SHA256:") {
		t.Errorf("verify = %+v, %v", res, err)
	}
}

func TestPlacementResolveAndNode(t *testing.T) {
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	reg := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(reg.Close)
	host := strings.TrimPrefix(reg.URL, "http://")

	dir := t.TempDir()
	buildtest.Project(t, dir)
	s, _ := store.Open(filepath.Join(dir, "a"))
	policy := filepath.Join(dir, "policy.json")
	c := serveUnix(t, s, WithNode("gpu-3", 8090), WithPolicyFile(policy))
	ctx := context.Background()

	ref := host + "/team/assistant:v1"
	built, err := c.Build(ctx, api.BuildRequest{Weightfile: filepath.Join(dir, "Weightfile"), Tags: []string{ref}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Push(ctx, api.PushRequest{Ref: ref}, nil); err != nil {
		t.Fatal(err)
	}
	images, err := c.Images(ctx)
	if err != nil || len(images) != 1 || images[0].Base != built.Config.Layers[0].Chain {
		t.Fatalf("images = %+v, %v", images, err)
	}

	res, err := c.Resolve(ctx, api.ResolveRequest{Ref: ref, Placement: true})
	if err != nil {
		t.Fatal(err)
	}
	// The assistant's experts sit on French, so French is baked into the
	// serving base, which is not the base layer.
	if res.Digest != built.Manifest.Digest.String() || res.Base != built.Config.Layers[0].Chain ||
		res.ServingBase == "" || res.ServingBase == res.Base || res.Bytes <= built.Manifest.Size {
		t.Errorf("placement resolve = %+v", res)
	}
	if plain, err := c.Resolve(ctx, api.ResolveRequest{Ref: ref}); err != nil || plain.ServingBase != "" {
		t.Errorf("plain resolve = %+v, %v", plain, err)
	}

	// The policy file the daemon was given applies, as a pull's would.
	_ = os.WriteFile(policy, []byte(`{"rules": [{"scope": "*", "action": "reject"}]}`), 0o644)
	if _, err := c.Resolve(ctx, api.ResolveRequest{Ref: ref, Placement: true}); err == nil || !strings.Contains(err.Error(), "rejects") {
		t.Errorf("placement under a reject rule: %v", err)
	}
	if got, err := c.Trust(ctx); err != nil || len(got.Rules) != 1 {
		t.Errorf("trust from the policy file = %+v, %v", got, err)
	}

	node, err := c.Node(ctx)
	if err != nil || node.Name != "gpu-3" || node.Serving || node.GatewayPort != 8090 {
		t.Errorf("node = %+v, %v", node, err)
	}

	// Over TCP the API needs the key.
	h := RequireKey(NewHandler(s, "test", nil), "k")
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	if _, err := api.NewRemoteClient(srv.URL, "k", nil).Images(ctx); err != nil {
		t.Errorf("with the key: %v", err)
	}
	if _, err := api.NewRemoteClient(srv.URL, "", nil).Images(ctx); err == nil || !strings.Contains(err.Error(), "API key") {
		t.Errorf("without the key: %v", err)
	}
	// Kubernetes probes cannot send the key; /healthz needs none.
	if resp, err := http.Get(srv.URL + "/healthz"); err != nil || resp.StatusCode != http.StatusOK {
		t.Errorf("health probe without the key: %v, %v", resp, err)
	} else {
		resp.Body.Close()
	}
}
