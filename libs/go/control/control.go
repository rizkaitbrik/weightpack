// Package control is Weightpack's cluster control plane: an OpenAI gateway
// in front of the node agents (wpkd on each GPU node), and a scheduler
// that keeps the catalogue's images running on them.
//
// The scheduler's rule is base affinity: an image goes to a node already
// serving its base (only its adapters load), else to one whose store holds
// its blobs, else to the least loaded node. Requests for a model go to the
// node serving it with the fewest requests in flight; a model in the
// catalogue that runs nowhere is placed on its first request.
//
// The control plane keeps no state of its own. Every pass it finds the
// agents (a headless Service's DNS name in Kubernetes), reads their
// containers, images and load, rereads the catalogue, and moves the
// cluster one step toward it.
package control

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

// DefaultOwner marks the containers a control plane starts; it stops only
// those.
const DefaultOwner = "wpk-control"

// Config sets up a Controller.
type Config struct {
	// Discovery finds the node agents.
	Discovery Discovery
	// AgentKey is the bearer token of the agents' API (wpkd --api-key) and
	// of their gateways (wpkd --gateway-key): one secret for both.
	AgentKey string
	// Catalogue is the path of the catalogue file.
	Catalogue string
	// APIKey, when set, is the bearer token clients must send.
	APIKey string
	// Owner marks the containers this control plane starts; "" means
	// DefaultOwner.
	Owner string
	// Interval is the time between passes; 0 means 5 seconds.
	Interval time.Duration
	// Client talks to the agents' APIs and gateways; nil means a default.
	Client *http.Client
	Logger *slog.Logger
}

// Controller is the control plane.
type Controller struct {
	cfg     Config
	handler http.Handler

	mu        sync.Mutex
	agents    map[string]*agent // by URL
	catalogue *Catalogue        // the last catalogue that loaded
	catErr    string
	resolved  map[string]resolution           // by image reference
	placing   map[string]*placement           // by model name
	status    map[string]string               // by model name: the last placement error
	failed    map[string]map[string]time.Time // by model, then agent: when a placement failed
	next      int                             // round-robin among equally loaded agents

	metrics metrics
}

// resolution is what an agent said about an image: its digest and bases.
type resolution struct {
	api.ResolveResponse
	at time.Time
}

// resolveTTL is how long a resolution holds; a tag that moves is noticed
// within it.
const resolveTTL = 30 * time.Second

// placement is a placement in progress; done closes when it ends.
type placement struct {
	done chan struct{}
	err  error
}

// New returns a control plane. Call Run to start its passes.
func New(cfg Config) *Controller {
	if cfg.Owner == "" {
		cfg.Owner = DefaultOwner
	}
	if cfg.Interval <= 0 {
		cfg.Interval = 5 * time.Second
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{Transport: &http.Transport{
			ResponseHeaderTimeout: 30 * time.Minute, // generation can take minutes
			MaxIdleConnsPerHost:   64,
		}}
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.DiscardHandler)
	}
	c := &Controller{
		cfg: cfg, agents: map[string]*agent{}, resolved: map[string]resolution{},
		placing: map[string]*placement{}, status: map[string]string{}, failed: map[string]map[string]time.Time{},
		metrics: newMetrics(),
	}
	c.handler = c.routes()
	return c
}

// ServeHTTP serves the OpenAI gateway and the control plane's status.
func (c *Controller) ServeHTTP(w http.ResponseWriter, r *http.Request) { c.handler.ServeHTTP(w, r) }

// Run makes a pass every Interval until ctx is done.
func (c *Controller) Run(ctx context.Context) {
	t := time.NewTicker(c.cfg.Interval)
	defer t.Stop()
	for {
		c.Sync(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// Sync makes one pass: it finds the agents, refreshes them, rereads the
// catalogue and starts what the catalogue lacks.
func (c *Controller) Sync(ctx context.Context) {
	c.discover(ctx)
	c.refreshAll(ctx)
	c.loadCatalogue()
	c.reconcile(ctx)
}

func (c *Controller) discover(ctx context.Context) {
	urls, err := c.cfg.Discovery.Agents(ctx)
	if err != nil {
		c.cfg.Logger.Warn("agent discovery failed", "err", err)
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	keep := map[string]bool{}
	for _, u := range urls {
		keep[u] = true
		if c.agents[u] == nil {
			c.agents[u] = &agent{url: u, client: api.NewRemoteClient(u, c.cfg.AgentKey, c.cfg.Client)}
			c.cfg.Logger.Info("agent found", "url", u)
		}
	}
	for u := range c.agents {
		if !keep[u] {
			delete(c.agents, u)
			c.cfg.Logger.Info("agent gone", "url", u)
		}
	}
}

func (c *Controller) refreshAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, a := range c.agentList() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a.refresh(ctx)
		}()
	}
	wg.Wait()
}

func (c *Controller) agentList() []*agent {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]*agent, 0, len(c.agents))
	for _, a := range c.agents {
		out = append(out, a)
	}
	return out
}

func (c *Controller) states() []agentState {
	var out []agentState
	for _, a := range c.agentList() {
		out = append(out, a.snapshot())
	}
	return out
}

func (c *Controller) loadCatalogue() {
	cat, err := LoadCatalogue(c.cfg.Catalogue)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		// Keep serving the last good catalogue.
		if c.catErr != err.Error() {
			c.cfg.Logger.Error("catalogue not loaded", "err", err)
		}
		c.catErr = err.Error()
		return
	}
	c.catalogue, c.catErr = cat, ""
}

func (c *Controller) currentCatalogue() *Catalogue {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.catalogue == nil {
		return &Catalogue{}
	}
	return c.catalogue
}

// resolve asks an agent which digest an image's reference names and what
// base it serves on. The agent checks its trust policy first, so an image
// the nodes would refuse is never placed.
func (c *Controller) resolve(ctx context.Context, ref string) (resolution, error) {
	c.mu.Lock()
	r, ok := c.resolved[ref]
	c.mu.Unlock()
	if ok && time.Since(r.at) < resolveTTL {
		return r, nil
	}
	var errs []error
	for _, a := range c.agentList() {
		if !a.snapshot().Up {
			continue
		}
		res, err := a.client.Resolve(ctx, api.ResolveRequest{Ref: ref, Placement: true})
		if err != nil {
			errs = append(errs, err)
			// The policy and the image are the same on every node; one
			// refusal is enough.
			break
		}
		r = resolution{ResolveResponse: *res, at: time.Now()}
		c.mu.Lock()
		c.resolved[ref] = r
		c.mu.Unlock()
		return r, nil
	}
	if len(errs) == 0 {
		return resolution{}, errors.New("no agent is up")
	}
	return resolution{}, errors.Join(errs...)
}

// setStatus records the outcome of a model's last placement.
func (c *Controller) setStatus(model string, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err == nil {
		delete(c.status, model)
		return
	}
	c.status[model] = err.Error()
}

func (c *Controller) agentByURL(u string) *agent {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.agents[u]
}

// errNoAgent says why no agent could take a model.
func errNoAgent(model string, up int) error {
	if up == 0 {
		return fmt.Errorf("cannot place %s: no node agent is up", model)
	}
	return fmt.Errorf("cannot place %s: each of the %d nodes up serves it already, lacks its engine or device, runs its most engines (wpkd --max-engines), or just failed to start it", model, up)
}
