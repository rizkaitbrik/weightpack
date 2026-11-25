package control

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
)

// Placement scores. Reusing a running engine beats everything: only the
// adapters load. Holding the image's blobs saves a download; holding its
// base layer saves most of one.
const (
	scoreEngine = 1000
	scoreImage  = 100
	scoreBase   = 10
)

// choose picks the agent to start model on, by base affinity, among agents
// that serve, are up, can run the model's engine and device, have no
// container of that name, have room for an engine unless they already run
// the image's serving base, and are not in avoid.
func choose(states []agentState, m Model, res resolution, avoid map[string]bool) (agentState, error) {
	type candidate struct {
		s     agentState
		score int
	}
	var cands []candidate
	up := 0
	for _, s := range states {
		if !s.Up || !s.Node.Serving {
			continue
		}
		up++
		if _, taken := s.container(m.Name); taken || avoid[s.URL] || !s.canRun(m) {
			continue
		}
		reuse := false
		for _, ct := range s.Containers {
			if ct.Status == "running" && ct.Base == res.ServingBase &&
				(m.Engine == "" || ct.Engine == m.Engine) && (m.Device == "" || ct.Device == m.Device) {
				reuse = true
			}
		}
		if !reuse && s.Node.MaxEngines > 0 && s.Node.Engines >= s.Node.MaxEngines {
			continue
		}
		score := -s.Node.Engines
		if reuse {
			score += scoreEngine
		}
		image, base := false, false
		for _, img := range s.Images {
			image = image || img.Digest == res.Digest
			base = base || (res.Base != "" && img.Base == res.Base)
		}
		if image {
			score += scoreImage
		} else if base {
			score += scoreBase
		}
		cands = append(cands, candidate{s, score})
	}
	if len(cands) == 0 {
		return agentState{}, errNoAgent(m.Name, up)
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		if a, b := len(cands[i].s.Containers), len(cands[j].s.Containers); a != b {
			return a < b
		}
		return cands[i].s.URL < cands[j].s.URL
	})
	return cands[0].s, nil
}

// ensure starts one replica of m and waits for it, or joins a start
// already under way. The start outlives ctx: other requests may wait on
// it.
func (c *Controller) ensure(ctx context.Context, m Model) error {
	c.mu.Lock()
	p := c.placing[m.Name]
	if p == nil {
		p = &placement{done: make(chan struct{})}
		c.placing[m.Name] = p
		go func() {
			p.err = c.place(context.Background(), m, "")
			c.mu.Lock()
			delete(c.placing, m.Name)
			c.mu.Unlock()
			close(p.done)
		}()
	}
	c.mu.Unlock()
	select {
	case <-p.done:
		return p.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// startAsync starts a placement for the reconciler unless one runs.
func (c *Controller) startAsync(m Model, on string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.placing[m.Name] != nil {
		return
	}
	p := &placement{done: make(chan struct{})}
	c.placing[m.Name] = p
	go func() {
		p.err = c.place(context.Background(), m, on)
		c.mu.Lock()
		delete(c.placing, m.Name)
		c.mu.Unlock()
		close(p.done)
	}()
}

// placeTimeout bounds a pull and an engine start, model load included.
const placeTimeout = 30 * time.Minute

// place starts one replica of m: on the agent at on when it is set, else
// on the one choose picks. The agent pulls the image (under its trust
// policy) and runs it.
func (c *Controller) place(ctx context.Context, m Model, on string) (err error) {
	ctx, cancel := context.WithTimeout(ctx, placeTimeout)
	defer cancel()
	start := time.Now()
	defer func() {
		c.setStatus(m.Name, err)
		c.metrics.placement(m.Name, err)
		if err != nil {
			c.cfg.Logger.Error("placement failed", "model", m.Name, "image", m.Image, "err", err)
		}
	}()
	res, err := c.resolve(ctx, m.Image)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", m.Image, err)
	}
	var target agentState
	if on != "" {
		a := c.agentByURL(on)
		if a == nil {
			return fmt.Errorf("agent %s is gone", on)
		}
		target = a.snapshot()
	} else if target, err = choose(c.states(), m, res, c.recentFailures(m.Name)); err != nil {
		return err
	}
	a := c.agentByURL(target.URL)
	if a == nil {
		return fmt.Errorf("agent %s is gone", target.URL)
	}
	// A node that fails to place the model is passed over for a while, so
	// the next try goes elsewhere.
	defer func() {
		if err != nil {
			c.placementFailed(m.Name, target.URL)
		}
	}()
	c.cfg.Logger.Info("placing", "model", m.Name, "image", m.Image, "digest", res.Digest, "node", target.Node.Name, "agent", a.url)
	// The run resolves m.Image in the node's store, so that name must
	// point at the digest: holding it under another name, or an older
	// digest under this one, is not enough. A pull of blobs the node holds
	// moves only the manifest and retags.
	if !current(target.Images, m.Image, res.Digest) {
		if _, err := a.client.Pull(ctx, api.PullRequest{Ref: m.Image}, nil); err != nil {
			return fmt.Errorf("node %s: %w", target.Node.Name, err)
		}
	}
	if _, err := a.client.Run(ctx, api.RunRequest{Ref: m.Image, Name: m.Name, Engine: m.Engine, Device: m.Device, Owner: c.cfg.Owner}, nil); err != nil {
		return fmt.Errorf("node %s: %w", target.Node.Name, err)
	}
	a.refresh(ctx)
	c.cfg.Logger.Info("placed", "model", m.Name, "node", target.Node.Name, "took", time.Since(start).Round(time.Second))
	return nil
}

// failureCooldown is how long a node that failed to place a model is
// passed over for it.
const failureCooldown = 2 * time.Minute

func (c *Controller) placementFailed(model, agentURL string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.failed[model] == nil {
		c.failed[model] = map[string]time.Time{}
	}
	c.failed[model][agentURL] = time.Now()
}

// recentFailures are the agents that failed to place model lately.
func (c *Controller) recentFailures(model string) map[string]bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := map[string]bool{}
	for u, at := range c.failed[model] {
		if time.Since(at) < failureCooldown {
			out[u] = true
		}
	}
	return out
}

// current reports whether a node's store resolves image to digest: by its
// own digest when image names one, else by its tag (latest when it has
// none).
func current(images []api.Image, image, digest string) bool {
	r, err := reference.Parse(image)
	if err != nil {
		return false
	}
	if r.Digest != "" {
		digest = r.Digest
	} else if r.Tag == "" {
		r.Tag = reference.DefaultTag
	}
	for _, img := range images {
		if img.Digest == digest && (r.Digest != "" || img.Ref == r.String()) {
			return true
		}
	}
	return false
}

// fresh reports whether a running replica is what the catalogue asks for:
// its image reference at the digest it resolves to now, on the engine
// and device the model pins, if it pins them.
func fresh(ct api.Container, m Model, digest string) bool {
	return ct.Image == m.Image && ct.Digest == digest &&
		(m.Engine == "" || ct.Engine == m.Engine) && (m.Device == "" || ct.Device == m.Device)
}

// reconcile moves each catalogue model one step toward its replicas:
// clears exited containers, adds a replica, replaces one that runs an old
// digest, or removes one too many. It stops containers it started for
// models no longer in the catalogue.
func (c *Controller) reconcile(ctx context.Context) {
	cat := c.currentCatalogue()
	states := c.states()
	wanted := map[string]bool{}
	for _, m := range cat.Models {
		wanted[m.Name] = true
	}
	// Containers of models dropped from the catalogue, and exited ones.
	for _, s := range states {
		if !s.Up {
			continue
		}
		var stop []string
		for _, ct := range s.Containers {
			if ct.Owner == c.cfg.Owner && (!wanted[ct.Name] || ct.Status != "running") {
				stop = append(stop, ct.Name)
			}
		}
		if len(stop) > 0 {
			c.stop(ctx, s.URL, stop...)
		}
	}
	for _, m := range cat.Models {
		c.reconcileModel(ctx, m, states)
	}
}

func (c *Controller) reconcileModel(ctx context.Context, m Model, states []agentState) {
	c.mu.Lock()
	busy := c.placing[m.Name] != nil
	c.mu.Unlock()
	if busy {
		return
	}
	var running []agentState
	for _, s := range states {
		if ct, ok := s.container(m.Name); ok && s.Up && ct.Owner == c.cfg.Owner && ct.Status == "running" {
			running = append(running, s)
		}
	}
	if len(running) == 0 && m.Replicas == 0 {
		return // on demand: placed on its first request
	}
	res, err := c.resolve(ctx, m.Image)
	if err != nil {
		c.setStatus(m.Name, fmt.Errorf("resolve %s: %w", m.Image, err))
		return
	}
	// A replica is stale when the catalogue names another image, engine or
	// device, or its tag has moved to another digest.
	var good, stale []agentState
	for _, s := range running {
		if ct, _ := s.container(m.Name); fresh(ct, m, res.Digest) {
			good = append(good, s)
		} else {
			stale = append(stale, s)
		}
	}
	target := m.Replicas
	if target == 0 {
		target = len(running) // on demand: keep as many as run
	}
	switch {
	case len(good) < target:
		_, err := choose(states, m, res, c.recentFailures(m.Name))
		switch {
		case err == nil:
			c.startAsync(m, "")
		case len(stale) > 0:
			// No room elsewhere: replace a stale replica where it runs.
			c.stop(ctx, stale[0].URL, m.Name)
			c.startAsync(m, stale[0].URL)
		default:
			c.setStatus(m.Name, err)
		}
	case len(stale) > 0:
		c.stop(ctx, stale[0].URL, m.Name)
	case len(good) > target:
		sort.Slice(good, func(i, j int) bool { return good[i].Inflight < good[j].Inflight })
		c.stop(ctx, good[0].URL, m.Name)
	}
}

// stop stops containers on an agent and refreshes it.
func (c *Controller) stop(ctx context.Context, agentURL string, names ...string) {
	a := c.agentByURL(agentURL)
	if a == nil {
		return
	}
	if _, err := a.client.Stop(ctx, api.StopRequest{Names: names}); err != nil {
		c.cfg.Logger.Warn("stop failed", "agent", agentURL, "containers", names, "err", err)
		return
	}
	c.cfg.Logger.Info("stopped", "agent", agentURL, "containers", names)
	a.refresh(ctx)
}
