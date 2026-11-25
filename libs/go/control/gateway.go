package control

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// maxBody bounds a request the gateway reads to route it.
const maxBody = 64 << 20

// routes is the control plane's HTTP API:
//
//	GET  /v1/models            every model running, and the catalogue's
//	POST /v1/chat/completions  routed by the body's model field
//	POST /v1/completions       routed by the body's model field
//	POST /v1/embeddings        routed by the body's model field
//	GET  /v1/cluster           agents, containers and models, as JSON
//	GET  /metrics              Prometheus metrics
//	GET  /healthz              200 once a catalogue has loaded
func (c *Controller) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", c.health)
	mux.HandleFunc("GET /metrics", c.serveMetrics)
	mux.HandleFunc("GET /v1/models", c.authorized(c.models))
	mux.HandleFunc("GET /v1/cluster", c.authorized(c.cluster))
	for _, p := range []string{"/v1/chat/completions", "/v1/completions", "/v1/embeddings"} {
		mux.HandleFunc("POST "+p, c.authorized(c.proxy))
	}
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		oaiError(w, http.StatusNotFound, "not_found", fmt.Sprintf("%s %s is not part of the API wpk-control serves", r.Method, r.URL.Path))
	})
	return mux
}

func (c *Controller) authorized(h http.HandlerFunc) http.HandlerFunc {
	if c.cfg.APIKey == "" {
		return h
	}
	want := []byte("Bearer " + c.cfg.APIKey)
	return func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			oaiError(w, http.StatusUnauthorized, "invalid_api_key", "missing or wrong API key; send it as Authorization: Bearer <key>")
			return
		}
		h(w, r)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// oaiError writes an error the way the OpenAI API does.
func oaiError(w http.ResponseWriter, status int, code, msg string) {
	typ := "invalid_request_error"
	if status >= 500 {
		typ = "server_error"
	}
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg, "type": typ, "code": code}})
}

func (c *Controller) health(w http.ResponseWriter, _ *http.Request) {
	c.mu.Lock()
	loaded, catErr := c.catalogue != nil, c.catErr
	c.mu.Unlock()
	if !loaded {
		msg := "no catalogue loaded yet"
		if catErr != "" {
			msg = catErr
		}
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "starting", "error": msg})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (c *Controller) models(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	seen := map[string]bool{}
	data := []model{}
	add := func(id string, created time.Time) {
		if !seen[id] {
			seen[id] = true
			data = append(data, model{ID: id, Object: "model", Created: created.Unix(), OwnedBy: "weightpack"})
		}
	}
	for _, s := range c.states() {
		for _, ct := range s.Containers {
			if s.Up && ct.Status == "running" {
				for _, m := range ct.Models {
					add(m, ct.Created)
				}
			}
		}
	}
	for _, m := range c.currentCatalogue().Models {
		add(m.Name, time.Time{}) // starts on its first request
	}
	sort.Slice(data, func(i, j int) bool { return data[i].ID < data[j].ID })
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// pick returns the agent to send a model's request to: among those serving
// it, the one with the fewest requests in flight, taking turns on ties.
func (c *Controller) pick(model string, avoid map[string]bool) *agent {
	var best []*agent
	var bestLoad int64
	for _, a := range c.agentList() {
		s := a.snapshot()
		if !s.Up || avoid[a.url] || !s.serves(model) {
			continue
		}
		switch load := a.inflight.Load(); {
		case best == nil || load < bestLoad:
			best, bestLoad = []*agent{a}, load
		case load == bestLoad:
			best = append(best, a)
		}
	}
	if len(best) == 0 {
		return nil
	}
	sort.Slice(best, func(i, j int) bool { return best[i].url < best[j].url })
	c.mu.Lock()
	c.next++
	n := c.next
	c.mu.Unlock()
	return best[n%len(best)]
}

func (c *Controller) proxy(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		oaiError(w, http.StatusRequestEntityTooLarge, "request_too_large", err.Error())
		return
	}
	var body struct {
		Model *string `json:"model"`
	}
	if err := json.Unmarshal(raw, &body); err != nil || body.Model == nil || *body.Model == "" {
		oaiError(w, http.StatusBadRequest, "invalid_model", "the request body must be a JSON object naming a model; GET /v1/models lists them")
		return
	}
	model := *body.Model
	start := time.Now()
	code := c.forward(w, r, model, raw)
	c.metrics.request(model, code, time.Since(start))
}

// forward sends a request to an agent serving model, placing the model
// first when it is in the catalogue and runs nowhere. A node that cannot
// be reached before it answers is skipped for another. It returns the
// status code sent.
func (c *Controller) forward(w http.ResponseWriter, r *http.Request, model string, raw []byte) int {
	avoid := map[string]bool{}
	for attempt := 0; attempt < 3; attempt++ {
		a := c.pick(model, avoid)
		if a == nil {
			m, ok := c.currentCatalogue().find(model)
			if !ok {
				oaiError(w, http.StatusNotFound, "model_not_found", fmt.Sprintf("model %q is not in the catalogue; GET /v1/models lists the models", model))
				return http.StatusNotFound
			}
			// A replica that runs but does not answer to this name (an
			// expert the image lacks) needs no other replica.
			if models, ok := c.runningModels(m.Name, avoid); ok {
				oaiError(w, http.StatusNotFound, "model_not_found", fmt.Sprintf("%s serves %s, not %q", m.Name, strings.Join(models, ", "), model))
				return http.StatusNotFound
			}
			if err := c.ensure(r.Context(), m); err != nil {
				if r.Context().Err() != nil {
					return 499 // the client went away
				}
				oaiError(w, http.StatusServiceUnavailable, "model_unavailable", fmt.Sprintf("model %q could not start: %v", model, err))
				return http.StatusServiceUnavailable
			}
			if a = c.pick(model, avoid); a == nil {
				oaiError(w, http.StatusNotFound, "model_not_found", fmt.Sprintf("model %q started but serves no model of that name; check the image's experts (GET /v1/cluster)", model))
				return http.StatusNotFound
			}
		}
		code, sent, err := c.send(w, r, a, raw)
		if err == nil || sent {
			return code
		}
		c.cfg.Logger.Warn("agent unreachable", "agent", a.url, "model", model, "err", err)
		a.markDown(err)
		avoid[a.url] = true
	}
	oaiError(w, http.StatusBadGateway, "agent_unreachable", fmt.Sprintf("no node serving %q answered", model))
	return http.StatusBadGateway
}

// runningModels returns the model names of a running replica of the
// container name, on an agent that is up and not avoided.
func (c *Controller) runningModels(name string, avoid map[string]bool) ([]string, bool) {
	for _, s := range c.states() {
		if ct, ok := s.container(name); ok && s.Up && !avoid[s.URL] && ct.Status == "running" {
			return ct.Models, true
		}
	}
	return nil, false
}

// send proxies one request to an agent's gateway, streaming the reply. sent
// reports whether any of the reply reached the client.
func (c *Controller) send(w http.ResponseWriter, r *http.Request, a *agent, raw []byte) (code int, sent bool, err error) {
	base, err := a.gatewayURL()
	if err != nil {
		return 0, false, err
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, base+r.URL.Path, bytes.NewReader(raw))
	if err != nil {
		return 0, false, err
	}
	req.Header.Set("Content-Type", "application/json")
	if v := r.Header.Get("Accept"); v != "" {
		req.Header.Set("Accept", v)
	}
	if c.cfg.AgentKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.AgentKey)
	}
	a.inflight.Add(1)
	defer a.inflight.Add(-1)
	resp, err := c.cfg.Client.Do(req)
	if err != nil {
		return 0, false, err
	}
	defer resp.Body.Close()
	for _, h := range []string{"Content-Type", "Cache-Control"} {
		if v := resp.Header.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	flusher, _ := w.(http.Flusher)
	stream := strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream")
	buf := make([]byte, 32<<10)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return resp.StatusCode, true, nil
			}
			if stream && flusher != nil {
				flusher.Flush()
			}
		}
		if rerr != nil {
			if !errors.Is(rerr, io.EOF) {
				c.cfg.Logger.Warn("reply cut short", "agent", a.url, "err", rerr)
			}
			return resp.StatusCode, true, nil
		}
	}
}

// clusterStatus is what GET /v1/cluster returns.
type clusterStatus struct {
	Agents    []agentState  `json:"agents"`
	Models    []modelStatus `json:"models"`
	Catalogue string        `json:"catalogue_error,omitempty"`
}

type modelStatus struct {
	Model
	// Running lists the nodes serving it.
	Running []string `json:"running"`
	// Error is why its last placement failed.
	Error string `json:"error,omitempty"`
}

func (c *Controller) cluster(w http.ResponseWriter, _ *http.Request) {
	states := c.states()
	sort.Slice(states, func(i, j int) bool { return states[i].URL < states[j].URL })
	out := clusterStatus{Agents: states}
	c.mu.Lock()
	out.Catalogue = c.catErr
	status := map[string]string{}
	for k, v := range c.status {
		status[k] = v
	}
	c.mu.Unlock()
	for _, m := range c.currentCatalogue().Models {
		ms := modelStatus{Model: m, Running: []string{}, Error: status[m.Name]}
		for _, s := range states {
			if ct, ok := s.container(m.Name); ok && s.Up && ct.Status == "running" {
				ms.Running = append(ms.Running, s.Node.Name)
			}
		}
		out.Models = append(out.Models, ms)
	}
	writeJSON(w, http.StatusOK, out)
}

// metrics counts requests and placements for Prometheus.
type metrics struct {
	mu         sync.Mutex
	requests   map[[2]string]int64 // model, code
	seconds    map[string]float64  // model: total request time
	placements map[[2]string]int64 // model, result
}

func newMetrics() metrics {
	return metrics{requests: map[[2]string]int64{}, seconds: map[string]float64{}, placements: map[[2]string]int64{}}
}

func (m *metrics) request(model string, code int, took time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests[[2]string{model, fmt.Sprint(code)}]++
	m.seconds[model] += took.Seconds()
}

func (m *metrics) placement(model string, err error) {
	result := "ok"
	if err != nil {
		result = "error"
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.placements[[2]string{model, result}]++
}

func (c *Controller) serveMetrics(w http.ResponseWriter, _ *http.Request) {
	var b strings.Builder
	states := c.states()
	up := 0
	for _, s := range states {
		if s.Up {
			up++
		}
	}
	fmt.Fprintf(&b, "# HELP wpk_control_agents Node agents found, and those up.\n# TYPE wpk_control_agents gauge\n")
	fmt.Fprintf(&b, "wpk_control_agents{state=\"found\"} %d\nwpk_control_agents{state=\"up\"} %d\n", len(states), up)
	fmt.Fprintf(&b, "# HELP wpk_control_replicas Nodes serving each catalogue model.\n# TYPE wpk_control_replicas gauge\n")
	for _, m := range c.currentCatalogue().Models {
		n := 0
		for _, s := range states {
			if ct, ok := s.container(m.Name); ok && s.Up && ct.Status == "running" {
				n++
			}
		}
		fmt.Fprintf(&b, "wpk_control_replicas{model=%q} %d\n", m.Name, n)
	}
	fmt.Fprintf(&b, "# HELP wpk_control_inflight Requests in flight per node.\n# TYPE wpk_control_inflight gauge\n")
	for _, s := range states {
		fmt.Fprintf(&b, "wpk_control_inflight{node=%q} %d\n", s.Node.Name, s.Inflight)
	}
	c.metrics.mu.Lock()
	fmt.Fprintf(&b, "# HELP wpk_control_requests_total Requests by model and status code.\n# TYPE wpk_control_requests_total counter\n")
	for _, k := range sortedKeys(c.metrics.requests) {
		fmt.Fprintf(&b, "wpk_control_requests_total{model=%q,code=%q} %d\n", k[0], k[1], c.metrics.requests[k])
	}
	fmt.Fprintf(&b, "# HELP wpk_control_request_seconds_total Time spent on requests, by model.\n# TYPE wpk_control_request_seconds_total counter\n")
	models := make([]string, 0, len(c.metrics.seconds))
	for m := range c.metrics.seconds {
		models = append(models, m)
	}
	sort.Strings(models)
	for _, m := range models {
		fmt.Fprintf(&b, "wpk_control_request_seconds_total{model=%q} %g\n", m, c.metrics.seconds[m])
	}
	fmt.Fprintf(&b, "# HELP wpk_control_placements_total Placements by model and result.\n# TYPE wpk_control_placements_total counter\n")
	for _, k := range sortedKeys(c.metrics.placements) {
		fmt.Fprintf(&b, "wpk_control_placements_total{model=%q,result=%q} %d\n", k[0], k[1], c.metrics.placements[k])
	}
	c.metrics.mu.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = io.WriteString(w, b.String())
}

func sortedKeys(m map[[2]string]int64) [][2]string {
	keys := make([][2]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i][0] != keys[j][0] {
			return keys[i][0] < keys[j][0]
		}
		return keys[i][1] < keys[j][1]
	})
	return keys
}
