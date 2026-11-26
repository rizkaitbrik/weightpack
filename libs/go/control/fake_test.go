package control

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

const agentKey = "agent-secret"

// fakeImage is an image in the fake registry: what a placement resolve
// answers for its reference.
type fakeImage struct {
	digest, base, servingBase string
	experts                   []string
	refused                   string // the trust policy's refusal, if any
}

// fakeRegistry is shared by the fake agents, as a real registry is.
type fakeRegistry struct {
	mu     sync.Mutex
	images map[string]fakeImage // by reference
}

func (r *fakeRegistry) get(ref string) (fakeImage, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	img, ok := r.images[ref]
	return img, ok
}

func (r *fakeRegistry) set(ref string, img fakeImage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.images[ref] = img
}

// fakeAgent speaks the node agent's API and gateway: enough for the
// control plane to place, run, stop and route.
type fakeAgent struct {
	name       string
	reg        *fakeRegistry
	api, gw    *httptest.Server
	maxEngines int
	failRun    string // when set, every run fails with it

	mu         sync.Mutex
	images     []api.Image
	containers []api.Container
	pulls      int
	served     map[string]int // model -> requests answered
}

func newFakeAgent(t *testing.T, name string, reg *fakeRegistry) *fakeAgent {
	t.Helper()
	a := &fakeAgent{name: name, reg: reg, served: map[string]int{}}
	a.gw = httptest.NewServer(http.HandlerFunc(a.gateway))
	a.api = httptest.NewServer(http.HandlerFunc(a.handle))
	t.Cleanup(func() {
		a.api.Close()
		a.gw.Close()
	})
	return a
}

func (a *fakeAgent) url() string { return a.api.URL }

func (a *fakeAgent) engines() int {
	bases := map[string]bool{}
	for _, c := range a.containers {
		if c.Status == "running" {
			bases[c.Base] = true
		}
	}
	return len(bases)
}

func (a *fakeAgent) handle(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer "+agentKey {
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(api.ErrorBody{Error: "missing or wrong API key"})
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	fail := func(status int, msg string) {
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(api.ErrorBody{Error: msg})
	}
	stream := func(e api.Event) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_ = json.NewEncoder(w).Encode(api.Event{Step: "fake", Message: "working"})
		_ = json.NewEncoder(w).Encode(e)
	}
	switch r.Method + " " + r.URL.Path {
	case "GET /v1/node":
		port := strings.TrimPrefix(a.gw.URL, "http://127.0.0.1:")
		var p int
		fmt.Sscan(port, &p)
		_ = json.NewEncoder(w).Encode(api.Node{Name: a.name, Serving: true, GatewayPort: p, Device: "cpu", Engines: a.engines(), MaxEngines: a.maxEngines})
	case "GET /v1/containers":
		_ = json.NewEncoder(w).Encode(append([]api.Container{}, a.containers...))
	case "GET /v1/images":
		_ = json.NewEncoder(w).Encode(append([]api.Image{}, a.images...))
	case "POST /v1/resolve":
		var req api.ResolveRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		img, ok := a.reg.get(req.Ref)
		switch {
		case !ok:
			fail(http.StatusBadGateway, req.Ref+": not found")
		case img.refused != "":
			fail(http.StatusForbidden, img.refused)
		default:
			_ = json.NewEncoder(w).Encode(api.ResolveResponse{Ref: req.Ref, Digest: img.digest, Base: img.base, ServingBase: img.servingBase})
		}
	case "POST /v1/pull":
		var req api.PullRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		img, ok := a.reg.get(req.Ref)
		if !ok {
			stream(api.Event{Error: "not found"})
			return
		}
		a.pulls++
		kept := a.images[:0]
		for _, im := range a.images {
			if im.Ref != req.Ref {
				kept = append(kept, im)
			}
		}
		a.images = append(kept, api.Image{Ref: req.Ref, Digest: img.digest, Base: img.base})
		stream(api.Event{Transfer: &api.TransferResult{Ref: req.Ref, Digest: img.digest}})
	case "POST /v1/run":
		var req api.RunRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if a.failRun != "" {
			stream(api.Event{Error: a.failRun})
			return
		}
		img, ok := a.reg.get(req.Ref)
		if !ok {
			stream(api.Event{Error: "no such image"})
			return
		}
		for _, c := range a.containers {
			if c.Name == req.Name {
				stream(api.Event{Error: "a container named " + req.Name + " is already running"})
				return
			}
		}
		newBase := true
		for _, c := range a.containers {
			newBase = newBase && !(c.Status == "running" && c.Base == img.servingBase)
		}
		if newBase && a.maxEngines > 0 && a.engines() >= a.maxEngines {
			stream(api.Event{Error: "this machine already runs its most engines"})
			return
		}
		// The run resolves the reference in the store, as the runtime does.
		digest := ""
		for _, im := range a.images {
			if im.Ref == req.Ref {
				digest = im.Digest
			}
		}
		if digest == "" {
			stream(api.Event{Error: "no such image: " + req.Ref})
			return
		}
		engine, device := req.Engine, req.Device
		if engine == "" {
			engine = "reference"
		}
		if device == "" {
			device = "cpu"
		}
		c := api.Container{Name: req.Name, Image: req.Ref, Digest: digest, Base: img.servingBase, Engine: engine, Device: device, Status: "running", Owner: req.Owner}
		c.Models = []string{req.Name}
		if len(img.experts) > 0 {
			c.Models = nil
			for _, e := range img.experts {
				c.Models = append(c.Models, req.Name+"/"+e)
			}
		}
		a.containers = append(a.containers, c)
		stream(api.Event{Container: &c})
	case "POST /v1/stop":
		var req api.StopRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		stop := map[string]bool{}
		for _, n := range req.Names {
			stop[n] = true
		}
		kept := a.containers[:0]
		var stopped []string
		for _, c := range a.containers {
			if stop[c.Name] {
				stopped = append(stopped, c.Name)
			} else {
				kept = append(kept, c)
			}
		}
		a.containers = kept
		_ = json.NewEncoder(w).Encode(api.StopResponse{Stopped: stopped})
	default:
		fail(http.StatusNotFound, "no such endpoint "+r.URL.Path)
	}
}

// gateway answers chat requests with which node and model they reached,
// streamed in two events when asked to.
func (a *fakeAgent) gateway(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer "+agentKey {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	var body struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	a.mu.Lock()
	serving := false
	for _, c := range a.containers {
		for _, m := range c.Models {
			serving = serving || (m == body.Model && c.Status == "running")
		}
	}
	if serving {
		a.served[body.Model]++
	}
	a.mu.Unlock()
	if !serving {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"message": "model " + body.Model + " is not running"}})
		return
	}
	answer := a.name + "|" + body.Model
	if !body.Stream {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"model": body.Model, "choices": []any{map[string]any{"message": map[string]string{"content": answer}}}})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	for _, part := range []string{answer[:len(answer)/2], answer[len(answer)/2:]} {
		chunk, _ := json.Marshal(map[string]any{"model": body.Model, "choices": []any{map[string]any{"delta": map[string]string{"content": part}}}})
		fmt.Fprintf(w, "data: %s\n\n", chunk)
		w.(http.Flusher).Flush()
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func (a *fakeAgent) servedCount(model string) int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.served[model]
}

func (a *fakeAgent) running() map[string]api.Container {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := map[string]api.Container{}
	for _, c := range a.containers {
		out[c.Name] = c
	}
	return out
}
