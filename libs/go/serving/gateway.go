package serving

import (
	"bufio"
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// maxGatewayBody bounds a request the gateway reads to route it.
const maxGatewayBody = 64 << 20

// Gateway returns the OpenAI-compatible API that fronts every container:
//
//	GET  /v1/models            the model names of every running container
//	POST /v1/chat/completions  routed by the body's model field
//	POST /v1/completions       routed by the body's model field
//	POST /v1/embeddings        routed by the body's model field
//	GET  /health               200 while the gateway runs
//
// With Config.APIKey set, every /v1 request must carry it as a bearer
// token. It swaps the model name for the engine's name for it on the way in and
// back on the way out, streams included, and sends an image's chat
// template with chat requests that do not bring their own.
func (rt *Runtime) Gateway() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /v1/models", rt.authorized(rt.models))
	for _, p := range []string{"/v1/chat/completions", "/v1/completions", "/v1/embeddings"} {
		mux.HandleFunc("POST "+p, rt.authorized(rt.proxy))
	}
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		oaiError(w, http.StatusNotFound, "not_found", fmt.Sprintf("%s %s is not part of the API wpkd serves", r.Method, r.URL.Path))
	})
	return mux
}

// authorized checks the bearer token when the runtime has an API key.
func (rt *Runtime) authorized(h http.HandlerFunc) http.HandlerFunc {
	if rt.cfg.APIKey == "" {
		return h
	}
	want := []byte("Bearer " + rt.cfg.APIKey)
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

func (rt *Runtime) models(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	rt.mu.Lock()
	data := []model{}
	for name, r := range rt.routes {
		if r.c.engine.hasExited() {
			continue
		}
		data = append(data, model{ID: name, Object: "model", Created: r.c.info.Created.Unix(), OwnedBy: "weightpack"})
	}
	rt.mu.Unlock()
	sort.Slice(data, func(i, j int) bool { return data[i].ID < data[j].ID })
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// lookup finds the route for a model. A request without a model goes to
// the only one running, if there is exactly one.
func (rt *Runtime) lookup(model string) (route, string, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if model == "" {
		if len(rt.routes) == 1 {
			for name, r := range rt.routes {
				return r, name, nil
			}
		}
		return route{}, "", errors.New("the request names no model; GET /v1/models lists them")
	}
	r, ok := rt.routes[model]
	if !ok {
		names := make([]string, 0, len(rt.routes))
		for n := range rt.routes {
			names = append(names, n)
		}
		sort.Strings(names)
		if len(names) == 0 {
			return route{}, "", fmt.Errorf("model %q is not running, and nothing is; start it with wpk run", model)
		}
		return route{}, "", fmt.Errorf("model %q is not running; running: %s", model, strings.Join(names, ", "))
	}
	return r, model, nil
}

func (rt *Runtime) proxy(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxGatewayBody))
	if err != nil {
		oaiError(w, http.StatusRequestEntityTooLarge, "request_too_large", err.Error())
		return
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		oaiError(w, http.StatusBadRequest, "invalid_json", "the request body is not a JSON object: "+err.Error())
		return
	}
	var model string
	if m, ok := body["model"]; ok {
		if err := json.Unmarshal(m, &model); err != nil {
			oaiError(w, http.StatusBadRequest, "invalid_model", "model must be a string")
			return
		}
	}
	rt2, name, err := rt.lookup(model)
	if err != nil {
		oaiError(w, http.StatusNotFound, "model_not_found", err.Error())
		return
	}
	e := rt2.c.engine
	if msg := e.exitError(); msg != "" {
		oaiError(w, http.StatusServiceUnavailable, "engine_exited", fmt.Sprintf("model %q is down: %s; wpk logs %s shows why", name, msg, rt2.c.info.Name))
		return
	}
	body["model"], _ = json.Marshal(rt2.upstream)
	if _, has := body["chat_template"]; !has && rt2.c.template != "" && r.URL.Path == "/v1/chat/completions" {
		body["chat_template"], _ = json.Marshal(rt2.c.template)
	}
	out, err := json.Marshal(body)
	if err != nil {
		oaiError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, e.url+r.URL.Path, bytes.NewReader(out))
	if err != nil {
		oaiError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if a := r.Header.Get("Accept"); a != "" {
		req.Header.Set("Accept", a)
	}
	resp, err := engineClient.Do(req)
	if err != nil {
		oaiError(w, http.StatusBadGateway, "engine_unreachable", fmt.Sprintf("the engine for %q did not answer: %v", name, err))
		return
	}
	defer resp.Body.Close()

	ct := resp.Header.Get("Content-Type")
	w.Header().Set("Content-Type", ct)
	switch {
	case strings.HasPrefix(ct, "text/event-stream"):
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(resp.StatusCode)
		relayEvents(w, resp.Body, name)
	case strings.HasPrefix(ct, "application/json"):
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			return
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(renameModel(raw, name))
	default:
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	}
}

// engineClient talks to engines on loopback. Generation can take minutes,
// so only the wait for response headers is bounded, generously.
var engineClient = &http.Client{Transport: &http.Transport{
	ResponseHeaderTimeout: 30 * time.Minute,
	MaxIdleConnsPerHost:   64,
}}

// renameModel sets the top-level model field of a JSON object to name,
// leaving every other value byte for byte as the engine wrote it.
func renameModel(raw []byte, name string) []byte {
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return raw
	}
	if _, ok := obj["model"]; !ok {
		return raw
	}
	obj["model"], _ = json.Marshal(name)
	out, err := json.Marshal(obj)
	if err != nil {
		return raw
	}
	return out
}

// relayEvents copies a server-sent event stream, renaming the model in
// each data line and flushing after each event.
func relayEvents(w http.ResponseWriter, body io.Reader, name string) {
	flusher, _ := w.(http.Flusher)
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 64<<10), 16<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if data, ok := bytes.CutPrefix(line, []byte("data: ")); ok && len(data) > 0 && data[0] == '{' {
			line = append([]byte("data: "), renameModel(data, name)...)
		}
		if _, err := w.Write(append(line, '\n')); err != nil {
			return
		}
		if len(line) == 0 && flusher != nil {
			flusher.Flush()
		}
	}
	if flusher != nil {
		flusher.Flush()
	}
}
