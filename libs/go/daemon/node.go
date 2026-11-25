package daemon

import (
	"crypto/subtle"
	"errors"
	"net/http"
	"os"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

// WithNode names the machine and the port its gateway listens on, as
// GET /v1/node reports them to a cluster control plane. The name defaults
// to the hostname.
func WithNode(name string, gatewayPort int) Option {
	return func(s *server) { s.node, s.gatewayPort = name, gatewayPort }
}

func (s *server) nodeInfo(w http.ResponseWriter, _ *http.Request) {
	n := api.Node{Name: s.node, Version: s.version, GatewayPort: s.gatewayPort}
	if n.Name == "" {
		n.Name, _ = os.Hostname()
	}
	if s.runtime != nil {
		n.Serving = true
		n.Device = s.runtime.Device()
		n.Engines = s.runtime.Engines()
		n.MaxEngines = s.runtime.MaxEngines()
	}
	writeJSON(w, http.StatusOK, n)
}

// RequireKey answers 401 to requests that do not carry key as a bearer
// token, except GET /healthz, which probes call. wpkd wraps the API in it
// when it serves it over TCP (--api).
func RequireKey(h http.Handler, key string) http.Handler {
	want := []byte("Bearer " + key)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/healthz" {
			h.ServeHTTP(w, r)
			return
		}
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			writeError(w, http.StatusUnauthorized, errUnauthorized)
			return
		}
		h.ServeHTTP(w, r)
	})
}

var errUnauthorized = errors.New("missing or wrong API key; send it as Authorization: Bearer <key>")
