package control

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

// Discovery finds the node agents: wpkd daemons serving their API on TCP
// (wpkd --api).
type Discovery interface {
	Agents(ctx context.Context) ([]string, error)
}

// ParseAgents reads the --agents flag: comma-separated base URLs
// (http://10.0.0.5:8091), or dns://<name>:<port>, which looks the name up
// on every pass and takes each address as an agent, as a Kubernetes
// headless Service lists its ready pods.
func ParseAgents(spec string) (Discovery, error) {
	if rest, ok := strings.CutPrefix(spec, "dns://"); ok {
		host, port, err := net.SplitHostPort(strings.TrimPrefix(rest, "/"))
		if err != nil {
			return nil, fmt.Errorf("agents %q: want dns://<name>:<port>: %w", spec, err)
		}
		return dnsDiscovery{host: host, port: port}, nil
	}
	var urls staticDiscovery
	for _, s := range strings.Split(spec, ",") {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		u, err := url.Parse(s)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return nil, fmt.Errorf("agent %q: want http://<host>:<port>", s)
		}
		urls = append(urls, strings.TrimRight(s, "/"))
	}
	if len(urls) == 0 {
		return nil, errors.New("no agents: set --agents")
	}
	return urls, nil
}

type staticDiscovery []string

func (s staticDiscovery) Agents(context.Context) ([]string, error) { return s, nil }

type dnsDiscovery struct{ host, port string }

func (d dnsDiscovery) Agents(ctx context.Context) ([]string, error) {
	addrs, err := net.DefaultResolver.LookupHost(ctx, d.host)
	if err != nil {
		return nil, fmt.Errorf("look up agents at %s: %w", d.host, err)
	}
	sort.Strings(addrs)
	out := make([]string, len(addrs))
	for i, a := range addrs {
		out[i] = "http://" + net.JoinHostPort(a, d.port)
	}
	return out, nil
}

// agent is the control plane's view of one node agent, refreshed every
// pass.
type agent struct {
	url    string
	client *api.Client

	mu         sync.Mutex
	up         bool
	err        string
	seen       time.Time
	node       api.Node
	containers []api.Container
	images     []api.Image

	inflight atomic.Int64 // requests the control plane is proxying to it
}

// refresh reads the agent's node, containers and images.
func (a *agent) refresh(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	node, err := a.client.Node(ctx)
	var containers []api.Container
	var images []api.Image
	if err == nil {
		containers, err = a.client.Containers(ctx)
	}
	if err == nil {
		images, err = a.client.Images(ctx)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err != nil {
		a.up, a.err = false, err.Error()
		return
	}
	a.up, a.err, a.seen = true, "", time.Now()
	a.node, a.containers, a.images = *node, containers, images
}

// snapshot copies the agent's state.
func (a *agent) snapshot() agentState {
	a.mu.Lock()
	defer a.mu.Unlock()
	return agentState{
		URL: a.url, Up: a.up, Error: a.err, Seen: a.seen, Node: a.node,
		Containers: append([]api.Container(nil), a.containers...), Images: append([]api.Image(nil), a.images...),
		Inflight: a.inflight.Load(),
	}
}

// markDown records that a request could not reach the agent; the next
// refresh decides whether it is back.
func (a *agent) markDown(err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.up, a.err = false, err.Error()
}

// gatewayURL is the agent's OpenAI gateway: its host and the gateway port
// it reports.
func (a *agent) gatewayURL() (string, error) {
	a.mu.Lock()
	port := a.node.GatewayPort
	a.mu.Unlock()
	if port == 0 {
		return "", fmt.Errorf("agent %s does not serve (wpkd runs without --gateway)", a.url)
	}
	u, err := url.Parse(a.url)
	if err != nil {
		return "", err
	}
	return u.Scheme + "://" + net.JoinHostPort(u.Hostname(), strconv.Itoa(port)), nil
}

// agentState is a copy of an agent's state, for placement and status.
type agentState struct {
	URL        string          `json:"url"`
	Up         bool            `json:"up"`
	Error      string          `json:"error,omitempty"`
	Seen       time.Time       `json:"seen"`
	Node       api.Node        `json:"node"`
	Containers []api.Container `json:"containers"`
	Images     []api.Image     `json:"-"`
	Inflight   int64           `json:"inflight"`
}

// serves reports whether the agent has a running container answering to
// model.
func (s agentState) serves(model string) bool {
	for _, c := range s.Containers {
		if c.Status != "running" {
			continue
		}
		for _, m := range c.Models {
			if m == model {
				return true
			}
		}
	}
	return false
}

// container returns the agent's container named name, if any.
func (s agentState) container(name string) (api.Container, bool) {
	for _, c := range s.Containers {
		if c.Name == name {
			return c, true
		}
	}
	return api.Container{}, false
}
