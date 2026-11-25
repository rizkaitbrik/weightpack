package control

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

type cluster struct {
	c         *Controller
	srv       *httptest.Server
	reg       *fakeRegistry
	catalogue string
}

func newCluster(t *testing.T, agents []*fakeAgent, reg *fakeRegistry, models ...Model) *cluster {
	t.Helper()
	var urls staticDiscovery
	for _, a := range agents {
		urls = append(urls, a.url())
	}
	cl := &cluster{reg: reg, catalogue: filepath.Join(t.TempDir(), "catalogue.json")}
	cl.setCatalogue(t, models...)
	cl.c = New(Config{Discovery: urls, AgentKey: agentKey, Catalogue: cl.catalogue, Interval: time.Hour})
	cl.srv = httptest.NewServer(cl.c)
	t.Cleanup(cl.srv.Close)
	return cl
}

func (cl *cluster) setCatalogue(t *testing.T, models ...Model) {
	t.Helper()
	raw, _ := json.Marshal(Catalogue{Models: models})
	if err := os.WriteFile(cl.catalogue, raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

// sync makes passes until no placement is under way.
func (cl *cluster) sync(t *testing.T) {
	t.Helper()
	cl.c.Sync(context.Background())
	deadline := time.Now().Add(5 * time.Second)
	for {
		cl.c.mu.Lock()
		n := len(cl.c.placing)
		cl.c.mu.Unlock()
		if n == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("placements did not finish")
		}
		time.Sleep(5 * time.Millisecond)
	}
	cl.c.refreshAll(context.Background())
}

func (cl *cluster) chat(t *testing.T, model string, stream bool, key string) (int, string) {
	t.Helper()
	body, _ := json.Marshal(map[string]any{"model": model, "stream": stream, "messages": []any{}})
	req, _ := http.NewRequest(http.MethodPost, cl.srv.URL+"/v1/chat/completions", bytes.NewReader(body))
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if !stream || resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		var out struct {
			Choices []struct {
				Message struct{ Content string } `json:"message"`
			} `json:"choices"`
			Error struct{ Message string } `json:"error"`
		}
		_ = json.Unmarshal(raw, &out)
		if len(out.Choices) > 0 {
			return resp.StatusCode, out.Choices[0].Message.Content
		}
		return resp.StatusCode, out.Error.Message
	}
	var content strings.Builder
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct{ Content string } `json:"delta"`
			} `json:"choices"`
		}
		_ = json.Unmarshal([]byte(data), &chunk)
		content.WriteString(chunk.Choices[0].Delta.Content)
	}
	return resp.StatusCode, content.String()
}

func newRegistry() *fakeRegistry { return &fakeRegistry{images: map[string]fakeImage{}} }

func TestChooseByBaseAffinity(t *testing.T) {
	res := resolution{ResolveResponse: api.ResolveResponse{Digest: "sha256:img", Base: "chain-base", ServingBase: "chain-serving"}}
	m := Model{Name: "tasks", Image: "reg.example/tasks:v1"}
	node := func(name string, engines, maxEngines int) api.Node {
		return api.Node{Name: name, Serving: true, Engines: engines, MaxEngines: maxEngines}
	}
	idle := agentState{URL: "a-idle", Up: true, Node: node("idle", 0, 0)}
	holdsBase := agentState{URL: "b-base", Up: true, Node: node("base", 1, 0), Images: []api.Image{{Digest: "sha256:other", Base: "chain-base"}}}
	holdsImage := agentState{URL: "c-image", Up: true, Node: node("image", 1, 0), Images: []api.Image{{Digest: "sha256:img", Base: "chain-base"}}}
	servesBase := agentState{URL: "d-engine", Up: true, Node: node("engine", 2, 2),
		Containers: []api.Container{{Name: "other", Base: "chain-serving", Engine: "reference", Status: "running"}}}
	full := agentState{URL: "e-full", Up: true, Node: node("full", 1, 1), Images: []api.Image{{Digest: "sha256:img"}}}
	down := agentState{URL: "f-down", Node: node("down", 0, 0)}

	for _, c := range []struct {
		states []agentState
		want   string
	}{
		{[]agentState{idle, holdsBase, holdsImage, servesBase, full, down}, "engine"}, // full, but its engine serves the base
		{[]agentState{idle, holdsBase, holdsImage, full, down}, "image"},
		{[]agentState{idle, holdsBase, full}, "base"},
		{[]agentState{idle, full, down}, "idle"},
	} {
		got, err := choose(c.states, m, res)
		if err != nil || got.Node.Name != c.want {
			t.Errorf("choose = %s, %v; want %s", got.Node.Name, err, c.want)
		}
	}
	if _, err := choose([]agentState{full, down}, m, res); err == nil || !strings.Contains(err.Error(), "--max-engines") {
		t.Errorf("only a full node: %v", err)
	}
	if _, err := choose([]agentState{down}, m, res); err == nil || !strings.Contains(err.Error(), "no node agent is up") {
		t.Errorf("no node up: %v", err)
	}
	// A pinned engine kind rules out reuse of another kind's engine.
	vllm := m
	vllm.Engine = "vllm"
	if got, _ := choose([]agentState{idle, servesBase}, vllm, res); got.Node.Name != "idle" {
		t.Errorf("vllm model went to %s", got.Node.Name)
	}
}

func TestOnDemandPlacementAndRouting(t *testing.T) {
	reg := newRegistry()
	reg.set("reg.example/tasks:v1", fakeImage{digest: "sha256:t1", base: "b", servingBase: "sb", experts: []string{"sql", "math"}})
	reg.set("reg.example/bad:v1", fakeImage{digest: "sha256:x", refused: "the trust policy for reg.example requires a trusted signature"})
	a, b := newFakeAgent(t, "gpu-a", reg), newFakeAgent(t, "gpu-b", reg)
	cl := newCluster(t, []*fakeAgent{a, b}, reg,
		Model{Name: "tasks", Image: "reg.example/tasks:v1"},
		Model{Name: "bad", Image: "reg.example/bad:v1"})
	cl.sync(t)
	if len(a.running())+len(b.running()) != 0 {
		t.Fatal("an on-demand model started before any request")
	}

	code, got := cl.chat(t, "tasks/sql", false, "")
	if code != http.StatusOK || !strings.HasSuffix(got, "|tasks/sql") {
		t.Fatalf("first request: %d %q", code, got)
	}
	node := strings.SplitN(got, "|", 2)[0]
	// The next requests, streamed or not, reach the same node: only it
	// serves the model.
	if code, got := cl.chat(t, "tasks/math", true, ""); code != http.StatusOK || got != node+"|tasks/math" {
		t.Errorf("streamed request: %d %q", code, got)
	}
	if n := len(a.running()) + len(b.running()); n != 1 {
		t.Errorf("%d containers after two requests", n)
	}

	if code, got := cl.chat(t, "nope", false, ""); code != http.StatusNotFound || !strings.Contains(got, "not in the catalogue") {
		t.Errorf("unknown model: %d %q", code, got)
	}
	if code, got := cl.chat(t, "tasks/history", false, ""); code != http.StatusNotFound || !strings.Contains(got, "serves tasks/sql, tasks/math") {
		t.Errorf("unknown expert: %d %q", code, got)
	}
	if code, got := cl.chat(t, "bad", false, ""); code != http.StatusServiceUnavailable || !strings.Contains(got, "trusted signature") {
		t.Errorf("refused image: %d %q", code, got)
	}

	resp, err := http.Get(cl.srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var list struct{ Data []struct{ ID string } }
	_ = json.NewDecoder(resp.Body).Decode(&list)
	resp.Body.Close()
	var ids []string
	for _, d := range list.Data {
		ids = append(ids, d.ID)
	}
	if strings.Join(ids, ",") != "bad,tasks,tasks/math,tasks/sql" {
		t.Errorf("models = %v", ids)
	}

	resp, _ = http.Get(cl.srv.URL + "/metrics")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	for _, want := range []string{`wpk_control_agents{state="up"} 2`, `wpk_control_requests_total{model="tasks/sql",code="200"} 1`,
		`wpk_control_placements_total{model="tasks",result="ok"} 1`, `wpk_control_placements_total{model="bad",result="error"} 1`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("metrics lack %s:\n%s", want, raw)
		}
	}
}

func TestReplicasRolloutAndRemoval(t *testing.T) {
	reg := newRegistry()
	reg.set("reg.example/chat:v1", fakeImage{digest: "sha256:c1", base: "b", servingBase: "sb1"})
	a, b, c := newFakeAgent(t, "gpu-a", reg), newFakeAgent(t, "gpu-b", reg), newFakeAgent(t, "gpu-c", reg)
	cl := newCluster(t, []*fakeAgent{a, b, c}, reg, Model{Name: "chat", Image: "reg.example/chat:v1", Replicas: 2})
	// A container someone started by hand is not the control plane's.
	c.containers = append(c.containers, api.Container{Name: "manual", Digest: "sha256:m", Status: "running", Models: []string{"manual"}})

	placed := func() (n int, nodes []string) {
		for _, ag := range []*fakeAgent{a, b, c} {
			if ct, ok := ag.running()["chat"]; ok {
				n++
				nodes = append(nodes, ag.name+"@"+ct.Digest)
			}
		}
		return n, nodes
	}
	for i := 0; i < 3; i++ {
		cl.sync(t)
	}
	if n, nodes := placed(); n != 2 {
		t.Fatalf("replicas = %v", nodes)
	}
	// Requests spread over both replicas.
	for i := 0; i < 6; i++ {
		if code, _ := cl.chat(t, "chat", false, ""); code != http.StatusOK {
			t.Fatalf("request %d: %d", i, code)
		}
	}
	busy := 0
	for _, ag := range []*fakeAgent{a, b, c} {
		if ag.servedCount("chat") > 0 {
			busy++
		}
	}
	if busy != 2 {
		t.Errorf("requests reached %d nodes, want 2", busy)
	}

	// The tag moves: replicas are replaced, never both at once.
	reg.set("reg.example/chat:v1", fakeImage{digest: "sha256:c2", base: "b", servingBase: "sb2"})
	cl.c.mu.Lock()
	cl.c.resolved = map[string]resolution{}
	cl.c.mu.Unlock()
	for i := 0; i < 8; i++ {
		cl.sync(t)
		if n, nodes := placed(); n == 0 {
			t.Fatalf("pass %d left no replica: %v", i, nodes)
		}
	}
	if n, nodes := placed(); n != 2 || strings.Contains(strings.Join(nodes, ","), "c1") {
		t.Errorf("after the rollout: %v", nodes)
	}

	// Scaling down stops one; dropping the model stops the rest, and
	// leaves the hand-started container alone.
	cl.setCatalogue(t, Model{Name: "chat", Image: "reg.example/chat:v1", Replicas: 1})
	cl.sync(t)
	if n, nodes := placed(); n != 1 {
		t.Errorf("after scaling down: %v", nodes)
	}
	cl.setCatalogue(t)
	cl.sync(t)
	if n, nodes := placed(); n != 0 {
		t.Errorf("after removal: %v", nodes)
	}
	if _, ok := c.running()["manual"]; !ok {
		t.Error("the control plane stopped a container it did not start")
	}
}

func TestFailoverAndKeys(t *testing.T) {
	reg := newRegistry()
	reg.set("reg.example/chat:v1", fakeImage{digest: "sha256:c1", base: "b", servingBase: "sb"})
	a, b := newFakeAgent(t, "gpu-a", reg), newFakeAgent(t, "gpu-b", reg)
	cl := newCluster(t, []*fakeAgent{a, b}, reg, Model{Name: "chat", Image: "reg.example/chat:v1", Replicas: 2})
	for i := 0; i < 2; i++ {
		cl.sync(t)
	}
	// One node's gateway dies between passes: requests go to the other.
	a.gw.Close()
	for i := 0; i < 4; i++ {
		if code, got := cl.chat(t, "chat", false, ""); code != http.StatusOK || !strings.HasPrefix(got, "gpu-b|") {
			t.Fatalf("request %d with gpu-a down: %d %q", i, code, got)
		}
	}

	cl.c.cfg.APIKey = "client-secret"
	cl.c.handler = cl.c.routes()
	if code, _ := cl.chat(t, "chat", false, ""); code != http.StatusUnauthorized {
		t.Errorf("no key: %d", code)
	}
	if code, _ := cl.chat(t, "chat", false, "client-secret"); code != http.StatusOK {
		t.Errorf("right key: %d", code)
	}
}

func TestCatalogueAndDiscovery(t *testing.T) {
	for _, bad := range []Catalogue{
		{Models: []Model{{Name: "x", Image: "no-registry:v1"}}},
		{Models: []Model{{Name: "-x", Image: "reg.example/x"}}},
		{Models: []Model{{Name: "x", Image: "reg.example/x"}, {Name: "x", Image: "reg.example/y"}}},
		{Models: []Model{{Name: "x", Image: "reg.example/x", Engine: "tgi"}}},
		{Models: []Model{{Name: "x", Image: "reg.example/x", Replicas: -1}}},
	} {
		if err := bad.Check(); err == nil {
			t.Errorf("accepted %+v", bad)
		}
	}
	path := filepath.Join(t.TempDir(), "c.json")
	_ = os.WriteFile(path, []byte(`{"models": [{"name": "x", "image": "reg.example/x", "replica": 2}]}`), 0o644)
	if _, err := LoadCatalogue(path); err == nil || !strings.Contains(err.Error(), "replica") {
		t.Errorf("misspelt field: %v", err)
	}

	d, err := ParseAgents("http://10.0.0.1:8091, http://10.0.0.2:8091/")
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := d.Agents(context.Background()); strings.Join(got, " ") != "http://10.0.0.1:8091 http://10.0.0.2:8091" {
		t.Errorf("static agents = %v", got)
	}
	d, err = ParseAgents("dns://localhost:8091")
	if err != nil {
		t.Fatal(err)
	}
	if got, err := d.Agents(context.Background()); err != nil || len(got) == 0 || !strings.HasSuffix(got[0], ":8091") {
		t.Errorf("dns agents = %v, %v", got, err)
	}
	for _, bad := range []string{"", "10.0.0.1:8091", "dns://no-port"} {
		if _, err := ParseAgents(bad); err == nil {
			t.Errorf("accepted agents %q", bad)
		}
	}
}

func TestPlacementPullsTheTagItRuns(t *testing.T) {
	reg := newRegistry()
	reg.set("reg.example/chat:v1", fakeImage{digest: "sha256:c2", base: "b", servingBase: "sb"})
	a := newFakeAgent(t, "gpu-a", reg)
	cl := newCluster(t, []*fakeAgent{a}, reg)
	m := Model{Name: "chat", Image: "reg.example/chat:v1"}
	ctx := context.Background()
	for _, c := range []struct {
		name   string
		images []api.Image
		pulls  int
	}{
		{"the digest under another name", []api.Image{{Ref: "reg.example/old:v1", Digest: "sha256:c2"}}, 1},
		{"an older digest under this name", []api.Image{{Ref: "reg.example/chat:v1", Digest: "sha256:c1"}}, 1},
		{"this name at this digest", []api.Image{{Ref: "reg.example/chat:v1", Digest: "sha256:c2"}}, 0},
	} {
		a.mu.Lock()
		a.images, a.containers, a.pulls = c.images, nil, 0
		a.mu.Unlock()
		cl.c.Sync(ctx)
		if err := cl.c.place(ctx, m, ""); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		ct := a.running()["chat"]
		if a.pulls != c.pulls || ct.Digest != "sha256:c2" {
			t.Errorf("%s: %d pulls, running %s", c.name, a.pulls, ct.Digest)
		}
	}
	if !current([]api.Image{{Ref: "reg.example/chat:latest", Digest: "sha256:c2"}}, "reg.example/chat", "sha256:c2") ||
		!current([]api.Image{{Ref: "x", Digest: "sha256:" + strings.Repeat("a", 64)}}, "reg.example/chat@sha256:"+strings.Repeat("a", 64), "") {
		t.Error("current misreads a reference without a tag, or with a digest")
	}
}

func TestEngineChangeReplacesReplicas(t *testing.T) {
	reg := newRegistry()
	reg.set("reg.example/chat:v1", fakeImage{digest: "sha256:c1", base: "b", servingBase: "sb"})
	a, b := newFakeAgent(t, "gpu-a", reg), newFakeAgent(t, "gpu-b", reg)
	cl := newCluster(t, []*fakeAgent{a, b}, reg, Model{Name: "chat", Image: "reg.example/chat:v1", Replicas: 1})
	engineOf := func() []string {
		var out []string
		for _, ag := range []*fakeAgent{a, b} {
			if ct, ok := ag.running()["chat"]; ok {
				out = append(out, ct.Engine)
			}
		}
		return out
	}
	cl.sync(t)
	if got := engineOf(); len(got) != 1 || got[0] != "reference" {
		t.Fatalf("engines = %v", got)
	}
	cl.setCatalogue(t, Model{Name: "chat", Image: "reg.example/chat:v1", Replicas: 1, Engine: "vllm"})
	for i := 0; i < 4; i++ {
		cl.sync(t)
	}
	if got := engineOf(); len(got) != 1 || got[0] != "vllm" {
		t.Errorf("after pinning vllm, engines = %v", got)
	}
}

func TestFindPrefersTheExactName(t *testing.T) {
	cat := Catalogue{Models: []Model{
		{Name: "foo", Image: "reg.example/foo"},
		{Name: "foo/bar", Image: "reg.example/foo-bar"},
		{Name: "foo/bar/baz", Image: "reg.example/baz"},
	}}
	for model, want := range map[string]string{
		"foo": "foo", "foo/x": "foo", "foo/bar": "foo/bar", "foo/bar/x": "foo/bar", "foo/bar/baz": "foo/bar/baz",
	} {
		if m, ok := cat.find(model); !ok || m.Name != want {
			t.Errorf("find(%q) = %q, %v; want %q", model, m.Name, ok, want)
		}
	}
	if _, ok := cat.find("bar"); ok {
		t.Error("found a model for bar")
	}
}
