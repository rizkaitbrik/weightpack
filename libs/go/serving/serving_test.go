package serving

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

type fixture struct {
	dir    string
	store  *store.Store
	worker *copyWorker
	starts string // the fake engine appends a line here on each start
	rt     *Runtime
	gw     *httptest.Server
}

func newFixture(t *testing.T, engine ...string) *fixture {
	t.Helper()
	dir := t.TempDir()
	buildtest.Project(t, dir)
	s, err := store.Open(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	f := &fixture{dir: dir, store: s, worker: &copyWorker{}, starts: filepath.Join(dir, "starts")}
	cmd := append([]string{os.Args[0], "fake-engine", "--starts", f.starts}, engine...)
	f.rt = New(Config{
		Store: s, Worker: f.worker, Device: "cpu", Endpoint: "http://gateway/v1", StartTimeout: 20 * time.Second,
		Engines: map[string][]string{EngineReference: cmd, EngineVLLM: cmd},
	})
	f.gw = httptest.NewServer(f.rt.Gateway())
	t.Cleanup(func() {
		f.gw.Close()
		f.rt.Close()
	})
	return f
}

// build builds a Weightfile over the project's files and tags it.
func (f *fixture) build(t *testing.T, tag, weightfile string) {
	t.Helper()
	p := filepath.Join(f.dir, tag+".Weightfile")
	if err := os.WriteFile(p, []byte(weightfile), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := build.Build(context.Background(), f.store, build.Options{Weightfile: p, Tags: []string{tag}}, nil); err != nil {
		t.Fatal(err)
	}
}

func (f *fixture) run(t *testing.T, req api.RunRequest) *api.Container {
	t.Helper()
	c, err := f.rt.Run(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("run %s: %v", req.Ref, err)
	}
	return c
}

func (f *fixture) engineStarts(t *testing.T) int {
	t.Helper()
	raw, err := os.ReadFile(f.starts)
	if os.IsNotExist(err) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Count(string(raw), "\n")
}

// chat posts a chat request to the gateway and returns the status and the
// decoded body.
func (f *fixture) chat(t *testing.T, body map[string]any) (int, map[string]any) {
	t.Helper()
	raw, _ := json.Marshal(body)
	resp, err := http.Post(f.gw.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, out
}

func content(t *testing.T, body map[string]any) string {
	t.Helper()
	choices, _ := body["choices"].([]any)
	if len(choices) == 0 {
		t.Fatalf("no choices in %v", body)
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	return msg["content"].(string)
}

func (f *fixture) loaded(t *testing.T, e *engine) []string {
	t.Helper()
	resp, err := http.Get(e.url + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct{ Data []string }
	_ = json.NewDecoder(resp.Body).Decode(&out)
	slices.Sort(out.Data)
	return out.Data
}

// Images on one base share one engine: the base alone, a stack ending in a
// LoRA layer, and a mix of experts all run on it, each as its own model.
func TestImagesOnOneBaseShareAnEngine(t *testing.T) {
	f := newFixture(t)
	f.build(t, "plain", "FROM ./base\n")
	f.build(t, "french", "FROM ./base\nLAYER french ./adapters/french\nTEMPLATE ./chat.jinja\n")
	f.build(t, "domains", "FROM ./base\nMIX domain targets=mlp top_k=1\n  EXPERT legal ./adapters/legal\n  EXPERT medical ./adapters/medical\n")

	var events []string
	plain, err := f.rt.Run(context.Background(), api.RunRequest{Ref: "plain"}, func(e api.Event) { events = append(events, e.Message) })
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(plain.Models, []string{"plain"}) || plain.Status != "running" || plain.Endpoint != "http://gateway/v1" || plain.Engine != EngineReference {
		t.Errorf("plain = %+v", plain)
	}
	if !strings.Contains(strings.Join(events, "\n"), "starting the reference engine") {
		t.Errorf("events = %q", events)
	}
	french := f.run(t, api.RunRequest{Ref: "french", Name: "fr"})
	domains := f.run(t, api.RunRequest{Ref: "domains"})
	if !slices.Equal(domains.Models, []string{"domains/legal", "domains/medical"}) || !slices.Equal(french.Models, []string{"fr"}) {
		t.Errorf("models: %v, %v", french.Models, domains.Models)
	}
	if french.EngineID != plain.EngineID || domains.EngineID != plain.EngineID || french.Base != plain.Base {
		t.Errorf("engines %s %s %s, bases %s %s: want one", plain.EngineID, french.EngineID, domains.EngineID, plain.Base, french.Base)
	}
	if n := f.engineStarts(t); n != 1 {
		t.Errorf("%d engine starts, want 1", n)
	}
	e := f.rt.containers["plain"].engine
	if got := f.loaded(t, e); len(got) != 4 { // the base and three adapters
		t.Errorf("engine models %v", got)
	}

	// The gateway routes by model, renames the model both ways, and sends
	// the image's chat template.
	status, body := f.chat(t, map[string]any{"model": "fr", "messages": []any{}})
	if status != 200 || body["model"] != "fr" {
		t.Fatalf("fr: %d %v", status, body)
	}
	parts := strings.Split(content(t, body), "|")
	if !strings.HasPrefix(parts[0], "lora-") || !strings.HasSuffix(parts[0], "-french") || parts[1] != "french" || parts[2] != "{{ messages }}" {
		t.Errorf("fr reached %q", parts)
	}
	_, body = f.chat(t, map[string]any{"model": "fr", "messages": []any{}, "chat_template": "mine"})
	if got := strings.Split(content(t, body), "|")[2]; got != "mine" {
		t.Errorf("a request's own template was replaced by %q", got)
	}
	_, body = f.chat(t, map[string]any{"model": "plain", "messages": []any{}})
	if got := strings.Split(content(t, body), "|"); !strings.HasPrefix(got[0], "base-") || got[2] != "" {
		t.Errorf("plain reached %q", got)
	}
	_, body = f.chat(t, map[string]any{"model": "domains/medical", "messages": []any{}})
	if got := strings.Split(content(t, body), "|")[1]; got != "medical" || body["model"] != "domains/medical" {
		t.Errorf("domains/medical reached %q (%v)", got, body["model"])
	}

	// Streams are relayed event by event, renamed.
	raw, _ := json.Marshal(map[string]any{"model": "fr", "messages": []any{}, "stream": true})
	resp, err := http.Post(f.gw.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	var lines []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		lines = append(lines, data)
		if data == "[DONE]" {
			continue
		}
		var chunk struct {
			Model   string `json:"model"`
			Choices []struct {
				Delta struct{ Content string } `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil || chunk.Model != "fr" {
			t.Errorf("chunk %s: %v", data, err)
		}
		text.WriteString(chunk.Choices[0].Delta.Content)
	}
	resp.Body.Close()
	if lines[len(lines)-1] != "[DONE]" || !strings.Contains(text.String(), "|french|") {
		t.Errorf("stream %q gave %q", lines, text.String())
	}

	// /v1/models lists every model.
	mresp, err := http.Get(f.gw.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var models struct {
		Data []struct{ ID string } `json:"data"`
	}
	_ = json.NewDecoder(mresp.Body).Decode(&models)
	mresp.Body.Close()
	var ids []string
	for _, m := range models.Data {
		ids = append(ids, m.ID)
	}
	if !slices.Equal(ids, []string{"domains/legal", "domains/medical", "fr", "plain"}) {
		t.Errorf("models %v", ids)
	}

	// Unknown models get OpenAI's error shape.
	status, body = f.chat(t, map[string]any{"model": "nope", "messages": []any{}})
	errBody, _ := body["error"].(map[string]any)
	if status != 404 || errBody["code"] != "model_not_found" || !strings.Contains(errBody["message"].(string), "fr, plain") {
		t.Errorf("unknown model: %d %v", status, body)
	}

	// Names are unique.
	if _, err := f.rt.Run(context.Background(), api.RunRequest{Ref: "french", Name: "fr"}, nil); err == nil || !strings.Contains(err.Error(), "already running") {
		t.Errorf("duplicate name: %v", err)
	}
	if _, err := f.rt.Run(context.Background(), api.RunRequest{Ref: "plain", Name: "domains/legal"}, nil); err == nil || !strings.Contains(err.Error(), "taken by container") {
		t.Errorf("model name clash: %v", err)
	}

	// Stopping a container unloads its adapter; the engine stays for the
	// others, and stops with the last. A name given twice counts once.
	if stopped, err := f.rt.Stop([]string{"fr", "fr"}); err != nil || !slices.Equal(stopped, []string{"fr"}) {
		t.Fatalf("stop fr fr: %v %v", stopped, err)
	}
	if e.hasExited() {
		t.Fatal("stopping one container twice stopped the shared engine")
	}
	if got := f.loaded(t, e); len(got) != 3 || slices.ContainsFunc(got, func(n string) bool { return strings.HasSuffix(n, "-french") }) {
		t.Errorf("after stopping fr the engine has %v", got)
	}
	if status, _ := f.chat(t, map[string]any{"model": "fr", "messages": []any{}}); status != 404 {
		t.Errorf("a stopped model answered %d", status)
	}
	if _, err := f.rt.Stop([]string{"nope"}); err == nil {
		t.Error("stopping an unknown container succeeded")
	}
	if got := f.rt.List(); len(got) != 2 {
		t.Errorf("list = %+v", got)
	}
	if _, err := f.rt.Stop([]string{"plain", "domains"}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-e.exited:
	case <-time.After(5 * time.Second):
		t.Fatal("the engine outlived its last container")
	}
	if len(f.rt.engines) != 0 || len(f.rt.routes) != 0 {
		t.Errorf("left %d engines, %d routes", len(f.rt.engines), len(f.rt.routes))
	}
}

// Layers below an image's adapters are baked into its serving base once;
// a second image on the same chain reuses it.
func TestServingBaseBakesLowerLayers(t *testing.T) {
	f := newFixture(t)
	f.build(t, "assistant", buildtest.Assistant)
	c := f.run(t, api.RunRequest{Ref: "assistant"})
	if !slices.Equal(c.Models, []string{"assistant/legal", "assistant/medical"}) || !slices.Equal(c.Adapters, []string{"legal", "medical"}) {
		t.Errorf("container = %+v", c)
	}
	if f.worker.count() != 1 {
		t.Fatalf("%d squashes, want 1", f.worker.count())
	}
	req := f.worker.calls[0]
	if len(req.Layers) != 1 || req.Layers[0].Id != "french" {
		t.Errorf("squash baked %v, want french alone", req.Layers)
	}
	if _, err := os.Stat(filepath.Join(f.store.Root(), "run", "bases", hexOf(c.Base), "base", "model.safetensors")); err != nil {
		t.Errorf("serving base: %v", err)
	}
	// The French layer is part of the base, so the image's base layer
	// chain is not the serving base.
	_, cfg, _, _ := f.rt.read("assistant")
	if c.Base == cfg.Layers[0].Chain {
		t.Error("the serving base is the plain base layer")
	}

	// Another image with the same layers below its adapter: French, then
	// legal on top as a stack.
	f.build(t, "again", "FROM ./base\nLAYER french ./adapters/french\nLAYER legal ./adapters/legal\n")
	again := f.run(t, api.RunRequest{Ref: "again"})
	if f.worker.count() != 1 || again.EngineID != c.EngineID || again.Base != c.Base || !slices.Equal(again.Adapters, []string{"legal"}) {
		t.Errorf("second image: %d squashes, engine %s vs %s", f.worker.count(), again.EngineID, c.EngineID)
	}
}

func TestEngineThatFailsToStart(t *testing.T) {
	f := newFixture(t, "--fail")
	f.build(t, "plain", "FROM ./base\n")
	_, err := f.rt.Run(context.Background(), api.RunRequest{Ref: "plain"}, nil)
	if err == nil || !strings.Contains(err.Error(), "exited while starting") || !strings.Contains(err.Error(), "out of memory loading the model") {
		t.Fatalf("err = %v", err)
	}
	if len(f.rt.List()) != 0 || len(f.rt.engines) != 0 || len(f.rt.reserved) != 0 {
		t.Errorf("a failed run left state behind: %d containers, %d engines", len(f.rt.List()), len(f.rt.engines))
	}
}

// A run given up while its engine loads stops the engine at once.
func TestCancelledRunStopsALoadingEngine(t *testing.T) {
	f := newFixture(t, "--hang")
	f.build(t, "plain", "FROM ./base\n")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := f.rt.Run(ctx, api.RunRequest{Ref: "plain"}, nil)
		done <- err
	}()
	var e *engine
	for e == nil {
		time.Sleep(20 * time.Millisecond)
		f.rt.mu.Lock()
		for _, x := range f.rt.engines {
			e = x
		}
		f.rt.mu.Unlock()
	}
	time.Sleep(300 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Error("a cancelled run succeeded")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the run did not return")
	}
	select {
	case <-e.exited:
	case <-time.After(5 * time.Second):
		t.Fatal("the loading engine kept running")
	}
}

func TestGatewayKey(t *testing.T) {
	rt := New(Config{Store: mustStore(t), APIKey: "s3cret"})
	gw := httptest.NewServer(rt.Gateway())
	defer gw.Close()
	get := func(auth string) int {
		req, _ := http.NewRequest(http.MethodGet, gw.URL+"/v1/models", nil)
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if got := get(""); got != 401 {
		t.Errorf("no key: %d", got)
	}
	if got := get("Bearer wrong"); got != 401 {
		t.Errorf("wrong key: %d", got)
	}
	if got := get("Bearer s3cret"); got != 200 {
		t.Errorf("right key: %d", got)
	}
	resp, err := http.Get(gw.URL + "/health")
	if err != nil || resp.StatusCode != 200 {
		t.Errorf("health needs no key: %v %v", resp.StatusCode, err)
	}
	resp.Body.Close()
}

func TestLogsAndPrune(t *testing.T) {
	f := newFixture(t)
	f.build(t, "french", "FROM ./base\nLAYER french ./adapters/french\n")
	f.run(t, api.RunRequest{Ref: "french"})

	var log strings.Builder
	if err := f.rt.Logs(context.Background(), "french", false, &log, func() {}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(log.String(), "fake engine listening") || !strings.Contains(log.String(), "wpkd starting reference engine") {
		t.Errorf("log = %q", log.String())
	}
	// Following stops when the caller does.
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if err := f.rt.Logs(ctx, "french", true, io.Discard, func() {}); err != nil {
		t.Errorf("follow: %v", err)
	}
	if err := f.rt.Logs(context.Background(), "nope", false, io.Discard, func() {}); err == nil {
		t.Error("logs of an unknown container")
	}

	// Files in use stay; once nothing runs they go.
	if n, err := f.rt.PruneFiles(false); err != nil || n != 0 {
		t.Errorf("prune while running freed %d (%v)", n, err)
	}
	if _, err := f.rt.Stop([]string{"french"}); err != nil {
		t.Fatal(err)
	}
	dry, err := f.rt.PruneFiles(true)
	if err != nil || dry == 0 {
		t.Fatalf("dry run: %d %v", dry, err)
	}
	if n, err := f.rt.PruneFiles(false); err != nil || n != dry {
		t.Errorf("prune freed %d, dry run said %d (%v)", n, dry, err)
	}
	for _, sub := range []string{"bases", "adapters", "logs"} {
		if entries, _ := os.ReadDir(filepath.Join(f.store.Root(), "run", sub)); len(entries) != 0 {
			t.Errorf("%s left: %v", sub, entries)
		}
	}
}

func TestPick(t *testing.T) {
	rt := New(Config{Store: mustStore(t), Engines: map[string][]string{EngineVLLM: {"no-such-vllm-binary"}}})
	cases := []struct {
		engine, device      string
		wantEngine, wantDev string
		wantErr             string
	}{
		{"", "cpu", EngineReference, "cpu", ""},
		{"", "mps", EngineReference, "mps", ""},
		{"", "cuda", EngineReference, "cuda", ""}, // vllm is not installed
		{"vllm", "cuda", EngineVLLM, "cuda", ""},
		{"vllm", "mps", "", "", "NVIDIA GPUs"},
		{"", "tpu", "", "", "unknown device"},
		{"tgi", "cpu", "", "", "unknown engine"},
	}
	for _, c := range cases {
		kind, dev, err := rt.pick(c.engine, c.device)
		if c.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("%s/%s: err %v, want %q", c.engine, c.device, err, c.wantErr)
			}
			continue
		}
		if err != nil || kind != c.wantEngine || dev != c.wantDev {
			t.Errorf("%s/%s: got %s/%s %v", c.engine, c.device, kind, dev, err)
		}
	}
	if d := DetectDevice(); d != "cuda" && d != "mps" && d != "cpu" {
		t.Errorf("DetectDevice = %q", d)
	}
}

func TestEngineCommands(t *testing.T) {
	ref := (&engine{kind: EngineReference, device: "mps", served: "base-1"}).command([]string{"uv", "run", "weightpack-engine"}, "/m", 9000)
	want := []string{"uv", "run", "weightpack-engine", "--model", "/m", "--host", "127.0.0.1", "--port", "9000", "--device", "mps", "--served-model-name", "base-1", "--exit-with-stdin"}
	if !slices.Equal(ref.Args, want) {
		t.Errorf("reference: %q", ref.Args)
	}
	v := (&engine{kind: EngineVLLM, device: "cuda", served: "base-1", maxRank: 64}).command([]string{"vllm", "serve"}, "/m", 9000)
	want = []string{"vllm", "serve", "/m", "--host", "127.0.0.1", "--port", "9000", "--served-model-name", "base-1", "--enable-lora", "--max-lora-rank", "64", "--max-loras", "8", "--trust-request-chat-template"}
	if !slices.Equal(v.Args, want) || !slices.Contains(v.Env, "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True") {
		t.Errorf("vllm: %q", v.Args)
	}
	for r, want := range map[int]int{0: 16, 8: 16, 17: 32, 64: 64, 300: 320, 512: 512} {
		if got, err := vllmRank(r); err != nil || got != want {
			t.Errorf("vllmRank(%d) = %d %v, want %d", r, got, err, want)
		}
	}
	if _, err := vllmRank(1024); err == nil {
		t.Error("rank 1024 accepted")
	}
}

func mustStore(t *testing.T) *store.Store {
	t.Helper()
	s, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestMaxEngines(t *testing.T) {
	f := newFixture(t)
	f.rt.cfg.MaxEngines = 1
	f.build(t, "french", "FROM ./base\nLAYER french ./adapters/french\n")
	f.build(t, "assistant", buildtest.Assistant)
	f.run(t, api.RunRequest{Ref: "french"})
	if f.rt.Engines() != 1 || f.rt.Device() != "cpu" {
		t.Errorf("engines = %d, device %q", f.rt.Engines(), f.rt.Device())
	}
	// The assistant's serving base bakes French in: another base, so
	// another engine, over the cap.
	if _, err := f.rt.Run(context.Background(), api.RunRequest{Ref: "assistant"}, nil); err == nil || !strings.Contains(err.Error(), "--max-engines") {
		t.Fatalf("a second base over the cap: %v", err)
	}
	// Another image on the running base only adds an adapter.
	f.build(t, "french2", "FROM ./base\nLAYER french ./adapters/french\n")
	f.run(t, api.RunRequest{Ref: "french2"})
	if _, err := f.rt.Stop([]string{"french", "french2"}); err != nil {
		t.Fatal(err)
	}
	f.run(t, api.RunRequest{Ref: "assistant"})
}
