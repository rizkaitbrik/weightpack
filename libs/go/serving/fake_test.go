package serving

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// The test binary doubles as a fake engine: `<test binary> fake-engine
// [--starts FILE] [--fail] <engine arguments>`. It speaks the subset of the
// OpenAI API wpkd uses, and its replies say which model, adapter and chat
// template a request reached.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "fake-engine" {
		os.Exit(fakeEngine(os.Args[2:]))
	}
	os.Exit(m.Run())
}

func fakeEngine(args []string) int {
	fs := flag.NewFlagSet("fake-engine", flag.ContinueOnError)
	starts := fs.String("starts", "", "append a line to this file on start")
	fail := fs.Bool("fail", false, "exit at once, as a broken engine does")
	hang := fs.Bool("hang", false, "never answer, as an engine loading a large model")
	rejectMixes := fs.Bool("reject-mixes", false, "refuse every mix, as for experts with saved modules")
	model := fs.String("model", "", "")
	host := fs.String("host", "127.0.0.1", "")
	port := fs.Int("port", 0, "")
	device := fs.String("device", "", "")
	served := fs.String("served-model-name", "base", "")
	exitWithStdin := fs.Bool("exit-with-stdin", false, "")
	noPlugin := fs.Bool("no-plugin", false, "as vLLM without weightpack-vllm: refuse mix directories")
	// vLLM's own flags; its model directory comes first, unflagged.
	fs.Bool("enable-lora", false, "")
	maxLoRARank := fs.Int("max-lora-rank", 0, "")
	fs.Int("max-loras", 0, "")
	fs.Bool("trust-request-chat-template", false, "")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *model == "" && fs.NArg() > 0 {
		*model = fs.Arg(0)
		if err := fs.Parse(fs.Args()[1:]); err != nil {
			return 2
		}
	}
	if *fail {
		fmt.Fprintln(os.Stderr, "fake engine: out of memory loading the model")
		return 1
	}
	if *hang {
		fmt.Fprintln(os.Stderr, "fake engine: loading the model")
		// A sleep, not select {}: with no other goroutine, Go takes an
		// empty select for a deadlock and exits, and the engine would die
		// at once instead of loading forever.
		for {
			time.Sleep(time.Hour)
		}
	}
	if _, err := os.Stat(filepath.Join(*model, "model.safetensors")); err != nil {
		fmt.Fprintf(os.Stderr, "fake engine: no model in %s\n", *model)
		return 1
	}
	if *starts != "" {
		f, err := os.OpenFile(*starts, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err == nil {
			fmt.Fprintln(f, *model, *device)
			_ = f.Close()
		}
	}
	if *exitWithStdin {
		go func() {
			_, _ = io.Copy(io.Discard, os.Stdin)
			os.Exit(0)
		}()
	}

	var mu sync.Mutex
	adapters := map[string]string{} // name -> path
	type mix struct {
		Experts []string `json:"experts"`
		TopK    int      `json:"top_k"`
		Targets []string `json:"targets"`
		Blocks  string   `json:"blocks"`
		Router  string   `json:"router"`
		// RouterFile holds a trained router; it must exist.
		RouterFile string `json:"router_file"`
	}
	mixes := map[string]mix{}
	routerOK := func(m mix) bool {
		if m.Router == "trained" {
			_, err := os.Stat(m.RouterFile)
			return err == nil
		}
		return m.Router == "arrow" && m.RouterFile == ""
	}
	answer := func(w http.ResponseWriter, status int, v any) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(v)
	}
	fail400 := func(w http.ResponseWriter, status int, msg string) {
		answer(w, status, map[string]any{"error": map[string]any{"message": msg}})
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { answer(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		ids := []string{*served}
		for n := range adapters {
			ids = append(ids, n)
		}
		for n := range mixes {
			ids = append(ids, n)
		}
		answer(w, 200, map[string]any{"data": ids})
	})
	mux.HandleFunc("POST /v1/load_lora_adapter", func(w http.ResponseWriter, r *http.Request) {
		var b struct{ LoraName, LoraPath string }
		_ = json.NewDecoder(r.Body).Decode(&struct {
			LoraName *string `json:"lora_name"`
			LoraPath *string `json:"lora_path"`
		}{&b.LoraName, &b.LoraPath})
		if _, err := os.Stat(filepath.Join(b.LoraPath, "adapter_config.json")); err != nil {
			fail400(w, 400, "no adapter at "+b.LoraPath)
			return
		}
		// A mix directory, as the weightpack-vllm plugin reads it: its
		// experts' adapters must exist and fit --max-lora-rank.
		if raw, err := os.ReadFile(filepath.Join(b.LoraPath, "weightpack_mix.json")); err == nil {
			if *noPlugin {
				fail400(w, 400, b.LoraPath+" doesn't contain tensors")
				return
			}
			var spec struct {
				TopK       int      `json:"top_k"`
				Targets    []string `json:"targets"`
				Blocks     string   `json:"blocks"`
				Router     string   `json:"router"`
				RouterFile string   `json:"router_file"`
				Experts    []struct {
					ID   string `json:"id"`
					Path string `json:"path"`
				} `json:"experts"`
			}
			var cfg struct {
				R int `json:"r"`
			}
			_ = json.Unmarshal(raw, &spec)
			m := mix{TopK: spec.TopK, Targets: spec.Targets, Blocks: spec.Blocks, Router: spec.Router, RouterFile: spec.RouterFile}
			if !routerOK(m) {
				fail400(w, 400, "mix router "+spec.Router+" in "+b.LoraPath)
				return
			}
			for _, e := range spec.Experts {
				if _, err := os.Stat(filepath.Join(e.Path, "adapter_config.json")); err != nil {
					fail400(w, 400, "mix expert "+e.ID+": no adapter at "+e.Path)
					return
				}
				m.Experts = append(m.Experts, e.ID)
			}
			cfgRaw, _ := os.ReadFile(filepath.Join(b.LoraPath, "adapter_config.json"))
			_ = json.Unmarshal(cfgRaw, &cfg)
			if *maxLoRARank > 0 && cfg.R > *maxLoRARank {
				fail400(w, 400, fmt.Sprintf("LoRA rank %d is greater than max_lora_rank %d", cfg.R, *maxLoRARank))
				return
			}
			mu.Lock()
			defer mu.Unlock()
			mixes[b.LoraName] = m
			w.WriteHeader(200)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if _, ok := adapters[b.LoraName]; ok {
			fail400(w, 400, "adapter "+b.LoraName+" is loaded already")
			return
		}
		adapters[b.LoraName] = b.LoraPath
		w.WriteHeader(200)
	})
	mux.HandleFunc("POST /v1/unload_lora_adapter", func(w http.ResponseWriter, r *http.Request) {
		var b struct {
			LoraName string `json:"lora_name"`
		}
		_ = json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		defer mu.Unlock()
		if _, ok := mixes[b.LoraName]; ok { // a mix loaded as an adapter, as on vLLM
			delete(mixes, b.LoraName)
			w.WriteHeader(200)
			return
		}
		if _, ok := adapters[b.LoraName]; !ok {
			fail400(w, 404, "adapter "+b.LoraName+" is not loaded")
			return
		}
		for n, m := range mixes {
			for _, e := range m.Experts {
				if e == b.LoraName {
					fail400(w, 400, "adapter "+b.LoraName+" is an expert of mix "+n)
					return
				}
			}
		}
		delete(adapters, b.LoraName)
		w.WriteHeader(200)
	})
	mux.HandleFunc("POST /v1/load_mix", func(w http.ResponseWriter, r *http.Request) {
		var b struct {
			Name string `json:"mix_name"`
			mix
		}
		_ = json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		defer mu.Unlock()
		for _, e := range b.Experts {
			if _, ok := adapters[e]; !ok {
				fail400(w, 404, "unknown model or field '"+e+"'")
				return
			}
		}
		if *rejectMixes {
			fail400(w, 400, "expert 'x' replaces whole modules (lm_head); token routing mixes LoRA updates only")
			return
		}
		if _, ok := mixes[b.Name]; ok || b.TopK < 1 || len(b.Targets) == 0 || !routerOK(b.mix) {
			fail400(w, 400, "bad mix "+b.Name)
			return
		}
		mixes[b.Name] = b.mix
		w.WriteHeader(200)
	})
	mux.HandleFunc("POST /v1/unload_mix", func(w http.ResponseWriter, r *http.Request) {
		var b struct {
			Name string `json:"mix_name"`
		}
		_ = json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		defer mu.Unlock()
		if _, ok := mixes[b.Name]; !ok {
			fail400(w, 404, "mix "+b.Name+" is not loaded")
			return
		}
		delete(mixes, b.Name)
		w.WriteHeader(200)
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var b struct {
			Model        string `json:"model"`
			ChatTemplate string `json:"chat_template"`
			Stream       bool   `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		path, isAdapter := adapters[b.Model]
		m, isMix := mixes[b.Model]
		mu.Unlock()
		if b.Model != *served && !isAdapter && !isMix {
			fail400(w, 404, "model "+b.Model+" does not exist")
			return
		}
		content := b.Model + "|" + filepath.Base(path) + "|" + b.ChatTemplate
		if isMix {
			routed := fmt.Sprintf("top%d of %s on %s", m.TopK, strings.Join(m.Experts, ","), strings.Join(m.Targets, ","))
			if m.Router == "trained" {
				routed += " by " + filepath.Base(m.RouterFile)
			}
			content = b.Model + "|" + routed + "|" + b.ChatTemplate
		}
		if !b.Stream {
			answer(w, 200, map[string]any{"id": "c1", "object": "chat.completion", "model": b.Model,
				"choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": content}}}})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		for _, part := range []string{content[:len(content)/2], content[len(content)/2:]} {
			chunk, _ := json.Marshal(map[string]any{"id": "c1", "object": "chat.completion.chunk", "model": b.Model,
				"choices": []any{map[string]any{"index": 0, "delta": map[string]string{"content": part}}}})
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			w.(http.Flusher).Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	ln, err := net.Listen("tcp", net.JoinHostPort(*host, strconv.Itoa(*port)))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "fake engine listening on %s, serving %s\n", ln.Addr(), *served)
	_ = http.Serve(ln, mux)
	return 0
}

// copyWorker stands in for the Python worker's squash: it "merges" every
// base tensor a layer touches into an unchanged copy, which squash accepts.
type copyWorker struct {
	workerpb.WorkerClient
	mu    sync.Mutex
	calls []*workerpb.SquashRequest
}

func (c *copyWorker) Squash(_ context.Context, req *workerpb.SquashRequest, _ ...grpc.CallOption) (*workerpb.SquashResponse, error) {
	c.mu.Lock()
	c.calls = append(c.calls, req)
	c.mu.Unlock()
	targets := map[string]bool{}
	for _, l := range req.Layers {
		for _, b := range l.Blobs {
			h, err := safetensors.Open(b.Path)
			if err != nil {
				return nil, err
			}
			for _, t := range h.Tensors {
				if m, ok := strings.CutSuffix(t.Name, ".lora_A.weight"); ok {
					targets[m+".weight"] = true
				}
			}
			_ = h.Close()
		}
	}
	resp := &workerpb.SquashResponse{}
	for _, b := range req.Base {
		h, err := safetensors.Open(b.Path)
		if err != nil {
			return nil, err
		}
		for _, t := range h.Tensors {
			if !targets[t.Name] {
				continue
			}
			file := fmt.Sprintf("%d.safetensors", len(resp.Tensors))
			r, _, err := safetensors.NewReader([]safetensors.Entry{{Name: t.Name, DType: t.DType, Shape: t.Shape, Data: h.Data(t)}}, nil)
			if err != nil {
				return nil, err
			}
			out, err := os.Create(filepath.Join(req.OutDir, file))
			if err != nil {
				return nil, err
			}
			_, err = io.Copy(out, r)
			_ = out.Close()
			if err != nil {
				return nil, err
			}
			resp.Tensors = append(resp.Tensors, &workerpb.MergedTensor{Name: t.Name, File: file})
		}
		_ = h.Close()
	}
	return resp, nil
}

func (c *copyWorker) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.calls)
}
