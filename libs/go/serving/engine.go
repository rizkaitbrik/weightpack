package serving

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

// Engine kinds.
const (
	// EngineReference is weightpack-engine: transformers and PEFT on cuda,
	// mps or cpu.
	EngineReference = "reference"
	// EngineVLLM is vLLM with runtime LoRA loading, on NVIDIA GPUs.
	EngineVLLM = "vllm"
)

// DefaultEngines are the commands that start each engine kind. wpkd adds
// the model directory, port and serving options.
var DefaultEngines = map[string][]string{
	EngineReference: {"weightpack-engine"},
	EngineVLLM:      {"vllm", "serve"},
}

// vllmRanks are the LoRA ranks vLLM accepts for --max-lora-rank.
var vllmRanks = []int{16, 32, 64, 128, 256, 320, 512}

// engine is one engine process serving one serving base on one device.
type engine struct {
	id      string
	key     string
	kind    string
	device  string
	base    string // the serving base's chain ID
	served  string // the base's model name inside the engine
	logPath string
	maxRank int // vLLM's --max-lora-rank; 0 for the reference engine

	url   string
	cmd   *exec.Cmd
	stdin io.Closer

	ready     chan struct{} // closed once started, or failed to
	startErr  error         // set before ready closes
	abort     chan struct{} // closed by stop, to give up a start
	abortOnce sync.Once
	exited    chan struct{} // closed when the process has exited
	exitErr   error         // set before exited closes

	refs int // containers and runs waiting on it; guarded by Runtime.mu

	amu      sync.Mutex
	adapters map[string]int // adapter name -> containers using it
	mixes    map[string]int // mix name -> containers using it
}

func (e *engine) hasExited() bool {
	select {
	case <-e.exited:
		return true
	default:
		return false
	}
}

// exitError says why the engine is gone, or "" while it runs.
func (e *engine) exitError() string {
	if !e.hasExited() {
		return ""
	}
	if e.exitErr != nil {
		return "engine exited: " + e.exitErr.Error()
	}
	return "engine exited"
}

// command builds the engine's command line.
func (e *engine) command(base []string, modelDir string, port int) *exec.Cmd {
	p := strconv.Itoa(port)
	var args []string
	switch e.kind {
	case EngineVLLM:
		// Images sharing a base may differ in chat template, so the gateway
		// sends each image's template with its requests.
		args = []string{modelDir, "--host", "127.0.0.1", "--port", p, "--served-model-name", e.served,
			"--enable-lora", "--max-lora-rank", strconv.Itoa(e.maxRank), "--max-loras", "8",
			"--trust-request-chat-template"}
	default:
		args = []string{"--model", modelDir, "--host", "127.0.0.1", "--port", p, "--device", e.device,
			"--served-model-name", e.served, "--exit-with-stdin"}
	}
	cmd := exec.Command(base[0], append(append([]string{}, base[1:]...), args...)...)
	cmd.Env = os.Environ()
	if e.kind == EngineVLLM {
		// Lets wpkd load and unload adapters while vLLM runs.
		cmd.Env = append(cmd.Env, "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True")
	}
	setProcAttr(cmd)
	return cmd
}

// start launches the engine and waits until it answers /health, then
// closes ready. Its output goes to its log file.
func (e *engine) start(base []string, modelDir string, timeout time.Duration) {
	defer close(e.ready)
	fail := func(err error) {
		e.startErr = err
	}
	if len(base) == 0 {
		fail(fmt.Errorf("no command for the %s engine", e.kind))
		close(e.exited)
		return
	}
	logFile, err := os.OpenFile(e.logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fail(err)
		close(e.exited)
		return
	}
	port, err := freePort()
	if err != nil {
		_ = logFile.Close()
		fail(err)
		close(e.exited)
		return
	}
	cmd := e.command(base, modelDir, port)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	stdin, err := cmd.StdinPipe()
	if err != nil {
		_ = logFile.Close()
		fail(err)
		close(e.exited)
		return
	}
	fmt.Fprintf(logFile, "--- wpkd starting %s engine %s: %s\n", e.kind, e.id, strings.Join(cmd.Args, " "))
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		fail(fmt.Errorf("start the %s engine (%s): %w; set its command with wpkd --engine-%s", e.kind, base[0], err, e.kind))
		close(e.exited)
		return
	}
	e.cmd, e.stdin, e.url = cmd, stdin, "http://127.0.0.1:"+strconv.Itoa(port)
	go func() {
		err := cmd.Wait()
		fmt.Fprintf(logFile, "--- %s engine %s exited: %v\n", e.kind, e.id, err)
		_ = logFile.Close()
		e.exitErr = err
		close(e.exited)
	}()

	hc := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for {
		resp, err := hc.Get(e.url + "/health")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		select {
		case <-e.exited:
			fail(fmt.Errorf("the %s engine exited while starting (%v); its last output:\n%s", e.kind, e.exitErr, tail(e.logPath, 20)))
			return
		case <-e.abort:
			e.kill()
			fail(fmt.Errorf("the %s engine was stopped while starting", e.kind))
			return
		case <-time.After(250 * time.Millisecond):
		}
		if time.Now().After(deadline) {
			e.kill()
			fail(fmt.Errorf("the %s engine did not answer within %s; its last output:\n%s", e.kind, timeout, tail(e.logPath, 20)))
			return
		}
	}
}

// stop asks the engine to exit, and kills it after 15 seconds. An engine
// still starting is killed at once.
func (e *engine) stop() {
	e.abortOnce.Do(func() { close(e.abort) })
	<-e.ready
	if e.cmd == nil || e.cmd.Process == nil {
		return
	}
	signalGroup(e.cmd, syscall.SIGTERM)
	select {
	case <-e.exited:
	case <-time.After(15 * time.Second):
		e.kill()
	}
}

func (e *engine) kill() {
	signalGroup(e.cmd, syscall.SIGKILL)
	<-e.exited
	if e.stdin != nil {
		_ = e.stdin.Close()
	}
}

// loadAdapter loads a LoRA adapter into the engine under name, or counts
// one more user of it if it is loaded already.
func (e *engine) loadAdapter(ctx context.Context, name, dir string) error {
	e.amu.Lock()
	defer e.amu.Unlock()
	if e.adapters[name] > 0 {
		e.adapters[name]++
		return nil
	}
	if err := e.post(ctx, "/v1/load_lora_adapter", map[string]string{"lora_name": name, "lora_path": dir}); err != nil {
		return fmt.Errorf("load adapter into the %s engine: %w", e.kind, err)
	}
	e.adapters[name] = 1
	return nil
}

// loadMix names, in the reference engine, a MIX of loaded adapters that
// routes each token to its own experts, or counts one more user of it.
func (e *engine) loadMix(ctx context.Context, name string, experts []string, mix spec.Layer) error {
	e.amu.Lock()
	defer e.amu.Unlock()
	if e.mixes[name] > 0 {
		e.mixes[name]++
		return nil
	}
	body := map[string]any{
		"mix_name": name, "experts": experts, "top_k": mix.TopK,
		"targets": mix.Targets, "blocks": mix.Blocks, "router": mix.Router,
	}
	if err := e.post(ctx, "/v1/load_mix", body); err != nil {
		return fmt.Errorf("set up per-token routing in the %s engine: %w", e.kind, err)
	}
	e.mixes[name] = 1
	return nil
}

// unloadMix drops one user of a mix and unloads it after the last.
func (e *engine) unloadMix(name string) {
	e.amu.Lock()
	defer e.amu.Unlock()
	if e.mixes[name]--; e.mixes[name] > 0 {
		return
	}
	delete(e.mixes, name)
	if !e.hasExited() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = e.post(ctx, "/v1/unload_mix", map[string]string{"mix_name": name})
	}
}

// unloadAdapter drops one user of an adapter and unloads it after the last.
func (e *engine) unloadAdapter(name string) {
	e.amu.Lock()
	defer e.amu.Unlock()
	if e.adapters[name]--; e.adapters[name] > 0 {
		return
	}
	delete(e.adapters, name)
	if !e.hasExited() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = e.post(ctx, "/v1/unload_lora_adapter", map[string]string{"lora_name": name})
	}
}

func (e *engine) post(ctx context.Context, path string, body any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url+path, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("%s: %s", resp.Status, engineMessage(msg))
	}
	return nil
}

// engineMessage pulls the message out of an engine's error body: OpenAI's
// {"error": {"message": ...}}, FastAPI's {"detail": ...}, or plain text.
func engineMessage(body []byte) string {
	var oai struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
		Detail  any    `json:"detail"`
	}
	if json.Unmarshal(body, &oai) == nil {
		switch {
		case oai.Error.Message != "":
			return oai.Error.Message
		case oai.Message != "":
			return oai.Message
		case oai.Detail != nil:
			return fmt.Sprint(oai.Detail)
		}
	}
	return strings.TrimSpace(string(body))
}

// freePort asks the kernel for a free loopback port.
func freePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

// tail returns the last n lines of a file.
func tail(path string, n int) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	const window = 64 << 10
	if info, err := f.Stat(); err == nil && info.Size() > window {
		_, _ = f.Seek(info.Size()-window, io.SeekStart)
	}
	raw, _ := io.ReadAll(f)
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	return strings.Join(lines[max(0, len(lines)-n):], "\n")
}

// vllmRank is the --max-lora-rank that fits adapters of rank r.
func vllmRank(r int) (int, error) {
	for _, v := range vllmRanks {
		if r <= v {
			return v, nil
		}
	}
	return 0, errors.New("vLLM serves LoRA ranks up to 512")
}
