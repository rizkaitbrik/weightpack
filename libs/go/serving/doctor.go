package serving

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	goruntime "runtime"
	"strings"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

// engineReport is what `weightpack-engine --check` prints.
type engineReport struct {
	Engine       string   `json:"engine"`
	Python       string   `json:"python"`
	Torch        string   `json:"torch"`
	Transformers string   `json:"transformers"`
	PEFT         string   `json:"peft"`
	CUDA         bool     `json:"cuda"`
	CUDADevices  []string `json:"cuda_devices"`
	MPS          bool     `json:"mps"`
}

// Doctor checks the device and the engines that would serve on it.
func (rt *Runtime) Doctor(ctx context.Context) []api.Check {
	device := rt.cfg.Device
	how := "set with wpkd --device"
	if device == "" {
		rt.detectOnce.Do(func() { rt.detected = DetectDevice() })
		device, how = rt.detected, "detected"
	}
	checks := []api.Check{{Name: "device", Status: api.CheckOK, Detail: fmt.Sprintf("%s (%s): %s", device, how, describeDevice(device))}}
	checks = append(checks, rt.checkReference(ctx, device))
	checks = append(checks, rt.checkVLLM(ctx, device))
	return checks
}

// describeDevice names the hardware behind a device.
func describeDevice(device string) string {
	switch device {
	case "cuda":
		out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader").Output()
		if err == nil {
			gpus := strings.Split(strings.TrimSpace(string(out)), "\n")
			return fmt.Sprintf("%d NVIDIA GPU(s): %s", len(gpus), strings.Join(gpus, "; "))
		}
		return "NVIDIA GPU"
	case "mps":
		chip, _ := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
		return strings.TrimSpace(string(chip)) + " GPU (Metal)"
	default:
		return fmt.Sprintf("%d CPU cores (%s/%s)", goruntime.NumCPU(), goruntime.GOOS, goruntime.GOARCH)
	}
}

func (rt *Runtime) checkReference(ctx context.Context, device string) api.Check {
	c := api.Check{Name: "reference engine"}
	cmd := rt.cfg.Engines[EngineReference]
	if _, err := exec.LookPath(cmd[0]); err != nil {
		c.Status = api.CheckFail
		c.Detail = fmt.Sprintf("%s not found", cmd[0])
		c.Fix = `pip install "weightpack-worker[serve]", or point wpkd --engine-reference at it (for a source checkout: "uv run --project python/worker --extra serve weightpack-engine")`
		return c
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	run := exec.CommandContext(ctx, cmd[0], append(append([]string{}, cmd[1:]...), "--check")...)
	var stderr bytes.Buffer
	run.Stderr = &stderr
	out, err := run.Output()
	var r engineReport
	if err == nil {
		err = json.Unmarshal(lastLine(out), &r)
	}
	if err != nil {
		c.Status = api.CheckFail
		c.Detail = fmt.Sprintf("%s --check failed: %v %s", strings.Join(cmd, " "), err, lastLines(stderr.String(), 3))
		c.Fix = `install its libraries: pip install "weightpack-worker[serve]" (torch, transformers, peft)`
		return c
	}
	c.Status = api.CheckOK
	c.Detail = fmt.Sprintf("weightpack-engine %s: torch %s, transformers %s, peft %s", r.Engine, r.Torch, r.Transformers, r.PEFT)
	switch {
	case device == "cuda" && !r.CUDA:
		c.Status = api.CheckWarn
		c.Detail += "; torch sees no CUDA GPU, so it would run on the CPU"
		c.Fix = "install a CUDA build of torch (https://pytorch.org/get-started/locally/) in the engine's environment"
	case device == "mps" && !r.MPS:
		c.Status = api.CheckWarn
		c.Detail += "; torch cannot use Metal here"
		c.Fix = "use a recent arm64 build of torch (pip install -U torch) on macOS 14 or later"
	case device == "cuda":
		c.Detail += "; CUDA: " + strings.Join(r.CUDADevices, ", ")
	case device == "mps":
		c.Detail += "; Metal available"
	}
	return c
}

func (rt *Runtime) checkVLLM(ctx context.Context, device string) api.Check {
	c := api.Check{Name: "vllm"}
	if device != "cuda" {
		c.Status, c.Detail = api.CheckSkip, "serves on NVIDIA GPUs only; "+device+" uses the reference engine"
		return c
	}
	cmd := rt.cfg.Engines[EngineVLLM]
	if _, err := exec.LookPath(cmd[0]); err != nil {
		c.Status = api.CheckWarn
		c.Detail = cmd[0] + " not found, so containers use the reference engine, which runs one request at a time"
		c.Fix = "pip install vllm, or point wpkd --engine-vllm at it"
		return c
	}
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	out, err := exec.CommandContext(ctx, cmd[0], "--version").Output()
	if err != nil {
		c.Status, c.Detail = api.CheckWarn, fmt.Sprintf("%s --version failed: %v", cmd[0], err)
		c.Fix = "check the vLLM install: " + cmd[0] + " --version"
		return c
	}
	c.Status, c.Detail = api.CheckOK, "vLLM "+strings.TrimSpace(string(lastLine(out)))+", the default engine on this machine"
	return c
}

// lastLine returns the last non-empty line, where tools print their answer
// after any warnings.
func lastLine(b []byte) []byte {
	lines := bytes.Split(bytes.TrimSpace(b), []byte("\n"))
	return bytes.TrimSpace(lines[len(lines)-1])
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return strings.Join(lines[max(0, len(lines)-n):], " | ")
}
