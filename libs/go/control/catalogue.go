package control

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/reference"
)

// Catalogue lists the images the cluster serves. It is a JSON file, as a
// Helm chart renders it into a ConfigMap; the control plane rereads it on
// every pass, so edits apply without a restart.
type Catalogue struct {
	Models []Model `json:"models"`
}

// Model is one image the cluster serves under a name.
type Model struct {
	// Name is what requests send as their model: Name for a plain image,
	// Name/<expert> for each expert of a MIX.
	Name string `json:"name"`
	// Image is a registry reference, such as ghcr.io/acme/qwen-tasks:v1.
	// When its tag moves to another digest, the replicas are replaced, one
	// at a time.
	Image string `json:"image"`
	// Replicas is how many nodes serve it. 0 starts one on its first
	// request.
	Replicas int `json:"replicas,omitempty"`
	// Engine and Device pick the engine and device; "" lets each node
	// choose (vLLM on NVIDIA GPUs, else the reference engine).
	Engine string `json:"engine,omitempty"`
	Device string `json:"device,omitempty"`
}

// nameForm is the container names the runtime accepts.
var nameForm = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$`)

// LoadCatalogue reads and checks a catalogue file.
func LoadCatalogue(path string) (*Catalogue, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Catalogue
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("catalogue %s: %w", path, err)
	}
	if err := c.Check(); err != nil {
		return nil, fmt.Errorf("catalogue %s: %w", path, err)
	}
	return &c, nil
}

// Check validates every model: a well-formed name used once, a registry
// reference, a known engine and device.
func (c *Catalogue) Check() error {
	seen := map[string]bool{}
	for _, m := range c.Models {
		if !nameForm.MatchString(m.Name) {
			return fmt.Errorf("model name %q: use letters, digits and . _ : / @ + -, starting with a letter or digit", m.Name)
		}
		if seen[m.Name] {
			return fmt.Errorf("model %q is listed twice", m.Name)
		}
		seen[m.Name] = true
		r, err := reference.Parse(m.Image)
		if err != nil {
			return fmt.Errorf("model %q: %w", m.Name, err)
		}
		if r.Host == "" {
			return fmt.Errorf("model %q: image %q names no registry", m.Name, m.Image)
		}
		if m.Replicas < 0 {
			return fmt.Errorf("model %q: replicas cannot be negative", m.Name)
		}
		switch m.Engine {
		case "", "reference", "vllm":
		default:
			return fmt.Errorf("model %q: engine %q: want reference or vllm", m.Name, m.Engine)
		}
		switch m.Device {
		case "", "cuda", "mps", "cpu":
		default:
			return fmt.Errorf("model %q: device %q: want cuda, mps or cpu", m.Name, m.Device)
		}
	}
	return nil
}

// find returns the catalogue model a request's model name belongs to:
// the model itself, or the MIX whose expert it names.
func (c *Catalogue) find(model string) (Model, bool) {
	for _, m := range c.Models {
		if model == m.Name || strings.HasPrefix(model, m.Name+"/") {
			return m, true
		}
	}
	return Model{}, false
}
