package build

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
)

// peftConfig is the part of a PEFT adapter_config.json the builder reads.
type peftConfig struct {
	PeftType      string         `json:"peft_type"`
	R             int            `json:"r"`
	Alpha         float64        `json:"lora_alpha"`
	RSLoRA        bool           `json:"use_rslora"`
	DoRA          bool           `json:"use_dora"`
	FanInFanOut   bool           `json:"fan_in_fan_out"`
	Bias          string         `json:"bias"`
	ModulesToSave []string       `json:"modules_to_save"`
	RankPattern   map[string]any `json:"rank_pattern"`
	AlphaPattern  map[string]any `json:"alpha_pattern"`
}

// loraTensor matches PEFT's saved names, such as
// base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight.
var loraTensor = regexp.MustCompile(`^(?:base_model\.model\.)?(.+)\.lora_(A|B)\.weight$`)

// canonicalTargets maps module names to the names Weightfiles and configs
// use. Other module names are kept as they are.
var canonicalTargets = map[string]string{
	"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
	"gate_proj": "gate", "up_proj": "up", "down_proj": "down",
}

var targetOrder = []string{"q", "k", "v", "o", "gate", "up", "down"}

// blockIndex finds the transformer block a module belongs to.
var blockIndex = regexp.MustCompile(`(?:^|\.)(?:layers|h|blocks)\.(\d+)\.`)

type loraModule struct {
	name string // module path in base naming, e.g. model.layers.0.mlp.up_proj
	a, b *safetensors.Tensor
}

func (b *builder) ingestLoRA(step string, src weightfile.Layer, layer, parent *spec.Layer, base *baseModel, mix *weightfile.Mix) error {
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: src.Line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	dir, err := b.sourceDir(step, src.Line, src.Source, fetchAdapter)
	if err != nil {
		return err
	}
	pc, err := readPEFTConfig(dir)
	if err != nil {
		return fail("%v", err)
	}
	weights := filepath.Join(dir, "adapter_model.safetensors")
	if _, err := os.Stat(weights); errors.Is(err, fs.ErrNotExist) {
		if _, binErr := os.Stat(filepath.Join(dir, "adapter_model.bin")); binErr == nil {
			return fail("found adapter_model.bin but no adapter_model.safetensors: pickle weights can run code when loaded, so save the adapter with safe_serialization=True")
		}
		return fail("no adapter_model.safetensors in %s", dir)
	}
	f, err := safetensors.Open(weights)
	if err != nil {
		return fail("%v", err)
	}
	defer f.Close()

	modules := map[string]*loraModule{}
	var saved []*safetensors.Tensor // modules_to_save: full copies that replace base tensors
	for i := range f.Tensors {
		t := &f.Tensors[i]
		m := loraTensor.FindStringSubmatch(t.Name)
		if m == nil {
			if isSavedModule(t, pc.ModulesToSave, base) {
				saved = append(saved, t)
				continue
			}
			return fail("tensor %s is not a LoRA A or B matrix, nor a module_to_save copy of a base tensor; embedding LoRAs, DoRA magnitudes and trained biases are not supported yet", t.Name)
		}
		mod := modules[m[1]]
		if mod == nil {
			mod = &loraModule{name: m[1]}
			modules[m[1]] = mod
		}
		if m[2] == "A" {
			mod.a = t
		} else {
			mod.b = t
		}
	}
	if len(modules) == 0 && len(saved) == 0 {
		return fail("adapter has no tensors")
	}
	names := make([]string, 0, len(modules))
	for n := range modules {
		names = append(names, n)
	}
	sort.Strings(names)

	// Check every module against the base before storing anything.
	targets := map[string]bool{}
	for _, n := range names {
		mod := modules[n]
		if mod.a == nil || mod.b == nil {
			return fail("%s has only one of lora_A and lora_B", n)
		}
		if err := checkModule(mod, pc.R, base); err != nil {
			return fail("%v", err)
		}
		targets[targetName(n)] = true
	}
	if mix != nil {
		if outside := outsideMix(names, mix); outside > 0 {
			b.progress(api.Event{Step: step, Warning: true, Message: fmt.Sprintf(
				"%d of %d modules lie outside MIX %s (targets=%s blocks=%s)",
				outside, len(names), mix.ID, strings.Join(mix.Targets, ","), blocksOrAll(mix.Blocks))})
		}
	}

	for _, n := range names {
		mod := modules[n]
		entries := []safetensors.Entry{
			{Name: n + ".lora_A.weight", DType: mod.a.DType, Shape: mod.a.Shape, Data: f.Data(*mod.a)},
			{Name: n + ".lora_B.weight", DType: mod.b.DType, Shape: mod.b.Shape, Data: f.Data(*mod.b)},
		}
		d, err := b.putTensors(entries, spec.MediaTypeLoRA,
			map[string]string{spec.AnnotationLayer: layer.ID, spec.AnnotationTensor: n + ".weight"})
		if err != nil {
			return fail("%s: %v", n, err)
		}
		b.blobs[layer.ID] = append(b.blobs[layer.ID], d)
	}

	// A saved module replaces its base tensor outright: a tensor.v1 blob in
	// the LoRA layer, under the base tensor's name.
	sort.Slice(saved, func(i, j int) bool { return saved[i].Name < saved[j].Name })
	for _, t := range saved {
		name := strings.TrimPrefix(t.Name, peftPrefix)
		d, err := b.putTensors([]safetensors.Entry{{Name: name, DType: t.DType, Shape: t.Shape, Data: f.Data(*t)}}, spec.MediaTypeTensor,
			map[string]string{spec.AnnotationLayer: layer.ID, spec.AnnotationTensor: name})
		if err != nil {
			return fail("%s: %v", name, err)
		}
		b.blobs[layer.ID] = append(b.blobs[layer.ID], d)
	}

	layer.Rank = pc.R
	layer.Alpha = pc.Alpha
	layer.RSLoRA = pc.RSLoRA
	layer.Targets = sortTargets(targets)
	var refs []spec.BlobRef
	var size int64
	for _, d := range b.blobs[layer.ID] {
		refs = append(refs, spec.BlobRefFromDescriptor(d))
		size += d.Size
	}
	layer.Digest = spec.LayerDigest(*layer, refs, nil)
	layer.Chain = spec.ChainID(parent.Chain, layer.Digest)
	note := ""
	if len(saved) > 0 {
		note = fmt.Sprintf(", %d saved modules", len(saved))
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("LoRA rank %d, alpha %s, %d modules (%s)%s, %s",
		pc.R, strconv.FormatFloat(pc.Alpha, 'f', -1, 64), len(names), strings.Join(layer.Targets, ","), note, api.FormatSize(size))})
	return nil
}

func readPEFTConfig(dir string) (*peftConfig, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("no adapter_config.json in %s (want a PEFT LoRA adapter directory)", dir)
		}
		return nil, err
	}
	var pc peftConfig
	if err := json.Unmarshal(raw, &pc); err != nil {
		return nil, fmt.Errorf("adapter_config.json: %w", err)
	}
	switch {
	case !strings.EqualFold(pc.PeftType, "LORA"):
		return nil, fmt.Errorf("adapter_config.json: peft_type %q, want LORA", pc.PeftType)
	case pc.R < 1:
		return nil, fmt.Errorf("adapter_config.json: r must be at least 1, got %d", pc.R)
	case pc.Alpha <= 0:
		return nil, fmt.Errorf("adapter_config.json: lora_alpha must be positive, got %v", pc.Alpha)
	case pc.DoRA:
		return nil, errors.New("DoRA adapters (use_dora) are not supported yet")
	case pc.FanInFanOut:
		return nil, errors.New("fan_in_fan_out adapters are not supported yet")
	case pc.Bias != "" && pc.Bias != "none":
		return nil, fmt.Errorf("adapters that train biases (bias=%q) are not supported yet", pc.Bias)
	case len(pc.RankPattern) > 0 || len(pc.AlphaPattern) > 0:
		return nil, errors.New("per-module rank_pattern and alpha_pattern are not supported yet")
	}
	return &pc, nil
}

// checkModule checks that a LoRA module fits the base: the base has the
// weight it modifies, A is [r, in] and B is [out, r] for a base weight of
// [out, in]. A mismatch means the adapter was trained on another model.
func checkModule(m *loraModule, r int, base *baseModel) error {
	w, ok := base.tensors[m.name+".weight"]
	if !ok {
		return fmt.Errorf("%s: the base has no tensor %s.weight; was this adapter trained on another model?", m.name, m.name)
	}
	if len(w.Shape) != 2 {
		return fmt.Errorf("%s: base weight has shape %v; LoRA needs a matrix", m.name, w.Shape)
	}
	out, in := w.Shape[0], w.Shape[1]
	rank := int64(r)
	if !slices.Equal(m.a.Shape, []int64{rank, in}) || !slices.Equal(m.b.Shape, []int64{out, rank}) {
		return fmt.Errorf("%s: lora_A %v and lora_B %v do not fit a base weight of %v at rank %d; was this adapter trained on another model?",
			m.name, m.a.Shape, m.b.Shape, w.Shape, r)
	}
	return nil
}

// peftPrefix starts every tensor name in a saved PEFT adapter.
const peftPrefix = "base_model.model."

// isSavedModule reports whether t is a modules_to_save copy: a full tensor,
// named base_model.model.<base tensor>, of a module the adapter config
// lists (PEFT matches module names by suffix), with the base tensor's dtype
// and shape.
func isSavedModule(t *safetensors.Tensor, modulesToSave []string, base *baseModel) bool {
	name, ok := strings.CutPrefix(t.Name, peftPrefix)
	if !ok {
		return false
	}
	w, ok := base.tensors[name]
	if !ok || w.DType != t.DType || !slices.Equal(w.Shape, t.Shape) {
		return false
	}
	module := name[:max(strings.LastIndex(name, "."), 0)]
	for _, m := range modulesToSave {
		if module == m || strings.HasSuffix(module, "."+m) {
			return true
		}
	}
	return false
}

func targetName(module string) string {
	last := module[strings.LastIndex(module, ".")+1:]
	if t, ok := canonicalTargets[last]; ok {
		return t
	}
	return last
}

// sortTargets orders canonical names first, in their usual order, then any
// other module names alphabetically.
func sortTargets(set map[string]bool) []string {
	var out, other []string
	for _, t := range targetOrder {
		if set[t] {
			out = append(out, t)
		}
	}
	for t := range set {
		if !slices.Contains(targetOrder, t) {
			other = append(other, t)
		}
	}
	sort.Strings(other)
	return append(out, other...)
}

// outsideMix counts the modules a MIX's targets and blocks do not cover.
func outsideMix(modules []string, mix *weightfile.Mix) int {
	n := 0
	for _, m := range modules {
		if !slices.Contains(mix.Targets, targetName(m)) || !inBlocks(m, mix.Blocks) {
			n++
		}
	}
	return n
}

// inBlocks reports whether a module's block is in a selection such as
// "0-3,8-31". "" selects every block; modules outside any block (lm_head)
// are never selected by a range.
func inBlocks(module, blocks string) bool {
	if blocks == "" {
		return true
	}
	m := blockIndex.FindStringSubmatch(module)
	if m == nil {
		return false
	}
	n, _ := strconv.Atoi(m[1])
	for _, part := range strings.Split(blocks, ",") {
		lo, hi, isRange := strings.Cut(part, "-")
		a, _ := strconv.Atoi(lo)
		z := a
		if isRange {
			z, _ = strconv.Atoi(hi)
		}
		if n >= a && n <= z {
			return true
		}
	}
	return false
}

func blocksOrAll(b string) string {
	if b == "" {
		return "all"
	}
	return b
}
