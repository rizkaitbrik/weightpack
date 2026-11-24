// Package buildtest writes small synthetic models and PEFT LoRA adapters for
// tests: a Llama-shaped base with two blocks, and adapters over it.
package buildtest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
)

// Shape of the synthetic base.
const (
	Hidden = 128
	Inter  = 256
	Vocab  = 64
	Blocks = 2
)

// Module sets for adapters.
var (
	Attn = []string{"self_attn.q_proj", "self_attn.v_proj"}
	MLP  = []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
)

// Tensor is an F32 tensor filled with bytes from a seeded generator.
type Tensor struct {
	Name  string
	Shape []int64
	Seed  int64
}

func (t Tensor) entry() safetensors.Entry {
	n, _ := safetensors.ByteSize("F32", t.Shape)
	data := make([]byte, n)
	rand.New(rand.NewSource(t.Seed)).Read(data)
	return safetensors.Entry{Name: t.Name, DType: "F32", Shape: t.Shape, Data: bytes.NewReader(data)}
}

// WriteSafetensors writes tensors to path.
func WriteSafetensors(t testing.TB, path string, ts []Tensor) {
	t.Helper()
	var es []safetensors.Entry
	for _, x := range ts {
		es = append(es, x.entry())
	}
	r, _, err := safetensors.NewReader(es, map[string]string{"format": "pt"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// WriteJSON writes v as JSON to path.
func WriteJSON(t testing.TB, path string, v any) {
	t.Helper()
	b, _ := json.Marshal(v)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// BaseTensors returns the base model's tensor table.
func BaseTensors() []Tensor {
	ts := []Tensor{
		{"model.embed_tokens.weight", []int64{Vocab, Hidden}, 1},
		{"model.norm.weight", []int64{Hidden}, 2},
		{"lm_head.weight", []int64{Vocab, Hidden}, 3},
	}
	for i := range Blocks {
		p := fmt.Sprintf("model.layers.%d.", i)
		s := int64(100 * (i + 1))
		ts = append(ts,
			Tensor{p + "input_layernorm.weight", []int64{Hidden}, s},
			Tensor{p + "post_attention_layernorm.weight", []int64{Hidden}, s + 1},
			Tensor{p + "self_attn.q_proj.weight", []int64{Hidden, Hidden}, s + 2},
			Tensor{p + "self_attn.k_proj.weight", []int64{Hidden, Hidden}, s + 3},
			Tensor{p + "self_attn.v_proj.weight", []int64{Hidden, Hidden}, s + 4},
			Tensor{p + "self_attn.o_proj.weight", []int64{Hidden, Hidden}, s + 5},
			Tensor{p + "mlp.gate_proj.weight", []int64{Inter, Hidden}, s + 6},
			Tensor{p + "mlp.up_proj.weight", []int64{Inter, Hidden}, s + 7},
			Tensor{p + "mlp.down_proj.weight", []int64{Hidden, Inter}, s + 8},
		)
	}
	return ts
}

// WriteBase writes the base model to dir: weights, config.json and
// tokenizer.json. With shards > 1 the tensors are split over several files
// with an index, as large models are.
func WriteBase(t testing.TB, dir string, shards int) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	ts := BaseTensors()
	if shards <= 1 {
		WriteSafetensors(t, filepath.Join(dir, "model.safetensors"), ts)
	} else {
		weightMap := map[string]string{}
		for s := range shards {
			name := fmt.Sprintf("model-%05d-of-%05d.safetensors", s+1, shards)
			var part []Tensor
			for i, x := range ts {
				if i%shards == s {
					part = append(part, x)
					weightMap[x.Name] = name
				}
			}
			WriteSafetensors(t, filepath.Join(dir, name), part)
		}
		WriteJSON(t, filepath.Join(dir, "model.safetensors.index.json"), map[string]any{"weight_map": weightMap})
	}
	WriteJSON(t, filepath.Join(dir, "config.json"), map[string]any{
		"model_type": "llama", "num_hidden_layers": Blocks, "hidden_size": Hidden,
	})
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Adapter describes a PEFT LoRA adapter over the base.
type Adapter struct {
	Rank    int
	Alpha   float64
	Modules []string // such as self_attn.q_proj or mlp.up_proj
	Blocks  []int
	Seed    int64
	Config  map[string]any // overrides for adapter_config.json
}

// WriteAdapter writes adapter_config.json and adapter_model.safetensors, with
// tensor names as PEFT saves them.
func WriteAdapter(t testing.TB, dir string, a Adapter) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	shapes := map[string][2]int64{ // out, in
		"self_attn.q_proj": {Hidden, Hidden}, "self_attn.k_proj": {Hidden, Hidden},
		"self_attn.v_proj": {Hidden, Hidden}, "self_attn.o_proj": {Hidden, Hidden},
		"mlp.gate_proj": {Inter, Hidden}, "mlp.up_proj": {Inter, Hidden}, "mlp.down_proj": {Hidden, Inter},
	}
	var ts []Tensor
	r := int64(a.Rank)
	for _, blk := range a.Blocks {
		for i, m := range a.Modules {
			s := shapes[m]
			p := fmt.Sprintf("base_model.model.model.layers.%d.%s.", blk, m)
			seed := a.Seed*1000 + int64(blk*10+i)
			ts = append(ts,
				Tensor{p + "lora_A.weight", []int64{r, s[1]}, seed},
				Tensor{p + "lora_B.weight", []int64{s[0], r}, seed + 500},
			)
		}
	}
	WriteSafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), ts)
	cfg := map[string]any{"peft_type": "LORA", "r": a.Rank, "lora_alpha": a.Alpha, "bias": "none",
		"target_modules": a.Modules, "use_rslora": false, "use_dora": false, "modules_to_save": nil}
	maps.Copy(cfg, a.Config)
	WriteJSON(t, filepath.Join(dir, "adapter_config.json"), cfg)
}

// Assistant is a Weightfile over the files Project writes: a stacked LoRA
// and a mix of two LoRA experts.
const Assistant = `FROM ./base
LAYER french ./adapters/french
MIX domain targets=mlp blocks=1 top_k=1
  EXPERT legal   ./adapters/legal
  EXPERT medical ./adapters/medical
  ROUTER arrow
TEMPLATE ./chat.jinja
LICENSE Apache-2.0
`

// Project writes a build context to dir: base/, adapters/{french,legal,
// medical}/, chat.jinja, and Assistant as the Weightfile. It returns the
// Weightfile's path. medical covers block 0, which the mix leaves out.
func Project(t testing.TB, dir string) string {
	t.Helper()
	WriteBase(t, filepath.Join(dir, "base"), 1)
	WriteAdapter(t, filepath.Join(dir, "adapters/french"), Adapter{Rank: 8, Alpha: 16, Modules: Attn, Blocks: []int{0, 1}, Seed: 1})
	WriteAdapter(t, filepath.Join(dir, "adapters/legal"), Adapter{Rank: 4, Alpha: 8, Modules: MLP, Blocks: []int{1}, Seed: 2})
	WriteAdapter(t, filepath.Join(dir, "adapters/medical"), Adapter{Rank: 4, Alpha: 8, Modules: MLP, Blocks: []int{0, 1}, Seed: 3})
	if err := os.WriteFile(filepath.Join(dir, "chat.jinja"), []byte("{{ messages }}"), 0o644); err != nil {
		t.Fatal(err)
	}
	wf := filepath.Join(dir, "Weightfile")
	if err := os.WriteFile(wf, []byte(Assistant), 0o644); err != nil {
		t.Fatal(err)
	}
	return wf
}
