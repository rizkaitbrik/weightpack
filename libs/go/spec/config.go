package spec

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// LayerKind is the kind of a logical layer in an image.
type LayerKind string

const (
	KindBase        LayerKind = "base"
	KindLoRA        LayerKind = "lora"
	KindDeltaXOR    LayerKind = "delta.xor"
	KindDeltaBit1   LayerKind = "delta.bit1"
	KindDeltaSparse LayerKind = "delta.sparse"
	KindFFNExpert   LayerKind = "ffn-expert"
	KindMix         LayerKind = "mix"
	KindRouter      LayerKind = "router"
	KindMeta        LayerKind = "meta"
)

// Config is the image config blob (MediaTypeConfig).
type Config struct {
	Schema    string  `json:"schema"`
	Arch      Arch    `json:"arch"`
	Tokenizer string  `json:"tokenizer"`
	Layers    []Layer `json:"layers"`
}

// Arch describes the base architecture every layer must match.
type Arch struct {
	Family      string `json:"family"`
	Fingerprint string `json:"fingerprint"`
	Blocks      int    `json:"blocks,omitempty"`
	Hidden      int    `json:"hidden,omitempty"`
}

// Layer is one logical layer of the composition graph.
type Layer struct {
	ID       string    `json:"id"`
	Kind     LayerKind `json:"kind"`
	Parent   string    `json:"parent,omitempty"`
	Chain    string    `json:"chain,omitempty"`
	Rank     int       `json:"rank,omitempty"`
	Targets  []string  `json:"targets,omitempty"`
	Blocks   string    `json:"blocks,omitempty"`
	TopK     int       `json:"top_k,omitempty"`
	Router   string    `json:"router,omitempty"`
	Experts  []string  `json:"experts,omitempty"`
	Fidelity *Fidelity `json:"fidelity,omitempty"`
}

// Fidelity records how a lossy layer was produced and what it cost.
type Fidelity struct {
	Source       string   `json:"source"`
	Codec        string   `json:"codec,omitempty"`
	Eval         string   `json:"eval,omitempty"`
	MaxDrop      float64  `json:"max_drop,omitempty"`
	MeasuredDrop *float64 `json:"measured_drop,omitempty"`
}

// SchemaV1 is the current config schema identifier.
const SchemaV1 = "weightpack.config.v1"

// ChainID returns sha256(parentChainID + " " + layerDigest), as Docker
// computes chain IDs. The base layer has no parent and its chain ID is its
// own digest.
func ChainID(parentChainID, layerDigest string) string {
	if parentChainID == "" {
		return layerDigest
	}
	sum := sha256.Sum256([]byte(parentChainID + " " + layerDigest))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// TensorInfo is one entry of a base model's tensor table.
type TensorInfo struct {
	Name  string
	DType string
	Shape []int64
}

// ArchFingerprint hashes the sorted (name, dtype, shape) list of a base model
// together with its tokenizer digest. Two models with the same fingerprint
// accept the same layers.
func ArchFingerprint(tensors []TensorInfo, tokenizerDigest string) string {
	lines := make([]string, 0, len(tensors)+1)
	for _, t := range tensors {
		dims := make([]string, len(t.Shape))
		for i, d := range t.Shape {
			dims[i] = fmt.Sprint(d)
		}
		lines = append(lines, t.Name+"\t"+t.DType+"\t"+strings.Join(dims, "x"))
	}
	sort.Strings(lines)
	lines = append(lines, "tokenizer\t"+tokenizerDigest)
	sum := sha256.Sum256([]byte(strings.Join(lines, "\n")))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// Validate checks the composition rules of the design:
// unique ids, known parents, exactly one base, and every expert of a mix
// sharing the mix's parent.
func (c *Config) Validate() error {
	if c.Schema != SchemaV1 {
		return fmt.Errorf("unsupported schema %q", c.Schema)
	}
	byID := make(map[string]Layer, len(c.Layers))
	bases := 0
	for _, l := range c.Layers {
		if l.ID == "" {
			return errors.New("layer with empty id")
		}
		if _, dup := byID[l.ID]; dup {
			return fmt.Errorf("duplicate layer id %q", l.ID)
		}
		byID[l.ID] = l
		if l.Kind == KindBase {
			bases++
		}
	}
	if bases != 1 {
		return fmt.Errorf("image must have exactly one base layer, found %d", bases)
	}
	for _, l := range c.Layers {
		if l.Kind == KindBase {
			if l.Parent != "" {
				return fmt.Errorf("base layer %q cannot have a parent", l.ID)
			}
			continue
		}
		if _, ok := byID[l.Parent]; !ok {
			return fmt.Errorf("layer %q: unknown parent %q", l.ID, l.Parent)
		}
		if l.Kind != KindMix {
			continue
		}
		if len(l.Experts) == 0 {
			return fmt.Errorf("mix %q has no experts", l.ID)
		}
		for _, e := range l.Experts {
			expert, ok := byID[e]
			if !ok {
				return fmt.Errorf("mix %q: unknown expert %q", l.ID, e)
			}
			if expert.Parent != l.Parent {
				return fmt.Errorf("mix %q: expert %q has parent %q, want %q", l.ID, e, expert.Parent, l.Parent)
			}
		}
		if l.TopK < 1 || l.TopK > len(l.Experts) {
			return fmt.Errorf("mix %q: top_k %d out of range 1..%d", l.ID, l.TopK, len(l.Experts))
		}
	}
	return nil
}
