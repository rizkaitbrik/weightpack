package weightfile

import (
	"fmt"

	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

// kindForCodec maps a codec= value to the layer kind it produces.
func kindForCodec(codec string) spec.LayerKind {
	switch codec {
	case CodecXOR:
		return spec.KindDeltaXOR
	case CodecBit1:
		return spec.KindDeltaBit1
	case CodecSparse:
		return spec.KindDeltaSparse
	default:
		return spec.KindLoRA
	}
}

// Config returns the layer graph the Weightfile declares, as an image config.
// The architecture, tokenizer, ranks and targets of adapters are only known
// once the sources are read, so the build fills them in; chain IDs are set
// when the layers have digests.
func (f *File) Config() (*spec.Config, error) {
	cfg := &spec.Config{
		Schema: spec.SchemaV1,
		Layers: []spec.Layer{{ID: ReservedBaseID, Kind: spec.KindBase}},
	}
	parent := ReservedBaseID
	for _, l := range f.Layers {
		cfg.Layers = append(cfg.Layers, weightLayer(l, parent))
		parent = l.ID
	}
	if m := f.Mix; m != nil {
		mix := spec.Layer{
			ID:      m.ID,
			Kind:    spec.KindMix,
			Parent:  parent,
			Targets: m.Targets,
			Blocks:  m.Blocks,
			TopK:    m.TopK,
			Router:  m.Router.Kind,
		}
		for _, e := range m.Experts {
			mix.Experts = append(mix.Experts, e.ID)
		}
		cfg.Layers = append(cfg.Layers, mix)
		for _, e := range m.Experts {
			cfg.Layers = append(cfg.Layers, weightLayer(e, parent))
		}
	}
	if err := cfg.Validate(); err != nil {
		// The parser enforces the same rules, so this is a bug in one of them.
		return nil, fmt.Errorf("weightfile produced an invalid config: %w", err)
	}
	return cfg, nil
}

func weightLayer(l Layer, parent string) spec.Layer {
	out := spec.Layer{ID: l.ID, Kind: kindForCodec(l.Codec), Parent: parent}
	if l.Codec != "" && l.Codec != CodecLoRA {
		fid := &spec.Fidelity{Source: l.Source.String(), Codec: l.Codec, Eval: l.Eval}
		if l.MaxDrop != nil {
			fid.MaxDrop = *l.MaxDrop
		}
		out.Fidelity = fid
	}
	return out
}
