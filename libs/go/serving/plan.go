package serving

import (
	"fmt"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

// plan is how an image is served: a dense serving base, and the LoRA layers
// an engine applies on top of it at request time.
//
// Engines apply one adapter per request, so everything below the image's
// adapters is baked into the base: its XOR layers, and every stacked LoRA
// layer but the top one. A stack ending in a LoRA layer serves that layer
// as the adapter; a MIX serves each expert as an adapter over the whole
// stack. Images that share those lower layers share the serving base, and
// with it one engine and one copy of the weights.
type plan struct {
	// base is the top layer of the serving base. Its chain ID names the
	// base; every image whose layers below their adapters are the same
	// has the same one.
	base spec.Layer
	// through is the layer to squash the image through to make the base,
	// or "" when the base is the image's base layer as it is.
	through string
	// adapters are served on top of the base: none, the top of a stack, or
	// a mix's experts.
	adapters []spec.Layer
	// mix is the image's MIX, if any; its experts are the adapters.
	mix *spec.Layer
}

func planFor(c *spec.Config) (*plan, error) {
	stack, mix, err := build.Stack(c)
	if err != nil {
		return nil, err
	}
	p := &plan{mix: mix}
	dense := stack
	if mix != nil {
		byID := map[string]spec.Layer{}
		for _, l := range c.Layers {
			byID[l.ID] = l
		}
		for _, id := range mix.Experts {
			p.adapters = append(p.adapters, byID[id])
		}
	} else if top := stack[len(stack)-1]; len(stack) > 1 && top.Kind == spec.KindLoRA {
		p.adapters = []spec.Layer{top}
		dense = stack[:len(stack)-1]
	}
	for _, a := range p.adapters {
		if a.Kind != spec.KindLoRA {
			return nil, fmt.Errorf("layer %q is %s; engines serve only LoRA layers on top of a base, so wpk squash the image and run the result", a.ID, a.Kind)
		}
	}
	p.base = dense[len(dense)-1]
	if len(dense) > 1 {
		p.through = p.base.ID
	}
	if p.base.Chain == "" {
		return nil, fmt.Errorf("layer %q has no chain ID; rebuild the image", p.base.ID)
	}
	return p, nil
}

// describe says what the plan does, for progress messages.
func (p *plan) describe() string {
	var b strings.Builder
	if p.through != "" {
		fmt.Fprintf(&b, "base %s baked through %s", short(p.base.Chain), p.through)
	} else {
		fmt.Fprintf(&b, "base %s", short(p.base.Chain))
	}
	switch {
	case p.mix != nil:
		fmt.Fprintf(&b, ", experts %s as adapters", strings.Join(p.mix.Experts, ", "))
	case len(p.adapters) == 1:
		fmt.Fprintf(&b, ", %s as an adapter", p.adapters[0].ID)
	}
	return b.String()
}

// short is the first 12 hex digits of a digest, as Docker prints image ids.
func short(d string) string {
	_, hex, ok := strings.Cut(d, ":")
	if !ok {
		hex = d
	}
	return hex[:min(12, len(hex))]
}
