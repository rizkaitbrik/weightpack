package build

import (
	"errors"
	"fmt"
	"maps"
	"strings"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
)

// inherited is what a FROM image passes on to the image being built.
type inherited struct {
	ref    string
	layers []spec.Layer // its base, then its stack, bottom first
	base   *baseModel
	// template and licenseFile are its image-level meta blobs, if any;
	// license is its SPDX licenses annotation. The Weightfile's own
	// TEMPLATE and LICENSE replace them.
	template, licenseFile *ocispec.Descriptor
	license               string
}

// image finds ref in the store, pulling it through b.pull when it is not
// there, and returns its checked manifest and config.
func (b *builder) image(step string, line int, ref string) (*ocispec.Manifest, *spec.Config, error) {
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	d, err := b.store.Resolve(ref)
	if errors.Is(err, store.ErrUnknownImage) {
		if b.pull == nil {
			return nil, nil, fail("%s is not in the local store; pull it first with wpk pull", ref)
		}
		b.progress(api.Event{Step: step, Message: "pulling " + ref})
		if err := b.pull(b.ctx, ref, b.progress); err != nil {
			return nil, nil, fail("%v", err)
		}
		d, err = b.store.Resolve(ref)
	}
	if err != nil {
		return nil, nil, fail("%v", err)
	}
	var m ocispec.Manifest
	var c spec.Config
	if err := b.store.ReadJSON(d.Digest.String(), &m); err != nil {
		return nil, nil, fail("%v", err)
	}
	if err := b.store.ReadJSON(m.Config.Digest.String(), &c); err != nil {
		return nil, nil, fail("%v", err)
	}
	if err := spec.CheckImage(&m, &c); err != nil {
		return nil, nil, fail("%s is inconsistent: %v", ref, err)
	}
	return &m, &c, nil
}

// stackOf returns an image's base and stack, bottom first, and its MIX if
// it has one. It refuses every other shape rather than guess: two layers on
// one parent outside a mix, a mix that is not on top of the stack, or a
// layer above a mix or an expert. Export and squash read images the same
// way.
func stackOf(c *spec.Config) ([]spec.Layer, *spec.Layer, error) {
	children := map[string][]spec.Layer{}
	var stack []spec.Layer
	var mix *spec.Layer
	for i, l := range c.Layers {
		switch l.Kind {
		case spec.KindBase:
			stack = []spec.Layer{l}
		case spec.KindMix:
			mix = &c.Layers[i]
		}
		if l.Parent != "" {
			children[l.Parent] = append(children[l.Parent], l)
		}
	}
	if len(stack) == 0 {
		return nil, nil, errors.New("the image has no base layer")
	}
	experts := map[string]bool{}
	if mix != nil {
		for _, e := range mix.Experts {
			experts[e] = true
		}
	}
	for {
		var next []spec.Layer
		for _, ch := range children[stack[len(stack)-1].ID] {
			if ch.Kind != spec.KindMix && !experts[ch.ID] {
				next = append(next, ch)
			}
		}
		if len(next) > 1 {
			var ids []string
			for _, l := range next {
				ids = append(ids, l.ID)
			}
			return nil, nil, fmt.Errorf("layers %s both sit on %s; an image must be one stack, with at most one MIX on top",
				strings.Join(ids, " and "), stack[len(stack)-1].ID)
		}
		if len(next) == 0 {
			break
		}
		stack = append(stack, next[0])
	}
	placed := map[string]bool{}
	for _, l := range stack {
		placed[l.ID] = true
	}
	if mix != nil {
		if top := stack[len(stack)-1].ID; mix.Parent != top {
			return nil, nil, fmt.Errorf("MIX %s sits on %s, not on the top of the stack (%s)", mix.ID, mix.Parent, top)
		}
		placed[mix.ID] = true
		for e := range experts {
			placed[e] = true
		}
	}
	var stray []string
	for _, l := range c.Layers {
		if !placed[l.ID] {
			stray = append(stray, l.ID)
		}
	}
	if len(stray) > 0 {
		return nil, nil, fmt.Errorf("layers %s sit above a MIX or an expert, which an image cannot express", strings.Join(stray, ", "))
	}
	return stack, mix, nil
}

// Stack returns an image's base and stack, bottom first, and its MIX if it
// has one, refusing every other shape. The runtime reads images with it.
func Stack(c *spec.Config) ([]spec.Layer, *spec.Layer, error) { return stackOf(c) }

// chainPath names the layers a chain runs through: "base → french".
func chainPath(byID map[string]spec.Layer, id string) string {
	var ids []string
	for id != "" {
		ids = append([]string{id}, ids...)
		id = byID[id].Parent
	}
	return strings.Join(ids, " → ")
}

// tensorTable reads the tensor names, dtypes and shapes of a base layer's
// blobs from their headers.
func (b *builder) tensorTable(ds []ocispec.Descriptor) (map[string]safetensors.Tensor, error) {
	out := map[string]safetensors.Tensor{}
	for _, d := range ds {
		if spec.BaseMediaType(d.MediaType) != spec.MediaTypeTensor {
			continue
		}
		p, err := b.store.Path(d.Digest.String())
		if err != nil {
			return nil, err
		}
		f, err := safetensors.Open(p)
		if err != nil {
			return nil, err
		}
		for _, t := range f.Tensors {
			out[t.Name] = t
		}
		_ = f.Close()
	}
	return out, nil
}

// fromImage resolves the FROM image and copies its base and stack into the
// build, keeping their ids, digests and chain IDs.
func (b *builder) fromImage(from weightfile.From) (*inherited, error) {
	ref := from.Source.Ref
	step := "FROM " + from.Source.String()
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: from.Line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	m, c, err := b.image(step, from.Line, ref)
	if err != nil {
		return nil, err
	}
	stack, mix, err := stackOf(c)
	if err != nil {
		return nil, fail("%s: %v", ref, err)
	}
	if mix != nil {
		return nil, fail("%s ends in MIX %s, and nothing can be stacked above a mix", ref, mix.ID)
	}
	inh := &inherited{ref: ref, layers: stack, license: m.Annotations[ocispec.AnnotationLicenses]}
	keep := map[string]bool{}
	for _, l := range stack {
		keep[l.ID] = true
	}
	for _, d := range m.Layers {
		id, ok := d.Annotations[spec.AnnotationLayer]
		switch {
		case ok && keep[id]:
			b.blobs[id] = append(b.blobs[id], d)
		case !ok && d.Annotations[spec.AnnotationTitle] == titleTemplate:
			inh.template = &d
		case !ok && d.Annotations[spec.AnnotationTitle] == titleLicense:
			inh.licenseFile = &d
		}
	}
	tensors, err := b.tensorTable(b.blobs[stack[0].ID])
	if err != nil {
		return nil, fail("%v", err)
	}
	inh.base = &baseModel{tensors: tensors, arch: c.Arch, tokenizer: c.Tokenizer}
	var ids []string
	for _, l := range stack {
		ids = append(ids, l.ID)
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("%s base, layers %s, chain %s", familyOr(c.Arch.Family), strings.Join(ids, " → "), short(stack[len(stack)-1].Chain))})
	b.inherited = inh
	return inh, nil
}

// graft puts the inherited layers in place of the Weightfile's base entry
// and rests the Weightfile's own layers on the inherited top.
func (b *builder) graft(wf *weightfile.File, cfg *spec.Config, inh *inherited) error {
	lines := map[string]int{}
	for _, l := range wf.Layers {
		lines[l.ID] = l.Line
	}
	if m := wf.Mix; m != nil {
		lines[m.ID] = m.Line
		for _, e := range m.Experts {
			lines[e.ID] = e.Line
		}
	}
	taken := map[string]bool{}
	for _, l := range inh.layers {
		taken[l.ID] = true
	}
	var errs []error
	for _, l := range cfg.Layers[1:] {
		if taken[l.ID] {
			errs = append(errs, &weightfile.Error{File: b.file, Line: lines[l.ID],
				Msg: fmt.Sprintf("%s: the FROM image already has a layer %s; give this one another id", l.ID, l.ID)})
		}
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	top := inh.layers[len(inh.layers)-1].ID
	own := cfg.Layers[1:]
	for i := range own {
		if own[i].Parent == weightfile.ReservedBaseID {
			own[i].Parent = top
		}
	}
	cfg.Layers = append(append([]spec.Layer{}, inh.layers...), own...)
	return nil
}

// ingestLayer dispatches a LAYER or EXPERT on its source.
func (b *builder) ingestLayer(step string, src weightfile.Layer, layer, parent *spec.Layer, base *baseModel, mix *weightfile.Mix, byID map[string]*spec.Layer) error {
	switch {
	case src.Source.Kind == weightfile.SourceOCI:
		return b.ingestImageLayer(step, src, layer, parent, mix, byID)
	case src.Codec == weightfile.CodecXOR:
		return b.ingestXOR(step, src, layer, parent, base, byID)
	case src.Codec == weightfile.CodecBit1 || src.Codec == weightfile.CodecSparse:
		return b.ingestLossy(step, src, layer, parent, base, byID)
	}
	return b.ingestLoRA(step, src, layer, parent, base, mix)
}

// ingestImageLayer takes the top layer of the image src names. The layer
// must have been built on exactly the chain it would now sit on: a layer
// only works on the chain it was trained on.
func (b *builder) ingestImageLayer(step string, src weightfile.Layer, layer, parent *spec.Layer, mix *weightfile.Mix, byID map[string]*spec.Layer) error {
	ref := src.Source.Ref
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: src.Line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	m, c, err := b.image(step, src.Line, ref)
	if err != nil {
		return err
	}
	stack, srcMix, err := stackOf(c)
	if err != nil {
		return fail("%s: %v", ref, err)
	}
	switch {
	case srcMix != nil:
		return fail("%s ends in MIX %s; name an image whose top layer is the one to use", ref, srcMix.ID)
	case len(stack) < 2:
		return fail("%s is only a base and has no layer to use", ref)
	}
	top, srcParent := stack[len(stack)-1], stack[len(stack)-2]
	if srcParent.Chain != parent.Chain {
		srcByID := map[string]spec.Layer{}
		for _, l := range c.Layers {
			srcByID[l.ID] = l
		}
		here := map[string]spec.Layer{}
		for id, l := range byID {
			here[id] = *l
		}
		return fail("layer %s of %s was built on %s (chain %s), but here it would sit on %s (chain %s); a layer only works on the chain it was built on",
			top.ID, ref, chainPath(srcByID, srcParent.ID), short(srcParent.Chain), chainPath(here, parent.ID), short(parent.Chain))
	}

	var refs []spec.BlobRef
	var modules []string
	var size int64
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] != top.ID {
			continue
		}
		nd := d
		nd.Annotations = maps.Clone(d.Annotations)
		nd.Annotations[spec.AnnotationLayer] = layer.ID
		b.blobs[layer.ID] = append(b.blobs[layer.ID], nd)
		refs = append(refs, spec.BlobRefFromDescriptor(nd))
		if t, ok := strings.CutSuffix(nd.Annotations[spec.AnnotationTensor], ".weight"); ok {
			modules = append(modules, t)
		}
		size += nd.Size
	}
	layer.Kind, layer.Rank, layer.Alpha, layer.RSLoRA = top.Kind, top.Rank, top.Alpha, top.RSLoRA
	layer.Targets, layer.Fidelity = top.Targets, top.Fidelity
	layer.Digest = spec.LayerDigest(*layer, refs, nil)
	if layer.Digest != top.Digest {
		return fail("layer %s of %s does not match its digest (this is a bug)", top.ID, ref)
	}
	layer.Chain = spec.ChainID(parent.Chain, layer.Digest)
	if mix != nil {
		if outside := outsideMix(modules, mix); outside > 0 {
			b.progress(api.Event{Step: step, Warning: true, Message: fmt.Sprintf(
				"%d of %d modules lie outside MIX %s (targets=%s blocks=%s)",
				outside, len(modules), mix.ID, strings.Join(mix.Targets, ","), blocksOrAll(mix.Blocks))})
		}
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("%s rank %d from %s (layer %s), %d modules, %s",
		top.Kind, top.Rank, ref, top.ID, len(modules), api.FormatSize(size))})
	return nil
}
