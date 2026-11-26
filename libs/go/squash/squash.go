// Package squash bakes an image's XOR and LoRA layers into its base tensors and
// stores the result as a new image with a single base layer: a dense model
// any engine can load once exported.
//
// The math runs in a build worker (see package worker). This package picks
// the layers to bake, hands the worker the blob paths, checks every tensor
// the worker returns against the base, and reassembles the base layer.
// Blobs whose tensors no layer touches are reused as they are, so a squash
// stores only the tensors it changed.
package squash

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// Options say what to squash and how to name the result.
type Options struct {
	// Expert picks the one expert to bake in when the image has a MIX. A
	// mix that routes per token cannot be squashed as a whole.
	Expert string
	// Tags to point at the squashed image.
	Tags []string
}

// Squash bakes the image ref names into a new dense image in the store.
func Squash(ctx context.Context, s *store.Store, w workerpb.WorkerClient, ref string, opts Options, progress func(api.Event)) (*api.BuildResult, error) {
	if progress == nil {
		progress = func(api.Event) {}
	}
	for _, t := range opts.Tags {
		if _, err := reference.ParseNamed(t); err != nil {
			return nil, err
		}
	}
	root, err := s.Resolve(ref)
	if err != nil {
		return nil, err
	}
	var m ocispec.Manifest
	var cfg spec.Config
	if err := s.ReadJSON(root.Digest.String(), &m); err != nil {
		return nil, err
	}
	if err := s.ReadJSON(m.Config.Digest.String(), &cfg); err != nil {
		return nil, err
	}
	if err := spec.CheckImage(&m, &cfg); err != nil {
		return nil, fmt.Errorf("%s is inconsistent, not squashing it: %w", ref, err)
	}
	chain, err := pickChain(&cfg, opts.Expert)
	if err != nil {
		return nil, fmt.Errorf("squash %s: %w", ref, err)
	}
	if len(chain) == 0 {
		return nil, fmt.Errorf("squash %s: the image is only a base; there is nothing to bake in", ref)
	}

	byLayer := map[string][]ocispec.Descriptor{}
	for _, d := range m.Layers {
		id := d.Annotations[spec.AnnotationLayer]
		byLayer[id] = append(byLayer[id], d)
	}
	var base spec.Layer
	for _, l := range cfg.Layers {
		if l.Kind == spec.KindBase {
			base = l
		}
	}

	req := &workerpb.SquashRequest{}
	baseTensors := map[string]safetensors.Tensor{}
	for _, d := range byLayer[base.ID] {
		if spec.BaseMediaType(d.MediaType) != spec.MediaTypeTensor {
			continue
		}
		p, err := s.Path(d.Digest.String())
		if err != nil {
			return nil, err
		}
		if err := readTensors(p, baseTensors); err != nil {
			return nil, err
		}
		req.Base = append(req.Base, &workerpb.Blob{Digest: d.Digest.String(), MediaType: d.MediaType, Path: p})
	}
	inDir, err := s.TempDir("squash-in-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(inDir)
	var ids []string
	sawLoRA := false
	for _, l := range chain {
		switch l.Kind {
		case spec.KindDeltaXOR:
			// XOR layers apply to the base's bits first, exactly; one above a
			// LoRA layer could not have been built.
			if sawLoRA {
				return nil, fmt.Errorf("squash %s: XOR layer %q sits above a LoRA layer", ref, l.ID)
			}
			x, err := build.DecodeXORLayer(s, l.ID, byLayer[l.ID], inDir)
			if err != nil {
				return nil, err
			}
			req.XorLayers = append(req.XorLayers, x)
			ids = append(ids, l.ID)
			continue
		case spec.KindLoRA:
			sawLoRA = true
		default:
			return nil, fmt.Errorf("squash %s: layer %q is %s, which squash cannot bake in yet", ref, l.ID, l.Kind)
		}
		scaling, err := Scaling(l)
		if err != nil {
			return nil, err
		}
		lw := &workerpb.LoRALayer{Id: l.ID, Scaling: scaling}
		for _, d := range byLayer[l.ID] {
			p, err := s.Path(d.Digest.String())
			if err != nil {
				return nil, err
			}
			lw.Blobs = append(lw.Blobs, &workerpb.Blob{Digest: d.Digest.String(), MediaType: d.MediaType, Path: p})
		}
		req.Layers = append(req.Layers, lw)
		ids = append(ids, l.ID)
	}

	outDir, err := s.TempDir("squash-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(outDir)
	req.OutDir = outDir

	step := "squash " + ref
	progress(api.Event{Step: step, Message: fmt.Sprintf("baking %s into the base", strings.Join(ids, ", "))})
	resp, err := w.Squash(ctx, req)
	if err != nil {
		return nil, err
	}
	merged, err := checkMerged(outDir, resp, baseTensors)
	if err != nil {
		return nil, err
	}
	progress(api.Event{Step: step, Message: fmt.Sprintf("worker merged %d tensors", len(merged))})

	wr := &writer{s: s}
	newBase, err := wr.relayout(byLayer[base.ID], merged)
	if err != nil {
		return nil, err
	}
	out := &spec.Config{
		Schema: spec.SchemaV1, Arch: cfg.Arch, Tokenizer: cfg.Tokenizer,
		Layers: []spec.Layer{{ID: base.ID, Kind: spec.KindBase}},
	}
	var refs []spec.BlobRef
	for _, d := range newBase {
		refs = append(refs, spec.BlobRefFromDescriptor(d))
	}
	out.Layers[0].Digest = spec.LayerDigest(out.Layers[0], refs, nil)
	out.Layers[0].Chain = spec.ChainID("", out.Layers[0].Digest)

	annotations := maps.Clone(m.Annotations)
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[spec.AnnotationSquashed] = strings.Join(ids, ",")
	annotations[spec.AnnotationSquashedFrom] = root.Digest.String()
	return wr.finish(out, append(newBase, byLayer[""]...), annotations, opts.Tags)
}

// Scaling returns a LoRA layer's scale as PEFT defines it: alpha / rank, or
// alpha / sqrt(rank) for rsLoRA.
func Scaling(l spec.Layer) (float64, error) {
	if l.Rank < 1 || l.Alpha <= 0 {
		return 0, fmt.Errorf("layer %q records no rank or alpha; rebuild the image", l.ID)
	}
	if l.RSLoRA {
		return l.Alpha / math.Sqrt(float64(l.Rank)), nil
	}
	return l.Alpha / float64(l.Rank), nil
}

// pickChain returns the layers to bake, bottom first: the stack, or with a
// MIX the stack and the chosen expert.
func pickChain(c *spec.Config, expert string) ([]spec.Layer, error) {
	byID := map[string]spec.Layer{}
	parents := map[string]bool{}
	var mix *spec.Layer
	for i, l := range c.Layers {
		byID[l.ID] = l
		parents[l.Parent] = true
		if l.Kind == spec.KindMix {
			mix = &c.Layers[i]
		}
	}
	var top string
	switch {
	case mix != nil && expert == "":
		return nil, fmt.Errorf("MIX %s routes between %s; pick one with --expert (a mix that routes per token cannot be squashed as a whole)",
			mix.ID, strings.Join(mix.Experts, ", "))
	case mix != nil:
		if !slices.Contains(mix.Experts, expert) {
			return nil, fmt.Errorf("MIX %s has no expert %q (experts: %s)", mix.ID, expert, strings.Join(mix.Experts, ", "))
		}
		top = expert
	case expert != "":
		return nil, fmt.Errorf("--expert %s given, but the image has no MIX", expert)
	default:
		var tops []string
		for _, l := range c.Layers {
			if l.Kind != spec.KindBase && !parents[l.ID] {
				tops = append(tops, l.ID)
			}
		}
		switch len(tops) {
		case 0:
			return nil, nil
		case 1:
			top = tops[0]
		default:
			return nil, fmt.Errorf("the image has several top layers (%s)", strings.Join(tops, ", "))
		}
	}
	var chain []spec.Layer
	for id := top; byID[id].Kind != spec.KindBase; id = byID[id].Parent {
		chain = append([]spec.Layer{byID[id]}, chain...)
	}
	return chain, nil
}

func readTensors(path string, into map[string]safetensors.Tensor) error {
	f, err := safetensors.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, t := range f.Tensors {
		into[t.Name] = t
	}
	return nil
}

// checkMerged checks the worker's output: each file is a plain name inside
// outDir holding exactly the tensor it claims, with the base tensor's dtype
// and shape. It returns tensor name -> file path.
func checkMerged(outDir string, resp *workerpb.SquashResponse, base map[string]safetensors.Tensor) (map[string]string, error) {
	merged := map[string]string{}
	for _, t := range resp.Tensors {
		if t.File == "" || t.File != filepath.Base(t.File) || strings.HasPrefix(t.File, ".") {
			return nil, fmt.Errorf("worker returned file %q outside its output directory", t.File)
		}
		want, ok := base[t.Name]
		if !ok {
			return nil, fmt.Errorf("worker returned tensor %q, which the base does not have", t.Name)
		}
		if _, dup := merged[t.Name]; dup {
			return nil, fmt.Errorf("worker returned tensor %q twice", t.Name)
		}
		p := filepath.Join(outDir, t.File)
		f, err := safetensors.Open(p)
		if err != nil {
			return nil, fmt.Errorf("worker output: %w", err)
		}
		got, ok := f.Lookup(t.Name)
		n := len(f.Tensors)
		_ = f.Close()
		if !ok || n != 1 || got.DType != want.DType || !slices.Equal(got.Shape, want.Shape) {
			return nil, fmt.Errorf("worker output %s does not hold %s as %s%v", t.File, t.Name, want.DType, want.Shape)
		}
		merged[t.Name] = p
	}
	return merged, nil
}

// writer stores blobs and counts what is new.
type writer struct {
	s        *store.Store
	newBlobs int
	newBytes int64
}

func (w *writer) put(r io.Reader, mediaType string, annotations map[string]string) (ocispec.Descriptor, error) {
	d, size, created, err := w.s.Ingest(r, "")
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if created {
		w.newBlobs++
		w.newBytes += size
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: digest.Digest(d), Size: size, Annotations: annotations}, nil
}

func (w *writer) putJSON(mediaType string, v any) (ocispec.Descriptor, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	return w.put(bytes.NewReader(raw), mediaType, nil)
}

// relayout rebuilds the base layer's blobs with merged tensors swapped in.
// A blob none of whose tensors changed is reused as it is; one that holds a
// changed tensor (alone, or packed with small ones) is rewritten in the
// same canonical form, so the layout matches what a build would produce.
func (w *writer) relayout(base []ocispec.Descriptor, merged map[string]string) ([]ocispec.Descriptor, error) {
	placed := 0
	var out []ocispec.Descriptor
	for _, d := range base {
		if spec.BaseMediaType(d.MediaType) != spec.MediaTypeTensor {
			out = append(out, d)
			continue
		}
		nd, n, err := w.rewrite(d, merged)
		if err != nil {
			return nil, err
		}
		placed += n
		out = append(out, nd)
	}
	if placed != len(merged) {
		return nil, errors.New("some merged tensors belong to no base blob (this is a bug)")
	}
	return out, nil
}

func (w *writer) rewrite(d ocispec.Descriptor, merged map[string]string) (ocispec.Descriptor, int, error) {
	p, err := w.s.Path(d.Digest.String())
	if err != nil {
		return ocispec.Descriptor{}, 0, err
	}
	f, err := safetensors.Open(p)
	if err != nil {
		return ocispec.Descriptor{}, 0, err
	}
	defer f.Close()
	changed := 0
	var entries []safetensors.Entry
	for _, t := range f.Tensors {
		e := safetensors.Entry{Name: t.Name, DType: t.DType, Shape: t.Shape, Data: f.Data(t)}
		if mp, ok := merged[t.Name]; ok {
			mf, err := safetensors.Open(mp)
			if err != nil {
				return ocispec.Descriptor{}, 0, err
			}
			defer mf.Close()
			mt, _ := mf.Lookup(t.Name)
			e.Data = mf.Data(mt)
			changed++
		}
		entries = append(entries, e)
	}
	if changed == 0 {
		return d, 0, nil
	}
	r, _, err := safetensors.NewReader(entries, nil)
	if err != nil {
		return ocispec.Descriptor{}, 0, err
	}
	nd, err := w.put(r, d.MediaType, maps.Clone(d.Annotations))
	return nd, changed, err
}

// finish writes the config and manifest, checks them and applies tags.
func (w *writer) finish(cfg *spec.Config, layers []ocispec.Descriptor, annotations map[string]string, tags []string) (*api.BuildResult, error) {
	cfgDesc, err := w.putJSON(spec.MediaTypeConfig, cfg)
	if err != nil {
		return nil, err
	}
	m := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: spec.ArtifactType,
		Config:       cfgDesc,
		Layers:       layers,
		Annotations:  annotations,
	}
	if err := spec.CheckImage(&m, cfg); err != nil {
		return nil, fmt.Errorf("squash produced an inconsistent image (this is a bug): %w", err)
	}
	mDesc, err := w.putJSON(ocispec.MediaTypeImageManifest, m)
	if err != nil {
		return nil, err
	}
	mDesc.ArtifactType = spec.ArtifactType
	res := &api.BuildResult{Manifest: mDesc, Config: cfg, NewBlobs: w.newBlobs, NewBytes: w.newBytes}
	seen := map[digest.Digest]bool{}
	for _, d := range append(layers, cfgDesc, mDesc) {
		if !seen[d.Digest] {
			seen[d.Digest] = true
			res.Blobs++
			res.Size += d.Size
		}
	}
	for _, t := range tags {
		name, err := w.s.Tag(t, mDesc)
		if err != nil {
			return nil, err
		}
		res.Tags = append(res.Tags, name)
	}
	return res, nil
}
