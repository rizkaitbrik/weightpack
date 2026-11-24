// Package build turns a Weightfile into an image in the local store.
//
// It handles local sources: FROM a directory holding safetensors weights,
// config.json and a tokenizer; LAYER and EXPERT from PEFT LoRA adapter
// directories. Registry and Hugging Face sources, imports of full fine-tunes
// (codec=xor|bit1|sparse) and ROUTER train are rejected with a clear error
// until they land.
//
// The builder never reads tensor values. It splits safetensors files along
// the byte ranges their headers give, checks adapter shapes against the base,
// and hands the bytes to the store, which hashes them as it writes.
package build

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
)

// Options says what to build.
type Options struct {
	// Weightfile is the path of the Weightfile. Local sources, TEMPLATE and
	// LICENSE paths resolve against its directory.
	Weightfile string
	// Tags to point at the built image, such as "assistant:v1".
	Tags []string
}

// Image-level meta blobs get these titles.
const (
	titleTemplate = "chat_template.jinja"
	titleLicense  = "LICENSE"
)

type builder struct {
	ctx      context.Context
	store    *store.Store
	dir      string
	file     string
	progress func(api.Event)

	// blobs holds each layer's manifest descriptors in build order; the
	// key "" holds image-level meta blobs.
	blobs    map[string][]ocispec.Descriptor
	newBlobs int
	newBytes int64
}

// Build reads the Weightfile, writes every blob of the image, its config and
// its manifest to the store, and tags the result. progress, if not nil,
// receives one event per step and for each warning.
//
// Building the same inputs twice gives the same manifest digest: nothing in
// the image depends on paths, file order or time.
func Build(ctx context.Context, s *store.Store, opts Options, progress func(api.Event)) (*api.BuildResult, error) {
	if progress == nil {
		progress = func(api.Event) {}
	}
	for _, t := range opts.Tags {
		if _, err := reference.ParseNamed(t); err != nil {
			return nil, err
		}
	}
	wf, err := weightfile.ParseFile(opts.Weightfile)
	if err != nil {
		return nil, err
	}
	cfg, err := wf.Config()
	if err != nil {
		return nil, err
	}
	if err := checkSupported(opts.Weightfile, wf); err != nil {
		return nil, err
	}

	b := &builder{
		ctx: ctx, store: s, progress: progress,
		dir: filepath.Dir(opts.Weightfile), file: opts.Weightfile,
		blobs: map[string][]ocispec.Descriptor{},
	}
	byID := make(map[string]*spec.Layer, len(cfg.Layers))
	for i := range cfg.Layers {
		byID[cfg.Layers[i].ID] = &cfg.Layers[i]
	}

	baseLayer := byID[weightfile.ReservedBaseID]
	base, err := b.ingestBase(wf.From, baseLayer)
	if err != nil {
		return nil, err
	}
	cfg.Arch = base.arch
	cfg.Tokenizer = base.tokenizer

	parent := baseLayer
	for _, l := range wf.Layers {
		layer := byID[l.ID]
		if err := b.ingestLoRA("LAYER "+l.ID, l, layer, parent, base, nil); err != nil {
			return nil, err
		}
		parent = layer
	}
	if m := wf.Mix; m != nil {
		mix := byID[m.ID]
		var experts []string
		for _, e := range m.Experts {
			layer := byID[e.ID]
			if err := b.ingestLoRA("EXPERT "+e.ID, e, layer, parent, base, m); err != nil {
				return nil, err
			}
			experts = append(experts, layer.Digest)
		}
		mix.Digest = spec.LayerDigest(*mix, nil, experts)
		mix.Chain = spec.ChainID(parent.Chain, mix.Digest)
		b.progress(api.Event{Step: "MIX " + m.ID, Message: fmt.Sprintf("%d experts, top_k=%d, router %s", len(m.Experts), m.TopK, m.Router.Kind)})
	}

	annotations, err := b.imageMeta(wf)
	if err != nil {
		return nil, err
	}
	return b.finish(cfg, annotations, opts.Tags)
}

// checkSupported rejects, all at once, what the builder cannot do yet.
func checkSupported(path string, wf *weightfile.File) error {
	var errs []error
	fail := func(line int, format string, args ...any) {
		errs = append(errs, &weightfile.Error{File: path, Line: line, Msg: fmt.Sprintf(format, args...)})
	}
	if k := wf.From.Source.Kind; k != weightfile.SourceLocal {
		fail(wf.From.Line, "FROM %s: only local directories can be built from so far; %s sources come with wpk pull", wf.From.Source, k)
	}
	layers := append([]weightfile.Layer{}, wf.Layers...)
	if wf.Mix != nil {
		layers = append(layers, wf.Mix.Experts...)
		if wf.Mix.Router.Kind == "train" {
			fail(wf.Mix.Router.Line, "ROUTER train lands in phase 3; use ROUTER arrow, which needs no training")
		}
	}
	for _, l := range layers {
		if l.Source.Kind != weightfile.SourceLocal {
			fail(l.Line, "%s: only local adapter directories can be built from so far; %s sources come with wpk pull", l.ID, l.Source.Kind)
		}
		if l.Codec != "" && l.Codec != weightfile.CodecLoRA {
			fail(l.Line, "%s: codec=%s imports a full fine-tune; delta import lands in phase 2", l.ID, l.Codec)
		}
	}
	return errors.Join(errs...)
}

// put stores a blob and returns its descriptor.
func (b *builder) put(r io.Reader, mediaType string, annotations map[string]string) (ocispec.Descriptor, error) {
	if err := b.ctx.Err(); err != nil {
		return ocispec.Descriptor{}, err
	}
	d, size, created, err := b.store.Ingest(r, "")
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if created {
		b.newBlobs++
		b.newBytes += size
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: digest.Digest(d), Size: size, Annotations: annotations}, nil
}

// putJSON stores v as compact JSON.
func (b *builder) putJSON(mediaType string, v any) (ocispec.Descriptor, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	return b.put(bytes.NewReader(raw), mediaType, nil)
}

// putFile stores a regular file as a meta blob.
func (b *builder) putFile(path string, annotations map[string]string) (ocispec.Descriptor, error) {
	f, err := os.Open(path)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	defer f.Close()
	if info, err := f.Stat(); err != nil {
		return ocispec.Descriptor{}, err
	} else if !info.Mode().IsRegular() {
		return ocispec.Descriptor{}, fmt.Errorf("%s is not a regular file", path)
	}
	return b.put(f, spec.MediaTypeMeta, annotations)
}

// resolve turns a path written in the Weightfile into a path on disk.
func (b *builder) resolve(p string) string {
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Join(b.dir, p)
}

func isPath(v string) bool {
	return v == "." || strings.HasPrefix(v, "./") || strings.HasPrefix(v, "../") || strings.HasPrefix(v, "/")
}

// imageMeta stores TEMPLATE and a LICENSE file as image-level meta blobs. An
// SPDX license id becomes the manifest's licenses annotation; "inherit"
// leaves the base's license files in charge.
func (b *builder) imageMeta(wf *weightfile.File) (map[string]string, error) {
	annotations := map[string]string{}
	if t := wf.Template; t != nil {
		d, err := b.putFile(b.resolve(t.Value), map[string]string{spec.AnnotationTitle: titleTemplate})
		if err != nil {
			return nil, &weightfile.Error{File: b.file, Line: t.Line, Msg: "TEMPLATE: " + err.Error()}
		}
		b.blobs[""] = append(b.blobs[""], d)
		b.progress(api.Event{Step: "TEMPLATE", Message: t.Value})
	}
	if l := wf.License; l != nil {
		switch {
		case strings.EqualFold(l.Value, "inherit"):
		case isPath(l.Value):
			d, err := b.putFile(b.resolve(l.Value), map[string]string{spec.AnnotationTitle: titleLicense})
			if err != nil {
				return nil, &weightfile.Error{File: b.file, Line: l.Line, Msg: "LICENSE: " + err.Error()}
			}
			b.blobs[""] = append(b.blobs[""], d)
		default:
			annotations[ocispec.AnnotationLicenses] = l.Value
		}
		b.progress(api.Event{Step: "LICENSE", Message: l.Value})
	}
	if len(annotations) == 0 {
		return nil, nil
	}
	return annotations, nil
}

// finish writes the config and manifest, checks them against each other and
// applies the tags.
func (b *builder) finish(cfg *spec.Config, annotations map[string]string, tags []string) (*api.BuildResult, error) {
	cfgDesc, err := b.putJSON(spec.MediaTypeConfig, cfg)
	if err != nil {
		return nil, err
	}
	var layers []ocispec.Descriptor
	for _, l := range cfg.Layers {
		layers = append(layers, b.blobs[l.ID]...)
	}
	layers = append(layers, b.blobs[""]...)

	m := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: spec.ArtifactType,
		Config:       cfgDesc,
		Layers:       layers,
		Annotations:  annotations,
	}
	if err := spec.CheckImage(&m, cfg); err != nil {
		return nil, fmt.Errorf("build produced an inconsistent image (this is a bug): %w", err)
	}
	mDesc, err := b.putJSON(ocispec.MediaTypeImageManifest, m)
	if err != nil {
		return nil, err
	}
	mDesc.ArtifactType = spec.ArtifactType

	res := &api.BuildResult{Manifest: mDesc, Config: cfg, NewBlobs: b.newBlobs, NewBytes: b.newBytes}
	seen := map[digest.Digest]bool{}
	for _, d := range append(layers, cfgDesc, mDesc) {
		if !seen[d.Digest] {
			seen[d.Digest] = true
			res.Blobs++
			res.Size += d.Size
		}
	}
	for _, t := range tags {
		name, err := b.store.Tag(t, mDesc)
		if err != nil {
			return nil, err
		}
		res.Tags = append(res.Tags, name)
	}
	return res, nil
}
