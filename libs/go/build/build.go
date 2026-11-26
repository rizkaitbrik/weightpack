// Package build turns a Weightfile into an image in the local store.
//
// Sources: FROM a directory holding safetensors weights, config.json and a
// tokenizer, or FROM an image (in the store, or pulled through
// Options.Pull); LAYER and EXPERT from a PEFT LoRA adapter directory, or
// from the top layer of an image built on the same chain. Hugging Face
// sources, imports of full fine-tunes (codec=xor|bit1|sparse) and ROUTER
// train are rejected with a clear error until they are supported.
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
	"github.com/rizkaitbrik/weightpack/libs/go/hf"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// Options says what to build.
type Options struct {
	// Weightfile is the path of the Weightfile. Local sources, TEMPLATE and
	// LICENSE paths resolve against its directory.
	Weightfile string
	// Tags to point at the built image, such as "assistant:v1".
	Tags []string
	// Pull fetches an image a FROM, LAYER or EXPERT names when the store
	// does not have it. Nil means such images must be in the store already.
	Pull PullFunc
	// Worker encodes codec=xor layers. Nil means such layers cannot be built.
	Worker workerpb.WorkerClient
	// HF fetches hf:// sources. Nil means such sources cannot be built.
	HF *hf.Client
	// NoCache reads and hashes every source again instead of reusing what
	// the build cache remembers.
	NoCache bool
}

// PullFunc fetches ref into the store, reporting progress.
type PullFunc func(ctx context.Context, ref string, progress func(api.Event)) error

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
	pull     PullFunc
	worker   workerpb.WorkerClient
	hf       *hf.Client
	cache    *buildCache

	// inherited is what a FROM image passes on, nil for a local FROM.
	inherited *inherited

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
		ctx: ctx, store: s, progress: progress, pull: opts.Pull, worker: opts.Worker, hf: opts.HF, cache: newBuildCache(CacheDir(s.Root())),
		dir: filepath.Dir(opts.Weightfile), file: opts.Weightfile,
		blobs: map[string][]ocispec.Descriptor{},
	}

	if opts.NoCache {
		b.cache.dir = ""
	}

	var base *baseModel
	top := weightfile.ReservedBaseID
	if wf.From.Source.Kind != weightfile.SourceOCI {
		base, err = b.ingestBase(wf.From, &cfg.Layers[0])
		if err != nil {
			return nil, err
		}
	} else {
		inh, err := b.fromImage(wf.From)
		if err != nil {
			return nil, err
		}
		if err := b.graft(wf, cfg, inh); err != nil {
			return nil, err
		}
		base = inh.base
		top = inh.layers[len(inh.layers)-1].ID
	}
	cfg.Arch = base.arch
	cfg.Tokenizer = base.tokenizer

	byID := make(map[string]*spec.Layer, len(cfg.Layers))
	for i := range cfg.Layers {
		byID[cfg.Layers[i].ID] = &cfg.Layers[i]
	}
	parent := byID[top]
	for _, l := range wf.Layers {
		layer := byID[l.ID]
		if err := b.ingestLayer("LAYER "+l.ID, l, layer, parent, base, nil, byID); err != nil {
			return nil, err
		}
		parent = layer
	}
	if m := wf.Mix; m != nil {
		mix := byID[m.ID]
		var experts []string
		for _, e := range m.Experts {
			layer := byID[e.ID]
			if err := b.ingestLayer("EXPERT "+e.ID, e, layer, parent, base, m, byID); err != nil {
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
	if b.cache.hits > 0 {
		b.progress(api.Event{Step: "cache", Message: fmt.Sprintf("reused %d blobs without reading their sources", b.cache.hits)})
	}
	return b.finish(cfg, annotations, opts.Tags)
}

// checkSupported rejects, all at once, what the builder cannot do yet.
func checkSupported(path string, wf *weightfile.File) error {
	var errs []error
	fail := func(line int, format string, args ...any) {
		errs = append(errs, &weightfile.Error{File: path, Line: line, Msg: fmt.Sprintf(format, args...)})
	}
	layers := append([]weightfile.Layer{}, wf.Layers...)
	if wf.Mix != nil {
		layers = append(layers, wf.Mix.Experts...)
		if wf.Mix.Router.Kind == "train" {
			fail(wf.Mix.Router.Line, "ROUTER train is not supported yet; use ROUTER arrow, which needs no training")
		}
	}
	for _, l := range layers {
		delta := l.Codec != "" && l.Codec != weightfile.CodecLoRA
		switch {
		case l.Source.Kind == weightfile.SourceOCI && delta:
			fail(l.Line, "%s: codec= applies to importing a full fine-tune; a layer from an image is stored already", l.ID)
		case delta && l.Codec != weightfile.CodecXOR:
			fail(l.Line, "%s: codec=%s, a lossy import, is not supported yet; codec=xor imports losslessly", l.ID, l.Codec)
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
	key := b.cache.fileBlobKey(path)
	if d, ok := b.cache.get(b.store, key); ok {
		d.MediaType, d.Annotations = spec.MediaTypeMeta, annotations
		return d, nil
	}
	d, err := b.put(f, spec.MediaTypeMeta, annotations)
	if err == nil {
		b.cache.set(key, d)
	}
	return d, err
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
// leaves the base's license files, or a FROM image's license, in charge.
// LABEL lines become manifest annotations too. A FROM image's template and
// license carry over unless the Weightfile sets its own; its labels do not.
func (b *builder) imageMeta(wf *weightfile.File) (map[string]string, error) {
	annotations := map[string]string{}
	if t := wf.Template; t != nil {
		d, err := b.putFile(b.resolve(t.Value), map[string]string{spec.AnnotationTitle: titleTemplate})
		if err != nil {
			return nil, &weightfile.Error{File: b.file, Line: t.Line, Msg: "TEMPLATE: " + err.Error()}
		}
		b.blobs[""] = append(b.blobs[""], d)
		b.progress(api.Event{Step: "TEMPLATE", Message: t.Value})
	} else if inh := b.inherited; inh != nil && inh.template != nil {
		b.blobs[""] = append(b.blobs[""], *inh.template)
	}
	// Without LICENSE, or with LICENSE inherit, a FROM image's license
	// carries over.
	if inh := b.inherited; inh != nil && (wf.License == nil || strings.EqualFold(wf.License.Value, "inherit")) {
		switch {
		case inh.licenseFile != nil:
			b.blobs[""] = append(b.blobs[""], *inh.licenseFile)
		case inh.license != "":
			annotations[ocispec.AnnotationLicenses] = inh.license
		}
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
	for _, l := range wf.Labels {
		annotations[l.Key] = l.Value
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
