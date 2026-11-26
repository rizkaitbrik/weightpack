package build

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// ServingFiles are what an engine loads to serve an image: a Hugging Face
// model directory with every layer below the adapters baked in, and a PEFT
// adapter directory per adapter, by layer id.
type ServingFiles struct {
	Base     string
	Adapters map[string]string
}

// ServingFilesFunc writes an image's serving files under dir, which does
// not exist yet. The daemon provides it (serving.WriteServingFiles), since
// baking a base needs packages that depend on this one.
type ServingFilesFunc func(ctx context.Context, image, dir string, progress func(api.Event)) (*ServingFiles, error)

// Router suffixes in a router file (spec.RouterFile).
const (
	routerWeight = ".router.weight"
	routerBias   = ".router.bias"
)

// trainedRouter gives a MIX its trained router: the file a ROUTER <path>
// line names, or one trained on ROUTER train's data with the experts
// frozen. It stores the router as the mix's router.v1 blob and settles the
// mix's digest and chain ID, which cover it.
func (b *builder) trainedRouter(m *weightfile.Mix, mix, parent *spec.Layer, byID map[string]*spec.Layer, base *baseModel, cfg *spec.Config, wf *weightfile.File) error {
	step := "ROUTER " + m.Router.Kind
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: m.Router.Line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	routed := b.routedModules(m, base)
	var d ocispec.Descriptor
	var err error
	if m.Router.Kind == "file" {
		step = "ROUTER " + m.Router.Path
		d, err = b.putRouter(b.resolve(m.Router.Path), m, routed)
	} else {
		d, err = b.trainRouter(step, m, mix, parent, byID, cfg, wf, routed)
	}
	if err != nil {
		return fail("%v", err)
	}
	b.setRouter(mix, parent, byID, &d)
	return nil
}

// setRouter puts d in the mix as its router, or with d nil leaves it
// Arrow's, and settles the mix's digest and chain ID.
func (b *builder) setRouter(mix, parent *spec.Layer, byID map[string]*spec.Layer, d *ocispec.Descriptor) {
	var refs []spec.BlobRef
	if d == nil {
		mix.Router = spec.RouterArrow
		delete(b.blobs, mix.ID)
	} else {
		mix.Router = spec.RouterTrained
		b.blobs[mix.ID] = []ocispec.Descriptor{*d}
		refs = []spec.BlobRef{spec.BlobRefFromDescriptor(*d)}
	}
	var experts []string
	for _, e := range mix.Experts {
		experts = append(experts, byID[e].Digest)
	}
	mix.Digest = spec.LayerDigest(*mix, refs, experts)
	mix.Chain = spec.ChainID(parent.Chain, mix.Digest)
}

// routedModules are the modules the mix routes in: those its experts touch
// within its targets and blocks, each with its input size.
func (b *builder) routedModules(m *weightfile.Mix, base *baseModel) map[string]int64 {
	out := map[string]int64{}
	for _, e := range m.Experts {
		for _, d := range b.blobs[e.ID] {
			if spec.BaseMediaType(d.MediaType) != spec.MediaTypeLoRA {
				continue
			}
			module := strings.TrimSuffix(d.Annotations[spec.AnnotationTensor], ".weight")
			if !containsString(m.Targets, targetName(module)) || !inBlocks(module, m.Blocks) {
				continue
			}
			if t, ok := base.tensors[module+".weight"]; ok && len(t.Shape) == 2 {
				out[module] = t.Shape[1]
			}
		}
	}
	return out
}

func containsString(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

// putRouter checks a router file against the mix and stores it as one
// canonical router.v1 blob: float32 weights [experts, in] and biases
// [experts] for exactly the modules the mix routes in.
func (b *builder) putRouter(path string, m *weightfile.Mix, routed map[string]int64) (ocispec.Descriptor, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	defer f.Close()
	n := int64(len(m.Experts))
	have := map[string]int{}
	var refs []tensorRef
	for _, t := range f.Tensors {
		module, suffix := "", ""
		switch {
		case strings.HasSuffix(t.Name, routerWeight):
			module, suffix = strings.TrimSuffix(t.Name, routerWeight), routerWeight
		case strings.HasSuffix(t.Name, routerBias):
			module, suffix = strings.TrimSuffix(t.Name, routerBias), routerBias
		default:
			return ocispec.Descriptor{}, fmt.Errorf("%s: tensor %s is neither a router weight nor a bias", path, t.Name)
		}
		in, ok := routed[module]
		if !ok {
			return ocispec.Descriptor{}, fmt.Errorf("%s: %s is not a module MIX %s routes in", path, module, m.ID)
		}
		want := []int64{n}
		if suffix == routerWeight {
			want = []int64{n, in}
		}
		if t.DType != "F32" || !equalShape(t.Shape, want) {
			return ocispec.Descriptor{}, fmt.Errorf("%s: %s is %s %v, want F32 %v (one row per expert)", path, t.Name, t.DType, t.Shape, want)
		}
		have[module]++
		refs = append(refs, tensorRef{file: f, t: t, name: t.Name})
	}
	var missing []string
	for module := range routed {
		if have[module] != 2 {
			missing = append(missing, module)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return ocispec.Descriptor{}, fmt.Errorf("%s has no router weight or bias for %d modules MIX %s routes in, such as %s", path, len(missing), m.ID, missing[0])
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].name < refs[j].name })
	return b.putTensors(refs, spec.MediaTypeRouter, map[string]string{
		spec.AnnotationLayer: m.ID, spec.AnnotationTitle: spec.RouterFile,
	})
}

func equalShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// trainRouter trains the mix's router in the build worker, on the files an
// engine would serve the mix from, or reuses the router an earlier build
// trained from the same chain, data, steps and worker version.
func (b *builder) trainRouter(step string, m *weightfile.Mix, mix, parent *spec.Layer, byID map[string]*spec.Layer, cfg *spec.Config, wf *weightfile.File, routed map[string]int64) (ocispec.Descriptor, error) {
	data := b.resolve(m.Router.Data)
	dataDigest, err := fileDigest(data)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("data: %w", err)
	}
	info, err := b.worker.Info(b.ctx, &workerpb.InfoRequest{})
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("ask the build worker its version: %w", err)
	}
	template, err := b.chatTemplate(wf)
	if err != nil {
		return ocispec.Descriptor{}, err
	}

	// The image with Arrow's router: what training starts from and runs on.
	b.setRouter(mix, parent, byID, nil)
	key := ""
	if b.cache.dir != "" {
		key = hashKey("router-train", mix.Chain, dataDigest, strconv.Itoa(m.Router.Steps), template.digest, info.Version)
	}
	if d, ok := b.cache.get(b.store, key); ok {
		d.MediaType = spec.MediaTypeRouter
		d.Annotations = map[string]string{spec.AnnotationLayer: m.ID, spec.AnnotationTitle: spec.RouterFile}
		b.cache.hits++
		b.progress(api.Event{Step: step, Message: "reusing the router an earlier build trained on the same chain and data"})
		return d, nil
	}
	newBlobs, newBytes := b.newBlobs, b.newBytes
	arrow, err := b.finish(cfg, nil, nil)
	b.newBlobs, b.newBytes = newBlobs, newBytes
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	tmp, err := b.store.TempDir("router-")
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	defer os.RemoveAll(tmp)
	templatePath, err := template.write(b, tmp)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	b.progress(api.Event{Step: step, Message: "writing the serving base and experts to train on"})
	files, err := b.servingFiles(b.ctx, arrow.Manifest.Digest.String(), filepath.Join(tmp, "serving"), b.progress)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	req := &workerpb.TrainRouterRequest{
		BaseDir: files.Base, Targets: m.Targets, Blocks: m.Blocks, TopK: int32(m.TopK),
		Data: data, Steps: int32(m.Router.Steps), OutFile: filepath.Join(tmp, spec.RouterFile),
		ChatTemplate: templatePath,
	}
	for _, e := range m.Experts {
		dir, ok := files.Adapters[e.ID]
		if !ok {
			return ocispec.Descriptor{}, fmt.Errorf("no serving files for expert %s", e.ID)
		}
		req.Experts = append(req.Experts, &workerpb.Adapter{Id: e.ID, Dir: dir})
	}
	stream, err := b.worker.TrainRouter(b.ctx, req)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("train: %w", err)
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("training on %s for %d steps, experts frozen", m.Router.Data, m.Router.Steps)})
	tenth := int32(0)
	for {
		p, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return ocispec.Descriptor{}, errors.New("the build worker stopped before it finished training")
		}
		if err != nil {
			return ocispec.Descriptor{}, fmt.Errorf("train: %w", err)
		}
		if p.Done {
			b.progress(api.Event{Step: step, Message: fmt.Sprintf(
				"held-out loss %.4f with the trained router, %.4f with Arrow (%d examples, %d held out, %s, %s)",
				p.TrainedLoss, p.ArrowLoss, p.Examples, p.Heldout, p.Device,
				(time.Duration(p.Seconds) * time.Second).Round(time.Second))})
			break
		}
		if p.Steps > 0 && p.Step*10/p.Steps > tenth {
			tenth = p.Step * 10 / p.Steps
			b.progress(api.Event{Step: step, Message: fmt.Sprintf("step %d of %d, loss %.4f", p.Step, p.Steps, p.Loss)})
		}
	}
	d, err := b.putRouter(req.OutFile, m, routed)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("the trained router: %w", err)
	}
	b.cache.set(key, d)
	return d, nil
}

// templateSource is the chat template the image will have: the
// Weightfile's TEMPLATE, else a FROM image's, else none (the tokenizer's).
type templateSource struct {
	path   string // a TEMPLATE file
	blob   string // or a FROM image's template blob
	digest string // of its contents, for the cache key
}

func (b *builder) chatTemplate(wf *weightfile.File) (templateSource, error) {
	if t := wf.Template; t != nil {
		p := b.resolve(t.Value)
		d, err := fileDigest(p)
		if err != nil {
			return templateSource{}, fmt.Errorf("TEMPLATE: %w", err)
		}
		return templateSource{path: p, digest: d}, nil
	}
	if inh := b.inherited; inh != nil && inh.template != nil {
		d := inh.template.Digest.String()
		return templateSource{blob: d, digest: d}, nil
	}
	return templateSource{}, nil
}

// write puts the template where the worker can read it, if it is a blob,
// and returns its path, or "" for none.
func (t templateSource) write(b *builder, dir string) (string, error) {
	if t.blob == "" {
		return t.path, nil
	}
	rc, err := b.store.Open(t.blob)
	if err != nil {
		return "", err
	}
	defer rc.Close()
	p := filepath.Join(dir, "chat_template.jinja")
	f, err := os.Create(p)
	if err != nil {
		return "", err
	}
	_, err = io.Copy(f, rc)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return p, err
}

// fileDigest is the sha256 digest of a regular file's contents.
func fileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if info, err := f.Stat(); err != nil {
		return "", err
	} else if !info.Mode().IsRegular() {
		return "", fmt.Errorf("%s is not a regular file", path)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}
