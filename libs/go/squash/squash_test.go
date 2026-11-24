package squash

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"google.golang.org/grpc"

	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/workerpb"
)

// fakeWorker stands in for the Python worker: for every base tensor a layer
// touches it writes the tensor with every byte inverted, so tests can tell
// merged tensors apart without doing any math. tamper may break the reply.
type fakeWorker struct {
	workerpb.WorkerClient
	req    *workerpb.SquashRequest
	tamper func(outDir string, resp *workerpb.SquashResponse)
}

func (f *fakeWorker) Squash(_ context.Context, req *workerpb.SquashRequest, _ ...grpc.CallOption) (*workerpb.SquashResponse, error) {
	f.req = req
	base := map[string]string{} // tensor -> blob path
	for _, b := range req.Base {
		h, err := safetensors.Open(b.Path)
		if err != nil {
			return nil, err
		}
		for _, t := range h.Tensors {
			base[t.Name] = b.Path
		}
		h.Close()
	}
	targets := map[string]bool{}
	for _, l := range req.Layers {
		for _, b := range l.Blobs {
			h, err := safetensors.Open(b.Path)
			if err != nil {
				return nil, err
			}
			for _, t := range h.Tensors {
				if m, ok := strings.CutSuffix(t.Name, ".lora_A.weight"); ok {
					targets[m+".weight"] = true
				}
			}
			h.Close()
		}
	}
	names := make([]string, 0, len(targets))
	for n := range targets {
		names = append(names, n)
	}
	sort.Strings(names)
	resp := &workerpb.SquashResponse{}
	for i, n := range names {
		file := fmt.Sprintf("%d.safetensors", i)
		if err := writeInverted(base[n], n, filepath.Join(req.OutDir, file)); err != nil {
			return nil, err
		}
		resp.Tensors = append(resp.Tensors, &workerpb.MergedTensor{Name: n, File: file})
	}
	if f.tamper != nil {
		f.tamper(req.OutDir, resp)
	}
	return resp, nil
}

func writeInverted(src, name, dst string) error {
	h, err := safetensors.Open(src)
	if err != nil {
		return err
	}
	defer h.Close()
	t, _ := h.Lookup(name)
	data, err := io.ReadAll(h.Data(t))
	if err != nil {
		return err
	}
	for i := range data {
		data[i] = ^data[i]
	}
	r, _, err := safetensors.NewReader([]safetensors.Entry{{Name: name, DType: t.DType, Shape: t.Shape, Data: bytes.NewReader(data)}}, nil)
	if err != nil {
		return err
	}
	out, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, out, 0o644)
}

type fixture struct {
	dir   string
	store *store.Store
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	dir := t.TempDir()
	buildtest.Project(t, dir)
	s, err := store.Open(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	return &fixture{dir: dir, store: s}
}

func (f *fixture) build(t *testing.T, name, weightfile string, tags ...string) ocispec.Descriptor {
	t.Helper()
	p := filepath.Join(f.dir, name)
	if err := os.WriteFile(p, []byte(weightfile), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := build.Build(context.Background(), f.store, build.Options{Weightfile: p, Tags: tags}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res.Manifest
}

func (f *fixture) load(t *testing.T, ref string) (*ocispec.Manifest, *spec.Config) {
	t.Helper()
	d, err := f.store.Resolve(ref)
	if err != nil {
		t.Fatal(err)
	}
	var m ocispec.Manifest
	var c spec.Config
	if err := f.store.ReadJSON(d.Digest.String(), &m); err != nil {
		t.Fatal(err)
	}
	if err := f.store.ReadJSON(m.Config.Digest.String(), &c); err != nil {
		t.Fatal(err)
	}
	return &m, &c
}

func tensorBlobs(m *ocispec.Manifest) map[string]ocispec.Descriptor {
	out := map[string]ocispec.Descriptor{}
	for _, d := range m.Layers {
		if d.MediaType == spec.MediaTypeTensor {
			key := d.Annotations[spec.AnnotationTensor]
			if key == "" {
				key = "(packed)"
			}
			out[key] = d
		}
	}
	return out
}

func TestSquashStack(t *testing.T) {
	f := newFixture(t)
	src := f.build(t, "French.Weightfile", "FROM ./base\nLAYER french ./adapters/french\nTEMPLATE ./chat.jinja\nLICENSE MIT\n", "french")
	w := &fakeWorker{}
	res, err := Squash(context.Background(), f.store, w, "french", Options{Tags: []string{"french-dense"}}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(w.req.Layers) != 1 || w.req.Layers[0].Id != "french" || w.req.Layers[0].Scaling != 2 || len(w.req.Layers[0].Blobs) != 4 {
		t.Errorf("request layers = %+v", w.req.Layers)
	}
	m, c := f.load(t, "french-dense")
	if err := spec.CheckImage(m, c); err != nil {
		t.Fatal(err)
	}
	srcM, srcC := f.load(t, "french")
	if len(c.Layers) != 1 || c.Layers[0].Kind != spec.KindBase || c.Arch != srcC.Arch || c.Tokenizer != srcC.Tokenizer {
		t.Errorf("config = %+v", c)
	}
	if m.Annotations[spec.AnnotationSquashed] != "french" || m.Annotations[spec.AnnotationSquashedFrom] != src.Digest.String() ||
		m.Annotations[ocispec.AnnotationLicenses] != "MIT" {
		t.Errorf("annotations = %v", m.Annotations)
	}

	// french adapts q and v in both blocks: those four blobs change and hold
	// the worker's output; every other base blob is reused as it was.
	before, after := tensorBlobs(srcM), tensorBlobs(m)
	if len(before) != len(after) {
		t.Fatalf("%d tensor blobs before, %d after", len(before), len(after))
	}
	var changed []string
	for name, d := range after {
		if d.Digest != before[name].Digest {
			changed = append(changed, name)
		}
	}
	sort.Strings(changed)
	want := []string{
		"model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.v_proj.weight",
		"model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.v_proj.weight",
	}
	if !slices.Equal(changed, want) {
		t.Errorf("changed blobs = %v, want %v", changed, want)
	}
	if res.NewBlobs != 4+2 {
		t.Errorf("new blobs = %d, want 6 (4 tensors, config, manifest)", res.NewBlobs)
	}
	var meta, template int
	for _, d := range m.Layers {
		if d.MediaType == spec.MediaTypeMeta {
			meta++
			if d.Annotations[spec.AnnotationLayer] == "" {
				template++
			}
		}
	}
	if meta != 3 || template != 1 {
		t.Errorf("meta blobs = %d (image-level %d), want 3 (1)", meta, template)
	}
}

func TestSquashPackedTensor(t *testing.T) {
	f := newFixture(t)
	// An adapter on lm_head, which the base keeps in its packed blob.
	dir := filepath.Join(f.dir, "adapters/head")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	buildtest.WriteSafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), []buildtest.Tensor{
		{Name: "base_model.model.lm_head.lora_A.weight", Shape: []int64{2, buildtest.Hidden}, Seed: 1},
		{Name: "base_model.model.lm_head.lora_B.weight", Shape: []int64{buildtest.Vocab, 2}, Seed: 2},
	})
	buildtest.WriteJSON(t, filepath.Join(dir, "adapter_config.json"), map[string]any{"peft_type": "LORA", "r": 2, "lora_alpha": 2, "use_rslora": true})
	f.build(t, "Head.Weightfile", "FROM ./base\nLAYER head ./adapters/head\n", "head")

	w := &fakeWorker{}
	if _, err := Squash(context.Background(), f.store, w, "head", Options{Tags: []string{"head-dense"}}, nil); err != nil {
		t.Fatal(err)
	}
	if got, want := w.req.Layers[0].Scaling, 2/1.4142135623730951; got != want {
		t.Errorf("rslora scaling = %v, want %v", got, want)
	}
	srcM, _ := f.load(t, "head")
	m, _ := f.load(t, "head-dense")
	before, after := tensorBlobs(srcM), tensorBlobs(m)
	for name, d := range after {
		if (name == "(packed)") == (d.Digest == before[name].Digest) {
			t.Errorf("%s: changed = %v, want only the packed blob to change", name, d.Digest != before[name].Digest)
		}
	}
	// The packed blob still holds the same tensors, lm_head now inverted.
	p, _ := f.store.Path(after["(packed)"].Digest.String())
	h, err := safetensors.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if len(h.Tensors) != 7 {
		t.Errorf("packed blob holds %d tensors, want 7", len(h.Tensors))
	}
}

func TestSquashMixNeedsOneExpert(t *testing.T) {
	f := newFixture(t)
	f.build(t, "Weightfile", buildtest.Assistant, "assistant")
	ctx := context.Background()
	for expert, want := range map[string]string{"": "pick one with --expert", "nope": "no expert \"nope\""} {
		_, err := Squash(ctx, f.store, &fakeWorker{}, "assistant", Options{Expert: expert}, nil)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("expert %q: err = %v", expert, err)
		}
	}
	w := &fakeWorker{}
	if _, err := Squash(ctx, f.store, w, "assistant", Options{Expert: "legal", Tags: []string{"legal"}}, nil); err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, l := range w.req.Layers {
		ids = append(ids, l.Id)
	}
	if !slices.Equal(ids, []string{"french", "legal"}) {
		t.Errorf("baked layers = %v, want french then legal", ids)
	}
	m, _ := f.load(t, "legal")
	if m.Annotations[spec.AnnotationSquashed] != "french,legal" {
		t.Errorf("annotations = %v", m.Annotations)
	}

	f.build(t, "Base.Weightfile", "FROM ./base\n", "plain")
	if _, err := Squash(ctx, f.store, &fakeWorker{}, "plain", Options{}, nil); err == nil || !strings.Contains(err.Error(), "only a base") {
		t.Errorf("base-only image: %v", err)
	}
	if _, err := Squash(ctx, f.store, &fakeWorker{}, "plain", Options{Expert: "x"}, nil); err == nil || !strings.Contains(err.Error(), "no MIX") {
		t.Errorf("expert without a mix: %v", err)
	}
}

func TestSquashRejectsBadWorkerOutput(t *testing.T) {
	cases := map[string]func(outDir string, resp *workerpb.SquashResponse){
		"escape":  func(_ string, r *workerpb.SquashResponse) { r.Tensors[0].File = "../../x.safetensors" },
		"unknown": func(_ string, r *workerpb.SquashResponse) { r.Tensors[0].Name = "model.nope.weight" },
		"twice":   func(_ string, r *workerpb.SquashResponse) { r.Tensors = append(r.Tensors, r.Tensors[0]) },
		"mislabel": func(_ string, r *workerpb.SquashResponse) {
			r.Tensors[0].Name, r.Tensors[1].Name = r.Tensors[1].Name, r.Tensors[0].Name
		},
		"garbage": func(dir string, r *workerpb.SquashResponse) {
			_ = os.WriteFile(filepath.Join(dir, r.Tensors[0].File), []byte("not safetensors"), 0o644)
		},
	}
	for name, tamper := range cases {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t)
			f.build(t, "French.Weightfile", "FROM ./base\nLAYER french ./adapters/french\n", "french")
			_, err := Squash(context.Background(), f.store, &fakeWorker{tamper: tamper}, "french", Options{Tags: []string{"dense"}}, nil)
			if err == nil {
				t.Fatal("bad worker output accepted")
			}
			if _, err := f.store.Resolve("dense"); err == nil {
				t.Error("a failed squash tagged an image")
			}
		})
	}
}
