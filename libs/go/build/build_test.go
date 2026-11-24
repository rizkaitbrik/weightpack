package build

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

type fixture struct {
	dir    string
	store  *store.Store
	events []api.Event
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	dir := t.TempDir()
	s, err := store.Open(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	buildtest.Project(t, dir)
	return &fixture{dir: dir, store: s}
}

func (f *fixture) weightfile(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(f.dir, name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func (f *fixture) build(t *testing.T, wf string, tags ...string) (*api.BuildResult, error) {
	t.Helper()
	f.events = nil
	return Build(context.Background(), f.store, Options{Weightfile: wf, Tags: tags}, func(e api.Event) {
		f.events = append(f.events, e)
	})
}

func (f *fixture) load(t *testing.T, ref string) (*ocispec.Manifest, *spec.Config) {
	t.Helper()
	desc, err := f.store.Resolve(ref)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := f.store.ReadBlob(desc.Digest.String())
	if err != nil {
		t.Fatal(err)
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	raw, err = f.store.ReadBlob(m.Config.Digest.String())
	if err != nil {
		t.Fatal(err)
	}
	var c spec.Config
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return &m, &c
}

func TestBuildStackAndMix(t *testing.T) {
	f := newFixture(t)
	res, err := f.build(t, f.weightfile(t, "Weightfile", buildtest.Assistant), "assistant:v1", "assistant")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(res.Tags, []string{"assistant:v1", "assistant:latest"}) {
		t.Errorf("tags = %v", res.Tags)
	}
	if res.NewBlobs != res.Blobs {
		t.Errorf("first build into an empty store: %d new of %d blobs", res.NewBlobs, res.Blobs)
	}

	m, c := f.load(t, "assistant:v1")
	if err := spec.CheckImage(m, c); err != nil {
		t.Fatalf("stored image fails CheckImage: %v", err)
	}
	if c.Arch.Family != "llama" || c.Arch.Blocks != buildtest.Blocks || c.Arch.Hidden != buildtest.Hidden || !strings.HasPrefix(c.Arch.Fingerprint, "sha256:") {
		t.Errorf("arch = %+v", c.Arch)
	}
	if !strings.HasPrefix(c.Tokenizer, "sha256:") {
		t.Errorf("tokenizer = %q", c.Tokenizer)
	}
	if m.Annotations[ocispec.AnnotationLicenses] != "Apache-2.0" {
		t.Errorf("annotations = %v", m.Annotations)
	}

	layers := map[string]spec.Layer{}
	for _, l := range c.Layers {
		layers[l.ID] = l
	}
	french := layers["french"]
	if french.Rank != 8 || french.Alpha != 16 || !slices.Equal(french.Targets, []string{"q", "v"}) {
		t.Errorf("french = %+v", french)
	}
	if french.Chain != spec.ChainID(layers["base"].Chain, french.Digest) {
		t.Error("french is not chained on base")
	}
	for _, e := range []string{"legal", "medical"} {
		if layers[e].Parent != "french" || layers[e].Chain != spec.ChainID(french.Chain, layers[e].Digest) {
			t.Errorf("expert %s not chained on french: %+v", e, layers[e])
		}
	}

	// Blob layout: large base tensors get their own blob, small ones share
	// one packed blob; each LoRA module is one blob.
	count := map[string]int{}
	var packed, template int
	for _, d := range m.Layers {
		id := d.Annotations[spec.AnnotationLayer]
		count[id]++
		if id == "base" && d.MediaType == spec.MediaTypeTensor && d.Annotations[spec.AnnotationTensor] == "" {
			packed++
		}
		if d.Annotations[spec.AnnotationTitle] == titleTemplate && id == "" {
			template++
		}
	}
	// 14 matrices + 1 packed blob (embed, lm_head, 5 norms) + config.json + tokenizer.json
	if count["base"] != 14+1+2 || packed != 1 {
		t.Errorf("base blobs = %d (packed %d), want 17 (packed 1)", count["base"], packed)
	}
	if count["french"] != 4 || count["legal"] != 3 || count["medical"] != 6 || count["domain"] != 0 || template != 1 {
		t.Errorf("layer blob counts = %v, template %d", count, template)
	}

	// medical covers block 0 too, which MIX domain (blocks=1) leaves out.
	var warned bool
	for _, e := range f.events {
		if e.Warning && e.Step == "EXPERT medical" && strings.Contains(e.Message, "3 of 6 modules lie outside MIX domain") {
			warned = true
		}
		if e.Warning && e.Step == "EXPERT legal" {
			t.Errorf("unexpected warning for legal: %s", e.Message)
		}
	}
	if !warned {
		t.Errorf("no outside-the-mix warning for medical; events: %+v", f.events)
	}
}

func TestRebuildIsReproducibleAndSharesTheBase(t *testing.T) {
	f := newFixture(t)
	wf := f.weightfile(t, "Weightfile", buildtest.Assistant)
	first, err := f.build(t, wf)
	if err != nil {
		t.Fatal(err)
	}
	again, err := f.build(t, wf)
	if err != nil {
		t.Fatal(err)
	}
	if again.Manifest.Digest != first.Manifest.Digest {
		t.Errorf("rebuild changed the digest: %s vs %s", again.Manifest.Digest, first.Manifest.Digest)
	}
	if again.NewBlobs != 0 || again.NewBytes != 0 {
		t.Errorf("rebuild stored %d new blobs (%d bytes)", again.NewBlobs, again.NewBytes)
	}

	// A second image on the same base stores only its own layer, its config
	// and its manifest.
	buildtest.WriteAdapter(t, filepath.Join(f.dir, "adapters/support"), buildtest.Adapter{Rank: 4, Alpha: 4, Modules: []string{"self_attn.o_proj"}, Blocks: []int{0, 1}, Seed: 9})
	support, err := f.build(t, f.weightfile(t, "Support.Weightfile", "FROM ./base\nLAYER support ./adapters/support\n"), "support")
	if err != nil {
		t.Fatal(err)
	}
	if support.NewBlobs != 2+2 {
		t.Errorf("support stored %d new blobs, want 4 (2 LoRA modules, config, manifest)", support.NewBlobs)
	}
	_, c1 := f.load(t, first.Manifest.Digest.String())
	_, c2 := f.load(t, "support")
	if c1.Layers[0].Chain != c2.Layers[0].Chain || c1.Arch.Fingerprint != c2.Arch.Fingerprint {
		t.Error("the two images do not share the base chain and fingerprint")
	}
}

func TestShardLayoutDoesNotChangeTheBase(t *testing.T) {
	f := newFixture(t)
	buildtest.WriteBase(t, filepath.Join(f.dir, "sharded"), 3)
	a, err := f.build(t, f.weightfile(t, "A", "FROM ./base\n"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := f.build(t, f.weightfile(t, "B", "FROM ./sharded\n"))
	if err != nil {
		t.Fatal(err)
	}
	if a.Config.Layers[0].Digest != b.Config.Layers[0].Digest {
		t.Error("one file and three shards of the same tensors gave different base digests")
	}
	if b.NewBlobs > 2 {
		t.Errorf("sharded copy of a stored base added %d blobs, want at most config and manifest", b.NewBlobs)
	}
}

func TestBuildRejects(t *testing.T) {
	cases := []struct {
		name  string
		setup func(t *testing.T, dir string)
		wf    string
		tags  []string
		want  string
	}{
		{name: "rank mismatch", wf: "FROM ./base\nLAYER x ./adapters/x\n", want: "do not fit",
			setup: func(t *testing.T, dir string) {
				buildtest.WriteAdapter(t, filepath.Join(dir, "adapters/x"), buildtest.Adapter{Rank: 4, Alpha: 8, Modules: []string{"self_attn.q_proj"}, Blocks: []int{0}, Config: map[string]any{"r": 8}})
			}},
		{name: "module not in base", wf: "FROM ./base\nLAYER x ./adapters/x\n", want: "the base has no tensor model.layers.5",
			setup: func(t *testing.T, dir string) {
				buildtest.WriteAdapter(t, filepath.Join(dir, "adapters/x"), buildtest.Adapter{Rank: 4, Alpha: 8, Modules: []string{"self_attn.q_proj"}, Blocks: []int{5}})
			}},
		{name: "dora", wf: "FROM ./base\nLAYER x ./adapters/x\n", want: "DoRA",
			setup: func(t *testing.T, dir string) {
				buildtest.WriteAdapter(t, filepath.Join(dir, "adapters/x"), buildtest.Adapter{Rank: 4, Alpha: 8, Modules: []string{"self_attn.q_proj"}, Blocks: []int{0}, Config: map[string]any{"use_dora": true}})
			}},
		{name: "not lora", wf: "FROM ./base\nLAYER x ./adapters/x\n", want: "peft_type",
			setup: func(t *testing.T, dir string) {
				buildtest.WriteAdapter(t, filepath.Join(dir, "adapters/x"), buildtest.Adapter{Rank: 4, Alpha: 8, Modules: []string{"self_attn.q_proj"}, Blocks: []int{0}, Config: map[string]any{"peft_type": "IA3"}})
			}},
		{name: "pickled adapter", wf: "FROM ./base\nLAYER x ./adapters/x\n", want: "safe_serialization",
			setup: func(t *testing.T, dir string) {
				buildtest.WriteAdapter(t, filepath.Join(dir, "adapters/x"), buildtest.Adapter{Rank: 4, Alpha: 8, Modules: []string{"self_attn.q_proj"}, Blocks: []int{0}})
				_ = os.Rename(filepath.Join(dir, "adapters/x/adapter_model.safetensors"), filepath.Join(dir, "adapters/x/adapter_model.bin"))
			}},
		{name: "pickled base", wf: "FROM ./pickled\n", want: "pickle weights",
			setup: func(t *testing.T, dir string) {
				_ = os.MkdirAll(filepath.Join(dir, "pickled"), 0o755)
				_ = os.WriteFile(filepath.Join(dir, "pickled/pytorch_model.bin"), []byte("x"), 0o644)
			}},
		{name: "no tokenizer", wf: "FROM ./notok\n", want: "no tokenizer.json or tokenizer.model",
			setup: func(t *testing.T, dir string) {
				buildtest.WriteBase(t, filepath.Join(dir, "notok"), 1)
				_ = os.Remove(filepath.Join(dir, "notok/tokenizer.json"))
			}},
		{name: "full fine-tune import", wf: "FROM ./base\nLAYER x ./adapters/french codec=xor\n", want: "lands in phase 2"},
		{name: "registry source", wf: "FROM ./base\nLAYER x oci://r.example.com/x:1\n", want: "only local adapter directories"},
		{name: "hf base", wf: "FROM hf://acme/llama\n", want: "only local directories"},
		{name: "router train", wf: "FROM ./base\nMIX m\n EXPERT a ./adapters/legal\n ROUTER train data=./d.jsonl\n", want: "phase 3"},
		{name: "missing template", wf: "FROM ./base\nTEMPLATE ./nope.jinja\n", want: "TEMPLATE"},
		{name: "bad tag", wf: "FROM ./base\n", tags: []string{"Bad Tag"}, want: "bad"},
		{name: "parse error", wf: "FROM ./base\nLAYER\n", want: "Weightfile"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newFixture(t)
			if c.setup != nil {
				c.setup(t, f.dir)
			}
			_, err := f.build(t, f.weightfile(t, "Weightfile", c.wf), c.tags...)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want it to contain %q", err, c.want)
			}
			if images, _ := f.store.Images(); len(images) != 0 {
				t.Errorf("a failed build tagged %d images", len(images))
			}
		})
	}
}

func TestBuildStopsWhenCancelled(t *testing.T) {
	f := newFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Build(ctx, f.store, Options{Weightfile: f.weightfile(t, "Weightfile", buildtest.Assistant)}, nil)
	if err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Fatalf("err = %v", err)
	}
}
