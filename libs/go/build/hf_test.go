package build

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/hf"
	"github.com/rizkaitbrik/weightpack/libs/go/hf/hftest"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

func (f *fixture) buildHF(t *testing.T, hub *hftest.Hub, cache, name, body string, w *xorWorker) (*api.BuildResult, error) {
	t.Helper()
	p := f.weightfile(t, name, body)
	opts := Options{Weightfile: p, HF: &hf.Client{Endpoint: hub.URL, CacheDir: cache, HTTP: hub.Client()}}
	if w != nil {
		opts.Worker = w
	}
	return Build(context.Background(), f.store, opts, nil)
}

func TestHFSourcesMatchLocalBuilds(t *testing.T) {
	f := newFixture(t)
	hub := hftest.New(t)
	hub.AddDir(t, "acme/tiny", filepath.Join(f.dir, "base"))
	hub.AddDir(t, "acme/french", filepath.Join(f.dir, "adapters/french"))
	// Several adapters in folders of one repository, as some publish them.
	hub.AddDir(t, "acme/loras", filepath.Join(f.dir, "adapters"))
	sharded := filepath.Join(f.dir, "sharded")
	buildtest.WriteBase(t, sharded, 3)
	hub.AddDir(t, "acme/tiny-sharded", sharded)
	cache := t.TempDir()

	local, err := f.build(t, f.weightfile(t, "Local", "FROM ./base\nLAYER french ./adapters/french\nLAYER legal ./adapters/legal\n"))
	if err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"repos":   "FROM hf://acme/tiny\nLAYER french hf://acme/french\nLAYER legal hf://acme/loras/legal\n",
		"sharded": "FROM hf://acme/tiny-sharded@main\nLAYER french hf://acme/loras/french\nLAYER legal hf://acme/loras/legal\n",
	} {
		res, err := f.buildHF(t, hub, cache, "HF-"+name, body, nil)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if res.Manifest.Digest != local.Manifest.Digest {
			t.Errorf("%s: digest %s, the local build gives %s", name, res.Manifest.Digest, local.Manifest.Digest)
		}
	}
	// Building again downloads nothing: the cache is keyed by commit.
	before := hub.Downloads.Load()
	if _, err := f.buildHF(t, hub, cache, "Again", "FROM hf://acme/tiny\nLAYER french hf://acme/french\n", nil); err != nil {
		t.Fatal(err)
	}
	if got := hub.Downloads.Load(); got != before {
		t.Errorf("a rebuild downloaded %d files", got-before)
	}
	// Only the files a build needs come down: no README, no tokenizer
	// copies from an adapter repository.
	if _, err := os.Stat(filepath.Join(cache, "acme--french")); err != nil {
		t.Fatal(err)
	}
	entries, _ := filepath.Glob(filepath.Join(cache, "acme--loras", "*", "legal", "*"))
	if len(entries) != 2 {
		t.Errorf("legal folder holds %v, want just the adapter's config and weights", entries)
	}
}

func TestHFSourcesRefuse(t *testing.T) {
	f := newFixture(t)
	hub := hftest.New(t)
	hub.Add("acme/pickled", map[string][]byte{"pytorch_model.bin": []byte("x"), "config.json": []byte("{}")})
	hub.Add("acme/pickled-lora", map[string][]byte{"adapter_config.json": []byte("{}"), "adapter_model.bin": []byte("x")})
	hub.Add("acme/not-lora", map[string][]byte{"config.json": []byte("{}")})
	cases := map[string]struct{ body, want string }{
		"pickled model":   {"FROM hf://acme/pickled\n", "pickle weights"},
		"pickled adapter": {"FROM ./base\nLAYER x hf://acme/pickled-lora\n", "pickle weights"},
		"not an adapter":  {"FROM ./base\nLAYER x hf://acme/not-lora\n", "no adapter_config.json"},
		"missing repo":    {"FROM hf://acme/nothing\n", "not found"},
	}
	for name, c := range cases {
		if _, err := f.buildHF(t, hub, t.TempDir(), "Bad", c.body, nil); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
	}
}

func TestXORFromHF(t *testing.T) {
	f := newFixture(t)
	hub := hftest.New(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	hub.AddDir(t, "acme/finance", filepath.Join(f.dir, "finance"))
	local, err := f.buildXOR(t, "Local", "FROM ./base\nLAYER finance ./finance codec=xor\n", &xorWorker{})
	if err != nil {
		t.Fatal(err)
	}
	res, err := f.buildHF(t, hub, t.TempDir(), "HF", "FROM ./base\nLAYER finance hf://acme/finance codec=xor\n", &xorWorker{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Manifest.Digest != local.Manifest.Digest {
		t.Errorf("hf fine-tune gives %s, local gives %s", res.Manifest.Digest, local.Manifest.Digest)
	}
}

func writeSavedModuleAdapter(t *testing.T, dir string, shape []int64, save []string) {
	t.Helper()
	buildtest.WriteAdapter(t, dir, buildtest.Adapter{Rank: 4, Alpha: 8, Modules: buildtest.Attn, Blocks: []int{0}, Seed: 5,
		Config: map[string]any{"modules_to_save": save}})
	// Rewrite the weights with a saved lm_head copy beside the LoRA matrices.
	ts := []buildtest.Tensor{{Name: "base_model.model.lm_head.weight", Shape: shape, Seed: 77}}
	for _, m := range buildtest.Attn {
		p := "base_model.model.model.layers.0." + m + "."
		ts = append(ts,
			buildtest.Tensor{Name: p + "lora_A.weight", Shape: []int64{4, buildtest.Hidden}, Seed: 1},
			buildtest.Tensor{Name: p + "lora_B.weight", Shape: []int64{buildtest.Hidden, 4}, Seed: 2})
	}
	buildtest.WriteSafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), ts)
}

func TestModulesToSave(t *testing.T) {
	f := newFixture(t)
	writeSavedModuleAdapter(t, filepath.Join(f.dir, "adapters/head"), []int64{buildtest.Vocab, buildtest.Hidden}, []string{"lm_head"})
	res, err := f.build(t, f.weightfile(t, "W", "FROM ./base\nLAYER head ./adapters/head\n"))
	if err != nil {
		t.Fatal(err)
	}
	m, c := f.load(t, res.Manifest.Digest.String())
	if err := spec.CheckImage(m, c); err != nil {
		t.Fatal(err)
	}
	var lora, saved int
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] != "head" {
			continue
		}
		switch d.MediaType {
		case spec.MediaTypeLoRA:
			lora++
		case spec.MediaTypeTensor:
			saved++
			if d.Annotations[spec.AnnotationTensor] != "lm_head.weight" {
				t.Errorf("saved blob for %s", d.Annotations[spec.AnnotationTensor])
			}
		}
	}
	if lora != 2 || saved != 1 {
		t.Errorf("head layer has %d LoRA and %d saved blobs, want 2 and 1", lora, saved)
	}

	for name, c := range map[string]struct {
		shape []int64
		save  []string
	}{
		"not listed":  {[]int64{buildtest.Vocab, buildtest.Hidden}, nil},
		"wrong shape": {[]int64{buildtest.Vocab, 3}, []string{"lm_head"}},
	} {
		dir := filepath.Join(f.dir, "adapters", "bad-"+strings.ReplaceAll(name, " ", "-"))
		writeSavedModuleAdapter(t, dir, c.shape, c.save)
		if _, err := f.build(t, f.weightfile(t, "Bad", "FROM ./base\nLAYER x "+dir+"\n")); err == nil || !strings.Contains(err.Error(), "module_to_save copy") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}
