package export

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/safetensors"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

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

func (f *fixture) build(t *testing.T, weightfile string, tags ...string) string {
	t.Helper()
	res, err := build.Build(context.Background(), f.store, build.Options{Weightfile: weightfile, Tags: tags}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res.Manifest.Digest.String()
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

func TestExportRebuildsTheSameImage(t *testing.T) {
	for name, maxShard := range map[string]int64{"one file": 0, "sharded": 200_000} {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t)
			want := f.build(t, filepath.Join(f.dir, "Weightfile"), "assistant")
			out := filepath.Join(f.dir, "out")
			res, err := Export(context.Background(), f.store, "assistant", Options{Dir: out, MaxShardSize: maxShard}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if res.Dir != out || res.Files == 0 || res.Bytes == 0 {
				t.Errorf("result = %+v", res)
			}

			// Rebuilding from the export, into a fresh store, gives the
			// same manifest digest.
			s2, _ := store.Open(filepath.Join(f.dir, "store2"))
			rebuilt, err := build.Build(context.Background(), s2, build.Options{Weightfile: filepath.Join(out, "Weightfile")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := rebuilt.Manifest.Digest.String(); got != want {
				t.Errorf("rebuilt digest %s, want %s", got, want)
			}

			shards, _ := filepath.Glob(filepath.Join(out, "base", "model*.safetensors"))
			_, indexErr := os.Stat(filepath.Join(out, "base", "model.safetensors.index.json"))
			if maxShard == 0 && (len(shards) != 1 || indexErr == nil) {
				t.Errorf("one-file export: shards %v, index error %v", shards, indexErr)
			}
			if maxShard > 0 {
				var index struct {
					Metadata struct {
						TotalSize int64 `json:"total_size"`
					} `json:"metadata"`
					WeightMap map[string]string `json:"weight_map"`
				}
				readJSON(t, filepath.Join(out, "base", "model.safetensors.index.json"), &index)
				if len(shards) < 2 || len(index.WeightMap) != len(buildtest.BaseTensors()) || index.Metadata.TotalSize == 0 {
					t.Errorf("sharded export: %d shards, %d mapped tensors", len(shards), len(index.WeightMap))
				}
				for _, p := range shards {
					if info, _ := os.Stat(p); info.Size() > maxShard+100_000 {
						t.Errorf("%s is %d bytes, cap %d", p, info.Size(), maxShard)
					}
				}
			}
		})
	}
}

func TestExportLayout(t *testing.T) {
	f := newFixture(t)
	f.build(t, filepath.Join(f.dir, "Weightfile"), "assistant")
	out := filepath.Join(f.dir, "out")
	if err := os.Mkdir(out, 0o755); err != nil { // an empty directory is fine
		t.Fatal(err)
	}
	if _, err := Export(context.Background(), f.store, "assistant", Options{Dir: out}, nil); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{
		"Weightfile", "chat_template.jinja",
		"base/model.safetensors", "base/config.json", "base/tokenizer.json",
		"adapters/french/adapter_config.json", "adapters/french/adapter_model.safetensors",
		"adapters/legal/adapter_config.json", "adapters/medical/adapter_model.safetensors",
	} {
		if _, err := os.Stat(filepath.Join(out, p)); err != nil {
			t.Errorf("missing %s", p)
		}
	}
	wf, _ := os.ReadFile(filepath.Join(out, "Weightfile"))
	for _, line := range []string{
		"FROM ./base", "LAYER french ./adapters/french", "MIX domain targets=gate,up,down blocks=1 top_k=1",
		"EXPERT legal ./adapters/legal", "ROUTER arrow", "TEMPLATE ./chat_template.jinja", "LICENSE Apache-2.0",
	} {
		if !strings.Contains(string(wf), line) {
			t.Errorf("Weightfile lacks %q:\n%s", line, wf)
		}
	}

	// PEFT naming in the adapter file, and a config PEFT and vLLM accept.
	h, err := safetensors.Open(filepath.Join(out, "adapters/french/adapter_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if _, ok := h.Lookup("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"); !ok || h.Metadata["format"] != "pt" {
		t.Errorf("adapter tensors: %+v, metadata %v", h.Tensors, h.Metadata)
	}
	var french, legal map[string]any
	readJSON(t, filepath.Join(out, "adapters/french/adapter_config.json"), &french)
	readJSON(t, filepath.Join(out, "adapters/legal/adapter_config.json"), &legal)
	// task_type stays null: the image does not record the adapter's task,
	// and null loads with PEFT's generic wrapper on any model.
	if tt, set := french["task_type"]; !set || tt != nil {
		t.Errorf("french task_type = %v, want null", tt)
	}
	if french["peft_type"] != "LORA" || french["r"] != 8.0 || french["lora_alpha"] != 16.0 || french["use_rslora"] != false {
		t.Errorf("french config = %v", french)
	}
	// french covers q and v in every block: the suffix list is exact.
	if got, _ := french["target_modules"].([]any); !slices.Equal(got, []any{"q_proj", "v_proj"}) {
		t.Errorf("french target_modules = %v", french["target_modules"])
	}
	// legal covers block 1 only: a regex names its modules one by one.
	re, ok := legal["target_modules"].(string)
	if !ok {
		t.Fatalf("legal target_modules = %v, want a regex", legal["target_modules"])
	}
	rx := regexp.MustCompile(re)
	if !rx.MatchString("model.layers.1.mlp.up_proj") || rx.MatchString("model.layers.0.mlp.up_proj") || rx.MatchString("model.layers.1.mlp.up_proj.x") {
		t.Errorf("legal target_modules %q matches the wrong modules", re)
	}
}

// Annotations, such as the provenance wpk squash records, come back through
// LABEL lines, so the rebuilt manifest is the same.
func TestExportKeepsAnnotations(t *testing.T) {
	f := newFixture(t)
	wf := filepath.Join(f.dir, "Labelled.Weightfile")
	body := "FROM ./base\nLAYER french ./adapters/french\nLICENSE \"MIT OR Apache-2.0\"\n" +
		"LABEL wpk.squashed=french,legal wpk.squashed.from=sha256:" + strings.Repeat("ab", 32) + "\n" +
		"LABEL org.example.note=\"a \\\"quoted\\\" #hash and \\\\ slash\" org.example.empty=\n"
	if err := os.WriteFile(wf, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	want := f.build(t, wf, "labelled")
	out := filepath.Join(f.dir, "out")
	if _, err := Export(context.Background(), f.store, "labelled", Options{Dir: out}, nil); err != nil {
		t.Fatal(err)
	}
	if got := f.build(t, filepath.Join(out, "Weightfile")); got != want {
		exported, _ := os.ReadFile(filepath.Join(out, "Weightfile"))
		t.Errorf("rebuilt %s, want %s; exported Weightfile:\n%s", got, want, exported)
	}
}

func TestQuote(t *testing.T) {
	for in, want := range map[string]string{
		"MIT": "MIT", "a b": `"a b"`, `say "hi"`: `"say \"hi\""`, `c:\x`: `"c:\\x"`, "#1": `"#1"`, "": "",
	} {
		if got, err := quote(in); err != nil || got != want {
			t.Errorf("quote(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	if _, err := quote("two\nlines"); err == nil {
		t.Error("a line break was accepted")
	}
}

func TestExportBaseOnlyImage(t *testing.T) {
	f := newFixture(t)
	wf := filepath.Join(f.dir, "Base.Weightfile")
	if err := os.WriteFile(wf, []byte("FROM ./base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := f.build(t, wf, "plain")
	out := filepath.Join(f.dir, "out")
	if _, err := Export(context.Background(), f.store, "plain", Options{Dir: out}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(out, "adapters")); !os.IsNotExist(err) {
		t.Error("a base-only image exported an adapters directory")
	}
	if got := f.build(t, filepath.Join(out, "Weightfile")); got != want {
		t.Errorf("rebuilt %s, want %s", got, want)
	}
}

func TestExportRefuses(t *testing.T) {
	f := newFixture(t)
	f.build(t, filepath.Join(f.dir, "Weightfile"), "assistant")
	full := filepath.Join(f.dir, "full")
	_ = os.MkdirAll(full, 0o755)
	_ = os.WriteFile(filepath.Join(full, "keep.txt"), []byte("mine"), 0o644)
	cases := map[string]Options{
		"relative":  {Dir: "out"},
		"not empty": {Dir: full},
		"no parent": {Dir: filepath.Join(f.dir, "missing", "out")},
	}
	for name, opts := range cases {
		if _, err := Export(context.Background(), f.store, "assistant", opts, nil); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if _, err := Export(context.Background(), f.store, "nothing", Options{Dir: filepath.Join(f.dir, "x")}, nil); err == nil {
		t.Error("unknown image: no error")
	}
	if b, _ := os.ReadFile(filepath.Join(full, "keep.txt")); string(b) != "mine" {
		t.Error("export touched a non-empty directory")
	}
	// A cancelled export leaves no partial directory behind.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Export(ctx, f.store, "assistant", Options{Dir: filepath.Join(f.dir, "cancelled")}, nil); err == nil {
		t.Error("cancelled export: no error")
	}
	if leftovers, _ := filepath.Glob(filepath.Join(f.dir, ".*partial*")); len(leftovers) != 0 {
		t.Errorf("left behind %v", leftovers)
	}
	if _, err := os.Stat(filepath.Join(f.dir, "cancelled")); !os.IsNotExist(err) {
		t.Error("cancelled export created its directory")
	}
}

func TestPartialExport(t *testing.T) {
	f := newFixture(t)
	f.build(t, filepath.Join(f.dir, "Weightfile"), "assistant")

	// Only one adapter: no base, no other adapter, no Weightfile.
	out := filepath.Join(f.dir, "legal")
	if _, err := Export(context.Background(), f.store, "assistant", Options{Dir: out, NoBase: true, Adapters: []string{"legal"}}, nil); err != nil {
		t.Fatal(err)
	}
	for p, want := range map[string]bool{
		"adapters/legal/adapter_config.json": true, "adapters/legal/adapter_model.safetensors": true,
		"adapters/french": false, "base": false, "Weightfile": false,
	} {
		if _, err := os.Stat(filepath.Join(out, p)); (err == nil) != want {
			t.Errorf("%s: exists %v, want %v", p, err == nil, want)
		}
	}
	// The adapter is described against the base all the same.
	full := filepath.Join(f.dir, "full")
	if _, err := Export(context.Background(), f.store, "assistant", Options{Dir: full}, nil); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"adapter_config.json", "adapter_model.safetensors"} {
		a, _ := os.ReadFile(filepath.Join(out, "adapters/legal", name))
		b, _ := os.ReadFile(filepath.Join(full, "adapters/legal", name))
		if string(a) != string(b) || len(a) == 0 {
			t.Errorf("partial %s differs from the full export's", name)
		}
	}

	// Only the base: an empty adapter list.
	base := filepath.Join(f.dir, "base-only")
	if _, err := Export(context.Background(), f.store, "assistant", Options{Dir: base, Adapters: []string{}}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(base, "base", "model.safetensors")); err != nil {
		t.Error("base missing")
	}
	if _, err := os.Stat(filepath.Join(base, "adapters")); !os.IsNotExist(err) {
		t.Error("an empty adapter list exported adapters")
	}
	if _, err := os.Stat(filepath.Join(base, "chat_template.jinja")); err != nil {
		t.Error("the chat template is missing")
	}

	if _, err := Export(context.Background(), f.store, "assistant", Options{Dir: filepath.Join(f.dir, "x"), Adapters: []string{"nope"}}, nil); err == nil || !strings.Contains(err.Error(), `no LoRA layer "nope"`) {
		t.Errorf("unknown adapter: %v", err)
	}
}
