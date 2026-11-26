package weightfile

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

const digest = "sha256:b45e6a1f0c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f"

func mustParse(t *testing.T, src string) *File {
	t.Helper()
	f, err := Parse(strings.NewReader(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	return f
}

func TestDesignExample(t *testing.T) {
	path := filepath.Join("testdata", "design.Weightfile")
	f, err := ParseFile(path)
	if err != nil {
		t.Fatal(err)
	}

	if f.From.Source.Kind != SourceOCI || f.From.Line != 2 {
		t.Errorf("FROM = %+v", f.From)
	}
	if len(f.Layers) != 1 || f.Layers[0].ID != "french" || f.Layers[0].Source.Kind != SourceLocal {
		t.Errorf("layers = %+v", f.Layers)
	}
	m := f.Mix
	if m == nil || m.ID != "domain" || len(m.Experts) != 3 || m.Router.Kind != "arrow" || m.Router.Line != 12 {
		t.Fatalf("mix = %+v", m)
	}
	if kinds := []SourceKind{m.Experts[0].Source.Kind, m.Experts[1].Source.Kind, m.Experts[2].Source.Kind}; !slices.Equal(kinds, []SourceKind{SourceLocal, SourceOCI, SourceHF}) {
		t.Errorf("expert source kinds = %v", kinds)
	}
	if f.Template == nil || f.Template.Value != "./chat.jinja" || f.License == nil || f.License.Value != "inherit" {
		t.Errorf("meta = %+v %+v", f.Template, f.License)
	}

	cfg, err := f.Config()
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.MarshalIndent(cfg, "", "  ")
	want, err := os.ReadFile(filepath.Join("testdata", "design.config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bytes.TrimSpace(got), bytes.TrimSpace(want)) {
		t.Errorf("config mismatch\n got: %s\nwant: %s", got, want)
	}
}

func TestStackOnly(t *testing.T) {
	f := mustParse(t, `
FROM ./base
LAYER a ./a
LAYER b ./b
LAYER c ./c
`)
	cfg, err := f.Config()
	if err != nil {
		t.Fatal(err)
	}
	var chain []string
	for _, l := range cfg.Layers {
		chain = append(chain, l.ID+"<"+l.Parent)
	}
	if want := []string{"base<", "a<base", "b<a", "c<b"}; !slices.Equal(chain, want) {
		t.Errorf("chain = %v, want %v", chain, want)
	}
}

func TestMixDefaults(t *testing.T) {
	f := mustParse(t, `
FROM ./base
MIX m
  EXPERT x ./x
  EXPERT y ./y
  EXPERT z ./z
`)
	m := f.Mix
	if !slices.Equal(m.Targets, modules) || m.Blocks != "" || m.TopK != 2 || m.Router.Kind != "arrow" {
		t.Errorf("defaults = %+v", m)
	}
	one := mustParse(t, "FROM ./base\nMIX m\n EXPERT x ./x\n")
	if one.Mix.TopK != 1 {
		t.Errorf("top_k default with one expert = %d", one.Mix.TopK)
	}
	cfg, err := f.Config()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Layers[1].Kind != spec.KindMix || cfg.Layers[2].Parent != "base" {
		t.Errorf("mix directly on base: %+v", cfg.Layers)
	}
}

func TestRouterTrain(t *testing.T) {
	f := mustParse(t, "FROM ./b\nMIX m\nEXPERT x ./x\nEXPERT y ./y\nROUTER train data=./mix.jsonl\n")
	if r := f.Mix.Router; r.Kind != "train" || r.Data != "./mix.jsonl" || r.Steps != 2000 {
		t.Errorf("router = %+v", r)
	}
	f = mustParse(t, "FROM ./b\nMIX m\nEXPERT x ./x\nROUTER TRAIN data=d steps=500\n")
	if r := f.Mix.Router; r.Kind != "train" || r.Steps != 500 {
		t.Errorf("router = %+v", r)
	}
	cfg, err := f.Config()
	if err != nil || cfg.Layers[1].Router != spec.RouterTrained {
		t.Errorf("config router = %q (%v)", cfg.Layers[1].Router, err)
	}
}

func TestRouterFile(t *testing.T) {
	f := mustParse(t, "FROM ./b\nMIX m\nEXPERT x ./x\nROUTER ./routers/m.safetensors\n")
	if r := f.Mix.Router; r.Kind != "file" || r.Path != "./routers/m.safetensors" || !r.Trained() {
		t.Errorf("router = %+v", r)
	}
	cfg, err := f.Config()
	if err != nil || cfg.Layers[1].Router != spec.RouterTrained {
		t.Errorf("config router = %q (%v)", cfg.Layers[1].Router, err)
	}
	if f := mustParse(t, "FROM ./b\nMIX m\nEXPERT x ./x\n"); f.Mix.Router.Trained() {
		t.Error("arrow counted as trained")
	}
	for _, bad := range []string{"ROUTER ./r.safetensors steps=3", "ROUTER r.safetensors"} {
		if _, err := Parse(strings.NewReader("FROM ./b\nMIX m\nEXPERT x ./x\n" + bad + "\n")); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

func TestTargetsAndBlocks(t *testing.T) {
	f := mustParse(t, "FROM ./b\nMIX m targets=down,attn blocks=0-3,8-31,40\n EXPERT x ./x\n")
	if want := []string{"q", "k", "v", "o", "down"}; !slices.Equal(f.Mix.Targets, want) {
		t.Errorf("targets = %v, want %v", f.Mix.Targets, want)
	}
	if f.Mix.Blocks != "0-3,8-31,40" {
		t.Errorf("blocks = %q", f.Mix.Blocks)
	}
	f = mustParse(t, "FROM ./b\nMIX m blocks=ALL\n EXPERT x ./x\n")
	if f.Mix.Blocks != "" {
		t.Errorf("blocks=all = %q", f.Mix.Blocks)
	}
}

func TestLexing(t *testing.T) {
	f := mustParse(t, "\ufeff# header comment\n"+
		"from ./base   # trailing comment\n"+
		"\n"+
		"LAYER spaced \"./my adapters/a\" \\\n"+
		"      codec=xor\n")
	if f.From.Line != 2 {
		t.Errorf("FROM line = %d", f.From.Line)
	}
	if l := f.Layers[0]; l.Line != 4 || l.Source.Ref != "./my adapters/a" || l.Codec != CodecXOR {
		t.Errorf("continued layer = %+v", l)
	}
	lines, errs := lex(strings.NewReader("LAYER \"a b\" \"./x \\\"y\\\"\" \\\n codec=xor\n"))
	if len(errs) != 0 || len(lines) != 1 {
		t.Fatalf("lex = %v %v", lines, errs)
	}
	if want := []string{"LAYER", "a b", `./x "y"`, "codec=xor"}; !slices.Equal(lines[0].fields, want) {
		t.Errorf("fields = %q, want %q", lines[0].fields, want)
	}
}

func TestHashInsideFieldIsNotAComment(t *testing.T) {
	f := mustParse(t, "FROM ./base\nLAYER a ./dir#1\n")
	if f.Layers[0].Source.Ref != "./dir#1" {
		t.Errorf("source = %q", f.Layers[0].Source.Ref)
	}
}

func TestErrors(t *testing.T) {
	cases := []struct {
		name, src string
		line      int
		want      string
	}{
		{"empty", "", 1, "missing FROM"},
		{"only comments", "# nothing\n", 1, "missing FROM"},
		{"layer before from", "LAYER a ./a\nFROM ./b\n", 1, "before FROM"},
		{"two froms", "FROM ./a\nFROM ./b\n", 2, "second FROM"},
		{"from usage", "FROM\n", 1, "usage: FROM"},
		{"unknown instruction", "FROM ./b\nCOPY a b\n", 2, `unknown instruction "COPY"`},
		{"train", "FROM ./b\nLAYER a TRAIN\n", 2, "TRAIN blocks are not supported"},
		{"reserved id", "FROM ./b\nLAYER base ./x\n", 2, "reserved"},
		{"bad id", "FROM ./b\nLAYER Fr ./x\n", 2, "bad layer id"},
		{"duplicate id", "FROM ./b\nLAYER a ./x\nMIX a\n EXPERT b ./y\n", 3, `already used at line 2`},
		{"bad source", "FROM ./b\nLAYER a ftp://x\n", 2, "unknown scheme"},
		{"tilde", "FROM ~/models/base\n", 1, "not expanded"},
		{"bad codec", "FROM ./b\nLAYER a ./x codec=zip\n", 2, "unknown codec"},
		{"lossy without max_drop", "FROM ./b\nLAYER a hf://o/m codec=bit1\n", 2, "needs max_drop"},
		{"max_drop on lossless", "FROM ./b\nLAYER a hf://o/m codec=xor max_drop=1\n", 2, "only to lossy"},
		{"bad max_drop", "FROM ./b\nLAYER a hf://o/m codec=bit1 max_drop=-1\n", 2, "max_drop must be"},
		{"unknown param", "FROM ./b\nLAYER a ./x rank=8\n", 2, `unknown parameter "rank"`},
		{"positional junk", "FROM ./b\nLAYER a ./x extra\n", 2, "want key=value"},
		{"dup param", "FROM ./b\nMIX m top_k=1 top_k=2\n EXPERT x ./x\n", 2, "given twice"},
		{"expert outside mix", "FROM ./b\nEXPERT x ./x\n", 2, "outside a MIX"},
		{"expert after template", "FROM ./b\nMIX m\n EXPERT x ./x\nTEMPLATE ./t\n EXPERT y ./y\n", 5, "outside a MIX"},
		{"router outside mix", "FROM ./b\nROUTER arrow\n", 2, "outside a MIX"},
		{"two routers", "FROM ./b\nMIX m\n EXPERT x ./x\n ROUTER arrow\n ROUTER arrow\n", 5, "second ROUTER"},
		{"unknown router", "FROM ./b\nMIX m\n EXPERT x ./x\n ROUTER gumbel\n", 4, "unknown router"},
		{"router train no data", "FROM ./b\nMIX m\n EXPERT x ./x\n ROUTER train\n", 4, "needs data"},
		{"empty mix", "FROM ./b\nMIX m\nTEMPLATE ./t\n", 2, "no EXPERT"},
		{"top_k too big", "FROM ./b\nMIX m top_k=3\n EXPERT x ./x\n EXPERT y ./y\n", 2, "top_k=3 but only 2"},
		{"top_k zero", "FROM ./b\nMIX m top_k=0\n EXPERT x ./x\n", 2, "top_k must be"},
		{"layer after mix", "FROM ./b\nMIX m\n EXPERT x ./x\nLAYER a ./a\n", 4, "LAYER after MIX"},
		{"two mixes", "FROM ./b\nMIX m\n EXPERT x ./x\nMIX n\n EXPERT y ./y\n", 4, "second MIX"},
		{"bad target", "FROM ./b\nMIX m targets=ffn\n EXPERT x ./x\n", 2, "unknown target"},
		{"bad blocks", "FROM ./b\nMIX m blocks=8-\n EXPERT x ./x\n", 2, "bad blocks range"},
		{"overlapping blocks", "FROM ./b\nMIX m blocks=0-8,8-9\n EXPERT x ./x\n", 2, "ascend without overlap"},
		{"two templates", "FROM ./b\nTEMPLATE ./a\nTEMPLATE ./b\n", 3, "second TEMPLATE"},
		{"license usage", "FROM ./b\nLICENSE\n", 2, "usage: LICENSE"},
		{"unterminated quote", "FROM \"./b\n", 1, "unterminated quote"},
		{"dangling continuation", "FROM ./b\nLAYER a ./a \\\n", 2, "continued line"},
		{"oci bad digest", "FROM registry.example.com/llama@sha256:b45e\n", 1, "digest must be"},
		{"oci uppercase", "FROM Llama/Base\n", 1, "bad path component"},
		{"hf no org", "FROM hf://llama\n", 1, "want hf://<org>/<name>"},
		{"label usage", "FROM ./b\nLABEL\n", 2, "usage: LABEL"},
		{"label no value", "FROM ./b\nLABEL team\n", 2, "want key=value"},
		{"label bad key", "FROM ./b\nLABEL =x\n", 2, "want key=value"},
		{"label twice", "FROM ./b\nLABEL a=1\nLABEL a=2\n", 3, "already set at line 2"},
		{"label license", "FROM ./b\nLABEL org.opencontainers.image.licenses=MIT\n", 2, "set the license with LICENSE"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(strings.NewReader(c.src))
			if err == nil {
				t.Fatalf("no error, want %q", c.want)
			}
			var we *Error
			if !errors.As(err, &we) {
				t.Fatalf("error %v is not an *Error", err)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error = %q, want it to contain %q", err, c.want)
			}
			if we.Line != c.line {
				t.Errorf("first error at line %d, want %d (%v)", we.Line, c.line, err)
			}
		})
	}
}

func TestLabels(t *testing.T) {
	f, err := Parse(strings.NewReader("FROM ./b\nLABEL org.example.team=nlp \"note=two words\" empty=\nMIX m\n EXPERT x ./x\nLABEL wpk.squashed.from=sha256:abc\n"))
	if err != nil {
		t.Fatal(err)
	}
	want := []Label{
		{Line: 2, Key: "org.example.team", Value: "nlp"},
		{Line: 2, Key: "note", Value: "two words"},
		{Line: 2, Key: "empty", Value: ""},
		{Line: 5, Key: "wpk.squashed.from", Value: "sha256:abc"},
	}
	if !reflect.DeepEqual(f.Labels, want) {
		t.Errorf("labels = %+v", f.Labels)
	}
}

func TestAllErrorsReportedInLineOrder(t *testing.T) {
	_, err := Parse(strings.NewReader("FROM ./b\nLAYER A ./a\nLAYER b ./b codec=nope\nBOGUS\n"))
	if err == nil {
		t.Fatal("no error")
	}
	msg := err.Error()
	var lines []string
	for _, l := range strings.Split(msg, "\n") {
		lines = append(lines, strings.SplitN(l, ":", 3)[1])
	}
	if want := []string{"2", "3", "4"}; !slices.Equal(lines, want) {
		t.Errorf("error lines = %v, want %v\n%s", lines, want, msg)
	}
}

func TestParseFileNamesTheFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "Weightfile")
	if err := os.WriteFile(path, []byte("FROM ./b\nLAYER A ./a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := ParseFile(path)
	if err == nil || !strings.HasPrefix(err.Error(), path+":2: ") {
		t.Fatalf("error = %v", err)
	}
}

func TestSourcePath(t *testing.T) {
	rel := Source{Kind: SourceLocal, Ref: "./adapters/a"}
	if got := rel.Path("/work"); got != "/work/adapters/a" {
		t.Errorf("relative = %q", got)
	}
	abs := Source{Kind: SourceLocal, Ref: "/models/base"}
	if got := abs.Path("/work"); got != "/models/base" {
		t.Errorf("absolute = %q", got)
	}
	if got := (Source{Kind: SourceHF, Ref: "o/m"}).Path("/work"); got != "" {
		t.Errorf("hf path = %q", got)
	}
}

func TestParseSource(t *testing.T) {
	ok := map[string]Source{
		".":                                    {SourceLocal, "."},
		"../x":                                 {SourceLocal, "../x"},
		"/abs/x":                               {SourceLocal, "/abs/x"},
		"hf://meta-llama/Llama-3.1-8B":         {SourceHF, "meta-llama/Llama-3.1-8B"},
		"hf://acme/fin@3e2a":                   {SourceHF, "acme/fin@3e2a"},
		"oci://ghcr.io/acme/medical:1.4":       {SourceOCI, "ghcr.io/acme/medical:1.4"},
		"llama":                                {SourceOCI, "llama"},
		"localhost:5000/llama:v1":              {SourceOCI, "localhost:5000/llama:v1"},
		"registry.example.com/llama@" + digest: {SourceOCI, "registry.example.com/llama@" + digest},
		"r.io/a/b:t@" + digest:                 {SourceOCI, "r.io/a/b:t@" + digest},
	}
	for in, want := range ok {
		got, err := ParseSource(in)
		if err != nil || got != want {
			t.Errorf("ParseSource(%q) = %+v, %v; want %+v", in, got, err, want)
		}
		if again, err := ParseSource(got.String()); err != nil || again != got {
			t.Errorf("round trip of %q: %+v, %v", in, again, err)
		}
	}
	for _, bad := range []string{"", "s3://b/k", "~/x", "hf://", "hf://only", "oci://", "UPPER", "r.io/a:bad tag", "a//b", "r.io:port/x"} {
		if _, err := ParseSource(bad); err == nil {
			t.Errorf("ParseSource(%q) accepted", bad)
		}
	}
}

func TestHFSources(t *testing.T) {
	for in, want := range map[string][3]string{
		"hf://Qwen/Qwen2.5-0.5B-Instruct":                      {"Qwen/Qwen2.5-0.5B-Instruct", "", "main"},
		"hf://acme/loras/adapters/legal@v2":                    {"acme/loras", "adapters/legal", "v2"},
		"hf://acme/m@3e2a1f0c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a": {"acme/m", "", "3e2a1f0c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a"},
	} {
		s, err := ParseSource(in)
		if err != nil {
			t.Errorf("%s: %v", in, err)
			continue
		}
		repo, folder, rev := s.HF()
		if [3]string{repo, folder, rev} != want {
			t.Errorf("%s: HF() = %q %q %q, want %v", in, repo, folder, rev, want)
		}
	}
	for _, bad := range []string{"hf://acme/m/../x", "hf://acme/m/./x", "hf://acme"} {
		if _, err := ParseSource(bad); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

func TestDensity(t *testing.T) {
	f := mustParse(t, "FROM ./b\nLAYER x ./x codec=sparse density=0.25 max_drop=1\nLAYER y ./y codec=sparse max_drop=1\n")
	if d := f.Layers[0].Density; d == nil || *d != 0.25 || f.Layers[1].Density != nil {
		t.Errorf("densities %v %v", f.Layers[0].Density, f.Layers[1].Density)
	}
	cfg, err := f.Config()
	if err != nil || cfg.Layers[1].Fidelity.Density != 0.25 || cfg.Layers[2].Fidelity.Density != DefaultDensity {
		t.Errorf("config: %v %+v", err, cfg.Layers)
	}
	for _, bad := range []string{
		"LAYER x ./x codec=bit1 density=0.2 max_drop=1",
		"LAYER x ./x codec=sparse density=0 max_drop=1",
		"LAYER x ./x codec=sparse density=1.5 max_drop=1",
		"LAYER x ./x density=0.2",
	} {
		if _, err := Parse(strings.NewReader("FROM ./b\n" + bad + "\n")); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

func TestCalibrate(t *testing.T) {
	f := mustParse(t, "FROM ./b\nLAYER x ./x codec=bit1 max_drop=1\nLAYER y ./y codec=bit1 calibrate=0 max_drop=1\nLAYER z ./z codec=sparse max_drop=1\n")
	if c := f.Layers[1].Calibrate; f.Layers[0].Calibrate != nil || c == nil || *c != 0 {
		t.Errorf("calibrate %v %v", f.Layers[0].Calibrate, c)
	}
	cfg, err := f.Config()
	if err != nil || cfg.Layers[1].Fidelity.CalibrationSteps != DefaultCalibrate ||
		cfg.Layers[2].Fidelity.CalibrationSteps != 0 || cfg.Layers[3].Fidelity.CalibrationSteps != 0 {
		t.Errorf("config: %v %+v", err, cfg.Layers)
	}
	for _, bad := range []string{
		"LAYER x ./x codec=sparse calibrate=10 max_drop=1",
		"LAYER x ./x codec=xor calibrate=10",
		"LAYER x ./x codec=bit1 calibrate=-1 max_drop=1",
		"LAYER x ./x codec=bit1 calibrate=1.5 max_drop=1",
		"LAYER x ./x codec=bit1 calibrate=1000000 max_drop=1",
	} {
		if _, err := Parse(strings.NewReader("FROM ./b\n" + bad + "\n")); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}
