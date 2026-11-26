package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVersion(t *testing.T) {
	var out bytes.Buffer
	if code := run([]string{"version"}, &out, &out); code != 0 {
		t.Fatalf("exit %d", code)
	}
	if !strings.HasPrefix(out.String(), "wpk ") {
		t.Fatalf("output %q", out.String())
	}
}

func TestUnknownCommand(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run([]string{"nope"}, &out, &errOut); code != 2 {
		t.Fatalf("exit %d", code)
	}
}

func TestPlannedCommandsFailClearly(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run([]string{"diff"}, &out, &errOut); code != 3 {
		t.Fatalf("exit %d", code)
	}
	if !strings.Contains(errOut.String(), "not implemented") {
		t.Fatalf("stderr %q", errOut.String())
	}
}

func TestValidate(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.json")
	if err := os.WriteFile(good, []byte(`{
	  "schema": "weightpack.config.v1",
	  "arch": {"family": "llama", "fingerprint": "sha256:9f1c"},
	  "layers": [
	    {"id": "base", "kind": "base"},
	    {"id": "legal", "kind": "lora", "parent": "base", "rank": 16}
	  ]
	}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := run([]string{"validate", good}, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}

	bad := filepath.Join(dir, "bad.json")
	_ = os.WriteFile(bad, []byte(`{"schema": "weightpack.config.v1", "layers": []}`), 0o600)
	if code := run([]string{"validate", bad}, &out, &errOut); code != 1 {
		t.Fatalf("invalid config: exit %d", code)
	}
}

func TestValidateWeightfile(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "Weightfile"), []byte(`FROM ./base
LAYER french ./adapters/french
MIX domain targets=mlp top_k=1
  EXPERT legal ./adapters/legal
  EXPERT fin   hf://acme/fin codec=bit1 max_drop=0.5
`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := run([]string{"validate", dir}, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	for _, want := range []string{"valid", "MIX     domain", "targets=gate,up,down", "codec=bit1"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}

	bad := filepath.Join(dir, "Bad")
	_ = os.WriteFile(bad, []byte("FROM ./base\nLAYER A ./a\nEXPERT x ./x\n"), 0o600)
	out.Reset()
	errOut.Reset()
	if code := run([]string{"validate", bad}, &out, &errOut); code != 1 {
		t.Fatalf("invalid Weightfile: exit %d", code)
	}
	if got := strings.Count(errOut.String(), bad+":"); got != 2 {
		t.Errorf("want 2 located errors, got %d:\n%s", got, errOut.String())
	}
}
