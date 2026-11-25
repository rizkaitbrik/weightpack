package main

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
)

func TestRmAndPrune(t *testing.T) {
	startDaemon(t)
	dir := t.TempDir()
	buildtest.Project(t, dir)
	_ = os.WriteFile(filepath.Join(dir, "French.Weightfile"), []byte("FROM ./base\nLAYER french ./adapters/french\n"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "Legal.Weightfile"), []byte("FROM ./base\nLAYER legal ./adapters/legal\n"), 0o644)
	var out, errOut bytes.Buffer
	for _, args := range [][]string{
		{"build", "-t", "french", filepath.Join(dir, "French.Weightfile")},
		{"build", "-t", "legal", filepath.Join(dir, "Legal.Weightfile")},
		{"rm", "legal"},
	} {
		if code := run(args, &out, &errOut); code != 0 {
			t.Fatalf("%v: exit %d\n%s", args, code, errOut.String())
		}
	}
	if !strings.Contains(out.String(), "Untagged legal") {
		t.Errorf("rm output: %s", out.String())
	}

	removed := regexp.MustCompile(`remove (\d+) blobs`)
	out.Reset()
	if code := run([]string{"prune", "--dry-run"}, &out, &errOut); code != 0 {
		t.Fatalf("dry run: %s", errOut.String())
	}
	dry := removed.FindStringSubmatch(out.String())
	// legal's 3 LoRA modules, its config and its manifest; the base is shared.
	if dry == nil || dry[1] != "5" {
		t.Errorf("dry run: %s", out.String())
	}
	out.Reset()
	if code := run([]string{"prune"}, &out, &errOut); code != 0 || !strings.Contains(out.String(), "Removed 5 blobs") {
		t.Fatalf("prune: exit %d, %s %s", code, out.String(), errOut.String())
	}
	out.Reset()
	if code := run([]string{"history", "french"}, &out, &errOut); code != 0 {
		t.Fatalf("the kept image is broken after prune: %s", errOut.String())
	}
	// Nothing left to prune.
	out.Reset()
	if code := run([]string{"prune"}, &out, &errOut); code != 0 || !strings.Contains(out.String(), "Removed 0 blobs") {
		t.Errorf("second prune: %s", out.String())
	}
	if code := run([]string{"rm", "nothing"}, &out, &errOut); code != 1 {
		t.Errorf("rm of a missing image: exit %d", code)
	}
}
