package main

import (
	"bytes"
	"path/filepath"
	"regexp"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
)

func TestExportThenRebuild(t *testing.T) {
	startDaemon(t)
	dir := t.TempDir()
	buildtest.Project(t, dir)
	digest := regexp.MustCompile(`Built (sha256:[0-9a-f]{64})`)

	var out, errOut bytes.Buffer
	if code := run([]string{"build", "-t", "assistant", dir}, &out, &errOut); code != 0 {
		t.Fatalf("build: exit %d\n%s", code, errOut.String())
	}
	first := digest.FindStringSubmatch(out.String())
	exported := filepath.Join(t.TempDir(), "assistant")
	out.Reset()
	if code := run([]string{"export", "-o", exported, "--max-shard-size", "200KB", "assistant"}, &out, &errOut); code != 0 {
		t.Fatalf("export: exit %d\n%s", code, errOut.String())
	}
	out.Reset()
	if code := run([]string{"build", exported}, &out, &errOut); code != 0 {
		t.Fatalf("rebuild: exit %d\n%s", code, errOut.String())
	}
	second := digest.FindStringSubmatch(out.String())
	if first == nil || second == nil || first[1] != second[1] {
		t.Errorf("rebuilt %v, built %v", second, first)
	}

	if code := run([]string{"export", "assistant"}, &out, &errOut); code != 2 {
		t.Errorf("export without -o: exit %d", code)
	}
	if code := run([]string{"export", "-o", exported, "assistant"}, &out, &errOut); code != 1 {
		t.Errorf("export into a non-empty directory: exit %d", code)
	}
}

func TestParseSize(t *testing.T) {
	for in, want := range map[string]int64{"5GB": 5e9, "500MB": 5e8, "64kb": 64e3, "1000": 1000, "1.5 GB": 15e8} {
		if got, err := parseSize(in); err != nil || got != want {
			t.Errorf("parseSize(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "GB", "-1MB", "five"} {
		if _, err := parseSize(bad); err == nil {
			t.Errorf("parseSize(%q) accepted", bad)
		}
	}
}
