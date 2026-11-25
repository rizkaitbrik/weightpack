package main

import (
	"bytes"
	"encoding/json"
	"net"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/daemon"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

// startDaemon serves the daemon API on a Unix socket and points wpk at it.
func startDaemon(t *testing.T) {
	t.Helper()
	dir, err := os.MkdirTemp("", "wpk") // short path: socket paths are limited
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	s, err := store.Open(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	socket := filepath.Join(dir, "s")
	ln, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(daemon.NewHandler(s, "test", nil))
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)
	t.Setenv("WPK_SOCKET", socket)
}

func TestBuildImagesInspectHistory(t *testing.T) {
	startDaemon(t)
	dir := t.TempDir()
	buildtest.Project(t, dir)

	var out, errOut bytes.Buffer
	// Flags may follow the directory, as with docker build.
	if code := run([]string{"build", dir, "-t", "assistant:v1"}, &out, &errOut); code != 0 {
		t.Fatalf("build: exit %d\n%s", code, errOut.String())
	}
	for _, want := range []string{"FROM ./base", "LAYER french", "EXPERT legal", "MIX domain", "Built sha256:", "Tagged assistant:v1"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("build output lacks %q:\n%s", want, out.String())
		}
	}
	if !strings.Contains(errOut.String(), "warning: EXPERT medical:") {
		t.Errorf("build stderr lacks the medical warning:\n%s", errOut.String())
	}

	out.Reset()
	if code := run([]string{"images"}, &out, &errOut); code != 0 {
		t.Fatalf("images: exit %d", code)
	}
	if lines := strings.Split(strings.TrimSpace(out.String()), "\n"); len(lines) != 2 || !strings.HasPrefix(lines[1], "assistant:v1") {
		t.Errorf("images output:\n%s", out.String())
	}

	out.Reset()
	if code := run([]string{"inspect", "assistant:v1"}, &out, &errOut); code != 0 {
		t.Fatalf("inspect: exit %d", code)
	}
	var detail api.ImageDetail
	if err := json.Unmarshal(out.Bytes(), &detail); err != nil || detail.Config.Arch.Family != "llama" {
		t.Errorf("inspect output: %v\n%s", err, out.String())
	}

	out.Reset()
	if code := run([]string{"history", "assistant:v1"}, &out, &errOut); code != 0 {
		t.Fatalf("history: exit %d", code)
	}
	for _, want := range []string{"french", "rank 8, alpha 16, q,v", "top_k 1 of legal,medical, router arrow, blocks 1", "image-level files"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("history lacks %q:\n%s", want, out.String())
		}
	}
}

func TestBuildReportsErrors(t *testing.T) {
	startDaemon(t)
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "Weightfile"), []byte("FROM ./base\nLAYER x ./adapters/x codec=bit1 max_drop=1\n"), 0o644)
	var out, errOut bytes.Buffer
	if code := run([]string{"build", dir}, &out, &errOut); code != 1 {
		t.Fatalf("exit %d", code)
	}
	if !strings.Contains(errOut.String(), "codec=bit1 runs in the build worker") {
		t.Errorf("stderr %q", errOut.String())
	}
	if code := run([]string{"build", "a", "b"}, &out, &errOut); code != 2 {
		t.Errorf("two paths: exit %d", code)
	}
}

func TestDaemonCommandsWithoutDaemon(t *testing.T) {
	t.Setenv("WPK_SOCKET", filepath.Join(t.TempDir(), "absent.sock"))
	for _, args := range [][]string{{"images"}, {"inspect", "x"}, {"history", "x"}, {"build", t.TempDir()}} {
		var out, errOut bytes.Buffer
		if code := run(args, &out, &errOut); code != 1 || !strings.Contains(errOut.String(), "is the daemon running") {
			t.Errorf("%v: exit %d, stderr %q", args, code, errOut.String())
		}
	}
}

func TestLayerDetailOfDeltas(t *testing.T) {
	drop := 0.42
	for _, c := range []struct {
		l    spec.Layer
		want string
	}{
		{spec.Layer{Kind: spec.KindDeltaXOR}, "exact"},
		{spec.Layer{Kind: spec.KindDeltaBit1, Fidelity: &spec.Fidelity{Codec: "bit1", Eval: "self", MaxDrop: 1, MeasuredDrop: &drop}}, "bit1, drop 0.42 points on self (max 1)"},
		{spec.Layer{Kind: spec.KindDeltaSparse, Fidelity: &spec.Fidelity{Codec: "sparse", Density: 0.1, Eval: "finance", MaxDrop: 2, MeasuredDrop: &drop}}, "sparse 10%, drop 0.42 points on finance (max 2)"},
		{spec.Layer{Kind: spec.KindDeltaBit1}, "lossy, no measured drop"},
	} {
		if got := layerDetail(c.l, spec.Arch{}); got != c.want {
			t.Errorf("%s: %q, want %q", c.l.Kind, got, c.want)
		}
	}
}
