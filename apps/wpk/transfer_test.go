package main

import (
	"bytes"
	"io"
	"log"
	"net/http/httptest"
	"strings"
	"testing"

	ggcr "github.com/google/go-containerregistry/pkg/registry"

	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
)

func TestTagPushPull(t *testing.T) {
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	reg := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(reg.Close)
	host := strings.TrimPrefix(reg.URL, "http://")
	remote := host + "/team/assistant:v1"

	// First daemon: build, tag with the registry's name, push.
	startDaemon(t)
	dir := t.TempDir()
	buildtest.Project(t, dir)
	var out, errOut bytes.Buffer
	steps := [][]string{
		{"build", "-t", "assistant", dir},
		{"tag", "assistant", remote},
		{"push", remote},
	}
	for _, args := range steps {
		if code := run(args, &out, &errOut); code != 0 {
			t.Fatalf("%v: exit %d\n%s", args, code, errOut.String())
		}
	}
	if !strings.Contains(out.String(), "Pushed "+remote) || !strings.Contains(out.String(), "sent 33 of 33 blobs") {
		t.Errorf("push output:\n%s", out.String())
	}
	out.Reset()
	if code := run([]string{"push", "assistant"}, &out, &errOut); code != 1 || !strings.Contains(errOut.String(), "names no registry") {
		t.Errorf("push without a registry: exit %d, %s", code, errOut.String())
	}

	// Second daemon with an empty store: pull and read the image back.
	startDaemon(t)
	out.Reset()
	if code := run([]string{"pull", "--exact", remote}, &out, &errOut); code != 0 {
		t.Fatalf("pull: exit %d\n%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "fetched 33 of 33 blobs") || !strings.Contains(out.String(), "Tagged "+remote) {
		t.Errorf("pull output:\n%s", out.String())
	}
	out.Reset()
	if code := run([]string{"images"}, &out, &errOut); code != 0 || !strings.Contains(out.String(), remote) {
		t.Errorf("images after pull: exit %d\n%s", code, out.String())
	}
	if code := run([]string{"tag", "only-one"}, &out, &errOut); code != 2 {
		t.Errorf("tag with one argument: exit %d", code)
	}
}
