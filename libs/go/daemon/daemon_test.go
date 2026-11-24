package daemon

import (
	"bytes"
	"context"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ggcr "github.com/google/go-containerregistry/pkg/registry"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

// serveUnix starts the handler on a Unix socket and returns a client for it.
func serveUnix(t *testing.T, s *store.Store) *api.Client {
	t.Helper()
	dir, err := os.MkdirTemp("", "wpkd") // short path: socket paths are limited
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	socket := filepath.Join(dir, "s")
	ln, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(NewHandler(s, "test", nil))
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)
	return api.NewClient(socket)
}

func TestBuildImagesInspect(t *testing.T) {
	dir := t.TempDir()
	wf := buildtest.Project(t, dir)
	s, err := store.Open(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	c := serveUnix(t, s)
	ctx := context.Background()

	var events []api.Event
	res, err := c.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"assistant:v1"}}, func(e api.Event) {
		events = append(events, e)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) == 0 || res.Manifest.Digest == "" || len(res.Tags) != 1 {
		t.Fatalf("result %+v after %d events", res, len(events))
	}

	images, err := c.Images(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(images) != 1 || images[0].Ref != "assistant:v1" || images[0].Digest != res.Manifest.Digest.String() ||
		images[0].Size != res.Size || images[0].Layers != 5 || images[0].Arch != "llama" {
		t.Errorf("images = %+v, build result size %d", images, res.Size)
	}

	detail, err := c.Inspect(ctx, "assistant:v1")
	if err != nil {
		t.Fatal(err)
	}
	if err := spec.CheckImage(&detail.Manifest, &detail.Config); err != nil {
		t.Errorf("inspected image fails CheckImage: %v", err)
	}
	if _, err := c.Inspect(ctx, "nothing:here"); err == nil || !strings.Contains(err.Error(), "no such image") {
		t.Errorf("inspect unknown = %v", err)
	}
}

func TestBuildErrors(t *testing.T) {
	dir := t.TempDir()
	s, _ := store.Open(filepath.Join(dir, "store"))
	c := serveUnix(t, s)
	ctx := context.Background()

	if _, err := c.Build(ctx, api.BuildRequest{Weightfile: "Weightfile"}, nil); err == nil || !strings.Contains(err.Error(), "absolute") {
		t.Errorf("relative path: %v", err)
	}
	// A build that fails after the stream started reports its error as the
	// last event.
	wf := filepath.Join(dir, "Weightfile")
	_ = os.WriteFile(wf, []byte("FROM ./missing\n"), 0o644)
	if _, err := c.Build(ctx, api.BuildRequest{Weightfile: wf}, nil); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Errorf("failing build: %v", err)
	}
}

func TestBlobLookup(t *testing.T) {
	s, _ := store.Open(t.TempDir())
	d, _, err := s.Put(bytes.NewReader([]byte("blob")), "")
	if err != nil {
		t.Fatal(err)
	}
	h := NewHandler(s, "test", nil)
	for path, want := range map[string]int{
		"/v1/blobs/" + d: http.StatusNoContent,
		"/v1/blobs/sha256:" + strings.Repeat("0", 64): http.StatusNotFound,
		"/v1/blobs/bogus": http.StatusBadRequest,
		"/healthz":        http.StatusOK,
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != want {
			t.Errorf("GET %s: status %d, want %d", path, rec.Code, want)
		}
	}
}

func TestClientWithoutDaemon(t *testing.T) {
	c := api.NewClient(filepath.Join(t.TempDir(), "absent.sock"))
	if _, err := c.Images(context.Background()); err == nil || !strings.Contains(err.Error(), "is the daemon running") {
		t.Errorf("err = %v", err)
	}
}

func TestTagPushPull(t *testing.T) {
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	reg := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(reg.Close)
	host := strings.TrimPrefix(reg.URL, "http://")

	dir := t.TempDir()
	wf := buildtest.Project(t, dir)
	a, _ := store.Open(filepath.Join(dir, "a"))
	b, _ := store.Open(filepath.Join(dir, "b"))
	ca, cb := serveUnix(t, a), serveUnix(t, b)
	ctx := context.Background()

	built, err := ca.Build(ctx, api.BuildRequest{Weightfile: wf, Tags: []string{"assistant"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	remoteRef := host + "/team/assistant:v1"
	tagged, err := ca.Tag(ctx, api.TagRequest{Source: "assistant", Target: remoteRef})
	if err != nil || tagged.Ref != remoteRef || tagged.Digest != built.Manifest.Digest.String() {
		t.Fatalf("tag = %+v, %v", tagged, err)
	}
	if _, err := ca.Tag(ctx, api.TagRequest{Source: "absent", Target: "x"}); err == nil || !strings.Contains(err.Error(), "no such image") {
		t.Errorf("tag of a missing image: %v", err)
	}

	var events []api.Event
	pushed, err := ca.Push(ctx, api.PushRequest{Ref: remoteRef}, func(e api.Event) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	if pushed.Moved != pushed.Blobs || len(events) == 0 {
		t.Errorf("push = %+v after %d events", pushed, len(events))
	}
	if _, err := ca.Push(ctx, api.PushRequest{Ref: "assistant"}, nil); err == nil || !strings.Contains(err.Error(), "names no registry") {
		t.Errorf("push without a registry: %v", err)
	}

	pulled, err := cb.Pull(ctx, api.PullRequest{Ref: remoteRef, Exact: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pulled.Digest != built.Manifest.Digest.String() || pulled.Tag != remoteRef {
		t.Errorf("pull = %+v", pulled)
	}
	detail, err := cb.Inspect(ctx, remoteRef)
	if err != nil {
		t.Fatal(err)
	}
	if err := spec.CheckImage(&detail.Manifest, &detail.Config); err != nil {
		t.Errorf("pulled image: %v", err)
	}
}
