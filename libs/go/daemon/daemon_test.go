package daemon

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
