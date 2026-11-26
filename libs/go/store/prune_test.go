package store

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// image stores a manifest over the given layer contents and returns it.
func image(t *testing.T, s *Store, layers ...string) ocispec.Descriptor {
	t.Helper()
	cfg, _, err := s.PutJSON("application/x-config", map[string][]string{"layers": layers})
	if err != nil {
		t.Fatal(err)
	}
	m := ocispec.Manifest{Versioned: specs.Versioned{SchemaVersion: 2}, MediaType: ocispec.MediaTypeImageManifest, Config: cfg}
	for _, l := range layers {
		d, size, err := s.Put(bytes.NewReader([]byte(l)), "")
		if err != nil {
			t.Fatal(err)
		}
		m.Layers = append(m.Layers, ocispec.Descriptor{MediaType: "application/x-layer", Digest: digest.Digest(d), Size: size})
	}
	md, _, err := s.PutJSON(ocispec.MediaTypeImageManifest, m)
	if err != nil {
		t.Fatal(err)
	}
	return md
}

func TestPrune(t *testing.T) {
	s, _ := Open(t.TempDir())
	shared := image(t, s, "base", "french")
	if _, err := s.Tag("french", shared); err != nil {
		t.Fatal(err)
	}
	other := image(t, s, "base", "legal") // shares "base"
	if _, err := s.Tag("legal", other); err != nil {
		t.Fatal(err)
	}
	orphan, _, _ := s.Put(bytes.NewReader([]byte("orphan")), "")
	old := filepath.Join(s.Root(), "tmp", "put-old")
	_ = os.WriteFile(old, []byte("x"), 0o600)
	_ = os.Chtimes(old, time.Now().Add(-2*time.Hour), time.Now().Add(-2*time.Hour))
	young := filepath.Join(s.Root(), "tmp", "put-young")
	_ = os.WriteFile(young, []byte("x"), 0o600)

	dry, err := s.Prune(true)
	if err != nil || dry.Blobs != 1 || dry.Bytes != int64(len("orphan")) {
		t.Fatalf("dry run = %+v, %v", dry, err)
	}
	if ok, _ := s.Has(orphan); !ok {
		t.Fatal("a dry run deleted a blob")
	}

	// Untag legal: its own layer, config and manifest go; base stays.
	if err := s.Untag("legal"); err != nil {
		t.Fatal(err)
	}
	res, err := s.Prune(false)
	if err != nil {
		t.Fatal(err)
	}
	if res.Blobs != 4 { // orphan, legal's layer, config and manifest
		t.Errorf("pruned %d blobs, want 4: %+v", res.Blobs, res)
	}
	for _, d := range []string{orphan, other.Digest.String()} {
		if ok, _ := s.Has(d); ok {
			t.Errorf("%s survived the prune", d)
		}
	}
	if _, err := s.Resolve("french"); err != nil {
		t.Errorf("the tagged image is gone: %v", err)
	}
	keep, _ := s.Reachable()
	for d := range keep {
		if ok, _ := s.Has(d); !ok {
			t.Errorf("prune removed reachable blob %s", d)
		}
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Error("an old temporary file survived")
	}
	if _, err := os.Stat(young); err != nil {
		t.Error("a young temporary file was removed")
	}
}
