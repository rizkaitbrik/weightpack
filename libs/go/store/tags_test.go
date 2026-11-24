package store

import (
	"errors"
	"testing"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func putManifest(t *testing.T, s *Store, note string) ocispec.Descriptor {
	t.Helper()
	m := ocispec.Manifest{MediaType: ocispec.MediaTypeImageManifest, ArtifactType: "application/x-test",
		Annotations: map[string]string{"note": note}}
	m.SchemaVersion = 2
	d, _, err := s.PutJSON(ocispec.MediaTypeImageManifest, m)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestTagResolveUntag(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	m1 := putManifest(t, s, "one")
	m2 := putManifest(t, s, "two")

	name, err := s.Tag("assistant", m1)
	if err != nil || name != "assistant:latest" {
		t.Fatalf("Tag = %q, %v", name, err)
	}
	if _, err := s.Tag("support:v1", m2); err != nil {
		t.Fatal(err)
	}
	for ref, want := range map[string]ocispec.Descriptor{
		"assistant":                      m1,
		"assistant:latest":               m1,
		"support:v1":                     m2,
		m2.Digest.String():               m2,
		"whatever@" + m1.Digest.String(): m1,
	} {
		got, err := s.Resolve(ref)
		if err != nil || got.Digest != want.Digest {
			t.Errorf("Resolve(%q) = %s, %v; want %s", ref, got.Digest, err, want.Digest)
		}
	}

	// Retagging moves the tag instead of adding a second entry.
	if _, err := s.Tag("assistant", m2); err != nil {
		t.Fatal(err)
	}
	images, _ := s.Images()
	if len(images) != 2 {
		t.Fatalf("Images = %d entries, want 2", len(images))
	}
	if got, _ := s.Resolve("assistant"); got.Digest != m2.Digest {
		t.Error("retag did not move the tag")
	}

	// A reopened store sees the same tags.
	s2, _ := Open(s.Root())
	if got, err := s2.Resolve("support:v1"); err != nil || got.Digest != m2.Digest {
		t.Errorf("after reopen: %s, %v", got.Digest, err)
	}

	if err := s.Untag("support:v1"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Resolve("support:v1"); !errors.Is(err, ErrUnknownImage) {
		t.Errorf("Resolve after Untag = %v", err)
	}
	if err := s.Untag("support:v1"); !errors.Is(err, ErrUnknownImage) {
		t.Errorf("second Untag = %v", err)
	}
}

func TestTagRejects(t *testing.T) {
	s, _ := Open(t.TempDir())
	m := putManifest(t, s, "x")
	if _, err := s.Tag("Bad Name", m); err == nil {
		t.Error("bad name accepted")
	}
	if _, err := s.Tag("a@"+m.Digest.String(), m); err == nil {
		t.Error("digest accepted in a tag")
	}
	missing := m
	missing.Digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
	if _, err := s.Tag("a", missing); err == nil {
		t.Error("tag to a missing manifest accepted")
	}
	if _, err := s.Resolve("nothing"); !errors.Is(err, ErrUnknownImage) {
		t.Errorf("Resolve(nothing) = %v", err)
	}
	if _, err := s.Resolve(missing.Digest.String()); !errors.Is(err, ErrUnknownImage) {
		t.Errorf("Resolve(missing digest) = %v", err)
	}
}
