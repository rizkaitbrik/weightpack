package store

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/reference"
)

// Tags live in <root>/index.json, an OCI image-layout index whose manifest
// descriptors carry the full reference ("name:tag") in the standard
// org.opencontainers.image.ref.name annotation.
const indexFile = "index.json"

// MaxManifestSize bounds manifests and configs read into memory.
const MaxManifestSize = 64 << 20

// ErrUnknownImage is returned when a reference resolves to nothing.
var ErrUnknownImage = errors.New("no such image")

// PutJSON stores v as compact JSON and returns its descriptor.
func (s *Store) PutJSON(mediaType string, v any) (ocispec.Descriptor, []byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return ocispec.Descriptor{}, nil, err
	}
	d, size, err := s.Put(bytes.NewReader(b), "")
	if err != nil {
		return ocispec.Descriptor{}, nil, err
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: digest.Digest(d), Size: size}, b, nil
}

// ReadBlob reads a small blob, such as a manifest or config, into memory.
func (s *Store) ReadBlob(dg string) ([]byte, error) {
	rc, err := s.Open(dg)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	b, err := io.ReadAll(io.LimitReader(rc, MaxManifestSize+1))
	if err != nil {
		return nil, err
	}
	if len(b) > MaxManifestSize {
		return nil, fmt.Errorf("blob %s is larger than %d bytes", dg, MaxManifestSize)
	}
	return b, nil
}

// Tag points ref at a manifest, replacing any image it named before. ref is
// a name with an optional tag ("latest" by default) and no digest.
func (s *Store) Tag(ref string, manifest ocispec.Descriptor) (string, error) {
	r, err := reference.ParseNamed(ref)
	if err != nil {
		return "", err
	}
	name := r.String()
	if ok, err := s.Has(manifest.Digest.String()); err != nil || !ok {
		return "", fmt.Errorf("tag %s: manifest %s is not in the store", name, manifest.Digest)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	idx, err := s.readIndex()
	if err != nil {
		return "", err
	}
	desc := manifest
	desc.Annotations = map[string]string{ocispec.AnnotationRefName: name}
	kept := idx.Manifests[:0]
	for _, m := range idx.Manifests {
		if m.Annotations[ocispec.AnnotationRefName] != name {
			kept = append(kept, m)
		}
	}
	idx.Manifests = append(kept, desc)
	return name, s.writeIndex(idx)
}

// Untag removes a tag. The blobs stay until garbage collection.
func (s *Store) Untag(ref string) error {
	r, err := reference.ParseNamed(ref)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	idx, err := s.readIndex()
	if err != nil {
		return err
	}
	kept := idx.Manifests[:0]
	for _, m := range idx.Manifests {
		if m.Annotations[ocispec.AnnotationRefName] != r.String() {
			kept = append(kept, m)
		}
	}
	if len(kept) == len(idx.Manifests) {
		return fmt.Errorf("%w: %s", ErrUnknownImage, r)
	}
	idx.Manifests = kept
	return s.writeIndex(idx)
}

// Resolve finds the manifest a reference names: a tag ("name[:tag]"), a
// digest-pinned name ("name@sha256:…") or a bare manifest digest.
func (s *Store) Resolve(ref string) (ocispec.Descriptor, error) {
	if reference.IsDigest(ref) {
		return s.describeManifest(ref)
	}
	r, err := reference.Parse(ref)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if r.Digest != "" {
		return s.describeManifest(r.Digest)
	}
	if r.Tag == "" {
		r.Tag = reference.DefaultTag
	}
	images, err := s.Images()
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	for _, m := range images {
		if m.Annotations[ocispec.AnnotationRefName] == r.String() {
			return m, nil
		}
	}
	return ocispec.Descriptor{}, fmt.Errorf("%w: %s", ErrUnknownImage, r)
}

// Images returns every tagged image, sorted by reference.
func (s *Store) Images() ([]ocispec.Descriptor, error) {
	s.mu.Lock()
	idx, err := s.readIndex()
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	out := idx.Manifests
	sort.Slice(out, func(i, j int) bool {
		return out[i].Annotations[ocispec.AnnotationRefName] < out[j].Annotations[ocispec.AnnotationRefName]
	})
	return out, nil
}

func (s *Store) describeManifest(dg string) (ocispec.Descriptor, error) {
	b, err := s.ReadBlob(dg)
	if errors.Is(err, ErrNotFound) {
		return ocispec.Descriptor{}, fmt.Errorf("%w: %s", ErrUnknownImage, dg)
	}
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(b, &m); err != nil || m.MediaType != ocispec.MediaTypeImageManifest {
		return ocispec.Descriptor{}, fmt.Errorf("%s is not an image manifest", dg)
	}
	return ocispec.Descriptor{
		MediaType: m.MediaType, ArtifactType: m.ArtifactType,
		Digest: digest.Digest(dg), Size: int64(len(b)),
	}, nil
}

func (s *Store) readIndex() (*ocispec.Index, error) {
	b, err := os.ReadFile(filepath.Join(s.root, indexFile))
	if errors.Is(err, fs.ErrNotExist) {
		idx := &ocispec.Index{MediaType: ocispec.MediaTypeImageIndex, Manifests: []ocispec.Descriptor{}}
		idx.SchemaVersion = 2
		return idx, nil
	}
	if err != nil {
		return nil, err
	}
	var idx ocispec.Index
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("read %s: %w", indexFile, err)
	}
	return &idx, nil
}

func (s *Store) writeIndex(idx *ocispec.Index) error {
	b, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Join(s.root, "tmp"), "index-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), filepath.Join(s.root, indexFile))
}
