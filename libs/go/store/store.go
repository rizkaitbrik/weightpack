// Package store implements the local content-addressed blob store used by
// wpkd. Blobs live under <root>/blobs/sha256/<first two hex chars>/<hex>,
// are written atomically and are verified against their digest on write.
package store

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ErrNotFound is returned when a blob is not in the store.
var ErrNotFound = errors.New("blob not found")

// ErrDigestMismatch is returned when written content does not match the
// expected digest.
var ErrDigestMismatch = errors.New("digest mismatch")

const algo = "sha256"

// Store is a content-addressed blob store rooted at a directory.
type Store struct {
	root string
}

// Open creates the store layout under root if needed and returns the store.
func Open(root string) (*Store, error) {
	for _, dir := range []string{
		filepath.Join(root, "blobs", algo),
		filepath.Join(root, "tmp"),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("create store layout: %w", err)
		}
	}
	return &Store{root: root}, nil
}

// Root returns the directory the store lives in.
func (s *Store) Root() string { return s.root }

// ParseDigest validates a "sha256:<64 hex>" digest and returns its hex part.
func ParseDigest(digest string) (string, error) {
	hexPart, ok := strings.CutPrefix(digest, algo+":")
	if !ok {
		return "", fmt.Errorf("unsupported digest %q: want %s:<hex>", digest, algo)
	}
	if len(hexPart) != sha256.Size*2 {
		return "", fmt.Errorf("digest %q: want %d hex characters", digest, sha256.Size*2)
	}
	if _, err := hex.DecodeString(hexPart); err != nil {
		return "", fmt.Errorf("digest %q: %w", digest, err)
	}
	return hexPart, nil
}

func (s *Store) blobPath(hexPart string) string {
	return filepath.Join(s.root, "blobs", algo, hexPart[:2], hexPart)
}

// Put streams r into the store and returns its digest and size. If expected
// is not empty, the content must hash to it or nothing is stored. Writing a
// blob that already exists is a no-op.
func (s *Store) Put(r io.Reader, expected string) (digest string, size int64, err error) {
	if expected != "" {
		if _, err := ParseDigest(expected); err != nil {
			return "", 0, err
		}
	}
	tmp, err := os.CreateTemp(filepath.Join(s.root, "tmp"), "put-*")
	if err != nil {
		return "", 0, err
	}
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
	}()

	h := sha256.New()
	size, err = io.Copy(io.MultiWriter(tmp, h), r)
	if err != nil {
		return "", 0, err
	}
	if err := tmp.Sync(); err != nil {
		return "", 0, err
	}
	hexPart := hex.EncodeToString(h.Sum(nil))
	digest = algo + ":" + hexPart
	if expected != "" && expected != digest {
		return "", 0, fmt.Errorf("%w: got %s, want %s", ErrDigestMismatch, digest, expected)
	}

	final := s.blobPath(hexPart)
	if _, err := os.Stat(final); err == nil {
		return digest, size, nil
	}
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return "", 0, err
	}
	if err := tmp.Close(); err != nil {
		return "", 0, err
	}
	if err := os.Rename(tmp.Name(), final); err != nil {
		return "", 0, err
	}
	return digest, size, nil
}

// Has reports whether the blob is present.
func (s *Store) Has(digest string) (bool, error) {
	hexPart, err := ParseDigest(digest)
	if err != nil {
		return false, err
	}
	_, err = os.Stat(s.blobPath(hexPart))
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	return err == nil, err
}

// Open returns a reader for the blob. The caller must close it.
func (s *Store) Open(digest string) (io.ReadCloser, error) {
	hexPart, err := ParseDigest(digest)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(s.blobPath(hexPart))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%w: %s", ErrNotFound, digest)
	}
	return f, err
}

// Missing returns the digests from want that the store does not hold, in
// order. It is what push and pull use to move only new blobs.
func (s *Store) Missing(want []string) ([]string, error) {
	var missing []string
	for _, d := range want {
		ok, err := s.Has(d)
		if err != nil {
			return nil, err
		}
		if !ok {
			missing = append(missing, d)
		}
	}
	return missing, nil
}

// Delete removes a blob. Deleting a missing blob is not an error.
func (s *Store) Delete(digest string) error {
	hexPart, err := ParseDigest(digest)
	if err != nil {
		return err
	}
	err = os.Remove(s.blobPath(hexPart))
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	return err
}
