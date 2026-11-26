package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// PruneResult says what a prune removed, or would remove in a dry run.
type PruneResult struct {
	Blobs     int   `json:"blobs"`
	Bytes     int64 `json:"bytes"`
	KeptBlobs int   `json:"kept_blobs"`
	KeptBytes int64 `json:"kept_bytes"`
}

// tmpGrace is how old a file in tmp/ must be before prune removes it: a
// younger one may belong to a write in progress.
const tmpGrace = time.Hour

// Reachable returns every blob a tagged image needs: its manifest, its
// config and its layer blobs.
func (s *Store) Reachable() (map[string]bool, error) {
	images, err := s.Images()
	if err != nil {
		return nil, err
	}
	keep := map[string]bool{}
	for _, img := range images {
		keep[img.Digest.String()] = true
		raw, err := s.ReadBlob(img.Digest.String())
		if err != nil {
			return nil, fmt.Errorf("image %s: %w", img.Annotations[ocispec.AnnotationRefName], err)
		}
		var m ocispec.Manifest
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, fmt.Errorf("image %s: %w", img.Annotations[ocispec.AnnotationRefName], err)
		}
		keep[m.Config.Digest.String()] = true
		for _, l := range m.Layers {
			keep[l.Digest.String()] = true
		}
	}
	return keep, nil
}

// Prune deletes every blob no tagged image reaches, and leftover temporary
// files, as docker image prune does. With dryRun it only counts. The caller
// must keep writers (builds, pulls) out while it runs: a blob written but
// not yet tagged looks unreachable.
func (s *Store) Prune(dryRun bool) (PruneResult, error) {
	keep, err := s.Reachable()
	if err != nil {
		return PruneResult{}, err
	}
	var res PruneResult
	err = filepath.WalkDir(filepath.Join(s.root, "blobs", algo), func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		digest := algo + ":" + d.Name()
		if _, perr := ParseDigest(digest); perr != nil {
			return nil // not a blob; leave it alone
		}
		if keep[digest] {
			res.KeptBlobs++
			res.KeptBytes += info.Size()
			return nil
		}
		res.Blobs++
		res.Bytes += info.Size()
		if dryRun {
			return nil
		}
		return os.Remove(p)
	})
	if err != nil || dryRun {
		return res, err
	}
	entries, _ := os.ReadDir(filepath.Join(s.root, "tmp"))
	for _, e := range entries {
		if info, err := e.Info(); err == nil && time.Since(info.ModTime()) > tmpGrace {
			_ = os.RemoveAll(filepath.Join(s.root, "tmp", e.Name()))
		}
	}
	return res, nil
}
