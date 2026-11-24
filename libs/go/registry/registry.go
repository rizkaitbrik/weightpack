// Package registry moves images between the local store and OCI registries.
//
// Both directions move only the blobs the other side lacks, so pushing a
// second image on a base the registry already holds sends just its own
// layers. A pull reads the manifest and config first and checks them with
// spec.CheckImage before fetching any tensor; every fetched blob is verified
// against its digest as the store writes it.
//
// Credentials come from the Docker configuration (~/.docker/config.json, or
// $DOCKER_CONFIG), including credential helpers, as `docker login` leaves
// them.
package registry

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/registry/remote"
	"oras.land/oras-go/v2/registry/remote/auth"
	"oras.land/oras-go/v2/registry/remote/credentials"
	"oras.land/oras-go/v2/registry/remote/retry"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

// Options tune a push or pull.
type Options struct {
	// PlainHTTP talks to the registry over HTTP. Registries on localhost
	// and 127.0.0.1 always use HTTP.
	PlainHTTP bool
	// Exact makes a pull refuse images with lossy layers.
	Exact bool
}

// concurrency is how many blobs move at once.
const concurrency = 4

// Push sends a local image to the registry its reference names, then tags
// it there (latest when the reference has neither tag nor digest). Only
// blobs the registry lacks are uploaded.
func Push(ctx context.Context, s *store.Store, ref string, opts Options, progress func(api.Event)) (*api.TransferResult, error) {
	if progress == nil {
		progress = func(api.Event) {}
	}
	r, err := remoteRef(ref, "push")
	if err != nil {
		return nil, err
	}
	// No tag and no digest means latest, as the local store resolves it and
	// as Pull reads it, so the image is tagged latest on the registry too.
	if r.Tag == "" && r.Digest == "" {
		r.Tag = reference.DefaultTag
	}
	root, err := s.Resolve(ref)
	if err != nil {
		return nil, err
	}
	var m ocispec.Manifest
	var cfg spec.Config
	if err := s.ReadJSON(root.Digest.String(), &m); err != nil {
		return nil, err
	}
	if err := s.ReadJSON(m.Config.Digest.String(), &cfg); err != nil {
		return nil, err
	}
	if err := spec.CheckImage(&m, &cfg); err != nil {
		return nil, fmt.Errorf("local image %s is inconsistent, not pushing it: %w", ref, err)
	}
	repo, err := newRepository(r, opts)
	if err != nil {
		return nil, err
	}

	step := "push " + r.String()
	res := newResult(r.String(), root, &m)
	progress(api.Event{Step: step, Message: fmt.Sprintf("%d blobs, %s", res.Blobs, api.FormatSize(res.Size))})
	if err := oras.CopyGraph(ctx, local{s}, repo, root, res.copyOptions(step, progress)); err != nil {
		return nil, fmt.Errorf("push %s: %w", r, err)
	}
	if r.Tag != "" {
		if err := repo.Tag(ctx, root, r.Tag); err != nil {
			return nil, fmt.Errorf("tag %s: %w", r, err)
		}
	}
	progress(api.Event{Step: step, Message: fmt.Sprintf("sent %d blobs, %s; the registry had the rest", res.Moved, api.FormatSize(res.MovedBytes))})
	return &res.TransferResult, nil
}

// Pull fetches an image into the store and, when the reference has a tag
// (or none at all, meaning latest), tags it locally under the same name.
// Pulling by digest sets no tag.
func Pull(ctx context.Context, s *store.Store, ref string, opts Options, progress func(api.Event)) (*api.TransferResult, error) {
	if progress == nil {
		progress = func(api.Event) {}
	}
	r, err := remoteRef(ref, "pull")
	if err != nil {
		return nil, err
	}
	if r.Tag == "" && r.Digest == "" {
		r.Tag = reference.DefaultTag
	}
	repo, err := newRepository(r, opts)
	if err != nil {
		return nil, err
	}
	target := r.Digest
	if target == "" {
		target = r.Tag
	}
	root, err := repo.Resolve(ctx, target)
	if err != nil {
		return nil, fmt.Errorf("pull %s: %w", r, err)
	}

	// Check the manifest and config before fetching any tensor.
	m, err := fetchManifest(ctx, repo, root, r)
	if err != nil {
		return nil, err
	}
	if m.Config.Size > store.MaxManifestSize {
		return nil, fmt.Errorf("pull %s: config of %d bytes is too large", r, m.Config.Size)
	}
	raw, err := content.FetchAll(ctx, repo, m.Config)
	if err != nil {
		return nil, fmt.Errorf("pull %s: config: %w", r, err)
	}
	var cfg spec.Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("pull %s: config: %w", r, err)
	}
	if err := spec.CheckImage(m, &cfg); err != nil {
		return nil, fmt.Errorf("pull %s: refusing an inconsistent image: %w", r, err)
	}
	if opts.Exact {
		for _, d := range m.Layers {
			if !spec.IsLossless(d.MediaType) {
				return nil, fmt.Errorf("pull %s: layer %q holds a lossy blob (%s); pull without --exact to accept it",
					r, d.Annotations[spec.AnnotationLayer], d.MediaType)
			}
		}
	}

	step := "pull " + r.String()
	res := newResult(r.String(), root, m)
	progress(api.Event{Step: step, Message: fmt.Sprintf("%d blobs, %s, %s base", res.Blobs, api.FormatSize(res.Size), orUnknown(cfg.Arch.Family))})
	if err := oras.CopyGraph(ctx, repo, local{s}, root, res.copyOptions(step, progress)); err != nil {
		return nil, fmt.Errorf("pull %s: %w", r, err)
	}
	if r.Tag != "" {
		tagged := reference.Reference{Host: r.Host, Path: r.Path, Tag: r.Tag}
		name, err := s.Tag(tagged.String(), root)
		if err != nil {
			return nil, err
		}
		res.Tag = name
	}
	progress(api.Event{Step: step, Message: fmt.Sprintf("fetched %d blobs, %s; the store had the rest", res.Moved, api.FormatSize(res.MovedBytes))})
	return &res.TransferResult, nil
}

// remoteRef parses a reference that must name a registry host.
func remoteRef(ref, verb string) (reference.Reference, error) {
	r, err := reference.Parse(ref)
	if err != nil {
		return reference.Reference{}, err
	}
	if r.Host == "" {
		return reference.Reference{}, fmt.Errorf(
			"%s %s: the reference names no registry; name one, as in registry.example.com/%s (wpk tag gives a local image another name)",
			verb, ref, r.Path)
	}
	return r, nil
}

func newRepository(r reference.Reference, opts Options) (*remote.Repository, error) {
	repo, err := remote.NewRepository(r.Name())
	if err != nil {
		return nil, err
	}
	repo.PlainHTTP = opts.PlainHTTP || isLoopback(r.Host)
	creds, err := credentials.NewStoreFromDocker(credentials.StoreOptions{})
	if err != nil {
		return nil, fmt.Errorf("read Docker credentials: %w", err)
	}
	repo.Client = &auth.Client{
		Client:     retry.DefaultClient,
		Cache:      auth.NewCache(),
		Credential: credentials.Credential(creds),
		ClientID:   "weightpack",
	}
	return repo, nil
}

// isLoopback reports whether host (with an optional port) is this machine.
func isLoopback(host string) bool {
	h, _, err := net.SplitHostPort(host)
	if err != nil {
		h = host
	}
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

func fetchManifest(ctx context.Context, repo *remote.Repository, root ocispec.Descriptor, r reference.Reference) (*ocispec.Manifest, error) {
	if root.MediaType != ocispec.MediaTypeImageManifest {
		return nil, fmt.Errorf("pull %s: not a Weightpack image (media type %q)", r, root.MediaType)
	}
	if root.Size > store.MaxManifestSize {
		return nil, fmt.Errorf("pull %s: manifest of %d bytes is too large", r, root.Size)
	}
	raw, err := content.FetchAll(ctx, repo, root)
	if err != nil {
		return nil, fmt.Errorf("pull %s: manifest: %w", r, err)
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("pull %s: manifest: %w", r, err)
	}
	if m.ArtifactType != spec.ArtifactType {
		return nil, fmt.Errorf("pull %s: not a Weightpack image (artifactType %q)", r, m.ArtifactType)
	}
	return &m, nil
}

// result counts what a transfer moves. oras calls the copy hooks from
// several goroutines.
type result struct {
	api.TransferResult
	mu         sync.Mutex
	nextReport int64
}

func newResult(ref string, root ocispec.Descriptor, m *ocispec.Manifest) *result {
	res := &result{TransferResult: api.TransferResult{Ref: ref, Digest: root.Digest.String()}}
	seen := map[string]bool{}
	for _, d := range append([]ocispec.Descriptor{root, m.Config}, m.Layers...) {
		if !seen[d.Digest.String()] {
			seen[d.Digest.String()] = true
			res.Blobs++
			res.Size += d.Size
		}
	}
	res.nextReport = reportEvery(res.Size)
	return res
}

// reportEvery spaces progress events about a tenth of an image apart, and
// never closer than 256 MB, so small images stay quiet.
func reportEvery(total int64) int64 { return max(total/10, 256<<20) }

func (res *result) copyOptions(step string, progress func(api.Event)) oras.CopyGraphOptions {
	return oras.CopyGraphOptions{
		Concurrency:      concurrency,
		MaxMetadataBytes: store.MaxManifestSize,
		PostCopy: func(_ context.Context, d ocispec.Descriptor) error {
			res.mu.Lock()
			defer res.mu.Unlock()
			res.Moved++
			res.MovedBytes += d.Size
			if res.MovedBytes >= res.nextReport && res.MovedBytes < res.Size {
				progress(api.Event{Step: step, Message: fmt.Sprintf("%s of at most %s", api.FormatSize(res.MovedBytes), api.FormatSize(res.Size))})
				res.nextReport = res.MovedBytes + reportEvery(res.Size)
			}
			return nil
		},
	}
}

// local adapts the store to the oras target interfaces.
type local struct{ s *store.Store }

func (l local) Fetch(_ context.Context, d ocispec.Descriptor) (io.ReadCloser, error) {
	return l.s.Open(d.Digest.String())
}

func (l local) Exists(_ context.Context, d ocispec.Descriptor) (bool, error) {
	return l.s.Has(d.Digest.String())
}

// Push stores a fetched blob. The store checks it against the expected
// digest before keeping it.
func (l local) Push(_ context.Context, d ocispec.Descriptor, r io.Reader) error {
	_, size, err := l.s.Put(r, d.Digest.String())
	if err != nil {
		return err
	}
	if size != d.Size {
		return fmt.Errorf("blob %s: %d bytes, descriptor says %d", d.Digest, size, d.Size)
	}
	return nil
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

var _ content.Storage = local{}
