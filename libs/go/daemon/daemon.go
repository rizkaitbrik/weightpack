// Package daemon is wpkd's HTTP API over the local store. See package api
// for the endpoints and wire types.
package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/export"
	"github.com/rizkaitbrik/weightpack/libs/go/hf"
	"github.com/rizkaitbrik/weightpack/libs/go/registry"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/squash"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
	"github.com/rizkaitbrik/weightpack/libs/go/worker"
)

const maxRequestBody = 1 << 20

type server struct {
	store   *store.Store
	version string
	log     *slog.Logger
	worker  *worker.Client
	hf      *hf.Client
	ops     sync.RWMutex // shared by store operations, exclusive for prune
}

// shared runs h holding the store lock shared.
func (s *server) shared(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.ops.RLock()
		defer s.ops.RUnlock()
		h(w, r)
	}
}

// Option configures the handler.
type Option func(*server)

// WithHF sets the Hugging Face client builds fetch hf:// sources with.
func WithHF(c *hf.Client) Option { return func(s *server) { s.hf = c } }

// WithWorker sets the build worker that squash (and later imports and
// evals) run on. Without one, those endpoints explain how to start one.
func WithWorker(c *worker.Client) Option { return func(s *server) { s.worker = c } }

// NewHandler returns the daemon's API over s.
func NewHandler(s *store.Store, version string, logger *slog.Logger, opts ...Option) http.Handler {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	srv := &server{store: s, version: version, log: logger}
	for _, o := range opts {
		o(srv)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", srv.health)
	mux.HandleFunc("GET /v1/blobs/{digest}", srv.blob)
	// Everything that writes blobs, reads them for a transfer, or changes
	// tags shares the store; prune takes it alone, so it never deletes a
	// blob in use.
	mux.HandleFunc("POST /v1/build", srv.shared(srv.build))
	mux.HandleFunc("POST /v1/push", srv.shared(srv.push))
	mux.HandleFunc("POST /v1/pull", srv.shared(srv.pull))
	mux.HandleFunc("POST /v1/tag", srv.shared(srv.tag))
	mux.HandleFunc("POST /v1/squash", srv.shared(srv.squash))
	mux.HandleFunc("POST /v1/export", srv.shared(srv.export))
	mux.HandleFunc("POST /v1/rm", srv.shared(srv.rm))
	mux.HandleFunc("POST /v1/prune", srv.prune)
	mux.HandleFunc("GET /v1/images", srv.images)
	mux.HandleFunc("GET /v1/images/inspect", srv.inspect)
	return mux
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, api.ErrorBody{Error: err.Error()})
}

func (s *server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "version": s.version, "root": s.store.Root()})
}

func (s *server) blob(w http.ResponseWriter, r *http.Request) {
	ok, err := s.store.Has(r.PathValue("digest"))
	switch {
	case err != nil:
		writeError(w, http.StatusBadRequest, err)
	case !ok:
		writeError(w, http.StatusNotFound, errors.New("blob not found"))
	default:
		w.WriteHeader(http.StatusNoContent)
	}
}

// decode reads a JSON request body into v, answering 400 on failure.
func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("bad request: %w", err))
		return false
	}
	return true
}

// startStream begins an NDJSON reply and returns a function that writes one
// event per line. It is safe to call from several goroutines.
func startStream(w http.ResponseWriter) func(api.Event) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	var mu sync.Mutex
	return func(e api.Event) {
		mu.Lock()
		defer mu.Unlock()
		_ = enc.Encode(e)
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// build streams progress as NDJSON. The last line has a result or an error.
// A client that disconnects cancels the build.
func (s *server) build(w http.ResponseWriter, r *http.Request) {
	var req api.BuildRequest
	if !decode(w, r, &req) {
		return
	}
	if !filepath.IsAbs(req.Weightfile) {
		writeError(w, http.StatusBadRequest, fmt.Errorf("weightfile must be an absolute path, got %q", req.Weightfile))
		return
	}
	send := startStream(w)
	s.log.Info("build started", "weightfile", req.Weightfile, "tags", req.Tags)
	opts := build.Options{Weightfile: req.Weightfile, Tags: req.Tags, Pull: s.pullForBuild, NoCache: req.NoCache}
	if s.worker != nil {
		opts.Worker = s.worker
	}
	opts.HF = s.hf
	res, err := build.Build(r.Context(), s.store, opts, send)
	if err != nil {
		s.log.Info("build failed", "weightfile", req.Weightfile, "err", err)
		send(api.Event{Error: err.Error()})
		return
	}
	s.log.Info("build finished", "digest", res.Manifest.Digest, "new_bytes", res.NewBytes)
	send(api.Event{Result: res})
}

// pullForBuild fetches an image a Weightfile names when the store lacks it,
// as docker build pulls a missing FROM image. Registries on loopback
// addresses use plain HTTP; for another insecure registry, wpk pull
// --plain-http the image first.
func (s *server) pullForBuild(ctx context.Context, ref string, progress func(api.Event)) error {
	_, err := registry.Pull(ctx, s.store, ref, registry.Options{}, progress)
	return err
}

// push and pull stream progress like build. A client that disconnects
// cancels the transfer.
func (s *server) push(w http.ResponseWriter, r *http.Request) {
	var req api.PushRequest
	if !decode(w, r, &req) {
		return
	}
	send := startStream(w)
	s.log.Info("push started", "ref", req.Ref)
	res, err := registry.Push(r.Context(), s.store, req.Ref, registry.Options{PlainHTTP: req.PlainHTTP}, send)
	s.finishTransfer("push", req.Ref, res, err, send)
}

func (s *server) pull(w http.ResponseWriter, r *http.Request) {
	var req api.PullRequest
	if !decode(w, r, &req) {
		return
	}
	send := startStream(w)
	s.log.Info("pull started", "ref", req.Ref, "exact", req.Exact)
	res, err := registry.Pull(r.Context(), s.store, req.Ref, registry.Options{PlainHTTP: req.PlainHTTP, Exact: req.Exact}, send)
	s.finishTransfer("pull", req.Ref, res, err, send)
}

func (s *server) finishTransfer(verb, ref string, res *api.TransferResult, err error, send func(api.Event)) {
	if err != nil {
		s.log.Info(verb+" failed", "ref", ref, "err", err)
		send(api.Event{Error: err.Error()})
		return
	}
	s.log.Info(verb+" finished", "ref", ref, "digest", res.Digest, "moved_bytes", res.MovedBytes)
	send(api.Event{Transfer: res})
}

// squash streams progress like build; the last line has the new image.
func (s *server) squash(w http.ResponseWriter, r *http.Request) {
	var req api.SquashRequest
	if !decode(w, r, &req) {
		return
	}
	if s.worker == nil {
		writeError(w, http.StatusServiceUnavailable, errors.New("wpkd has no build worker; restart it with --worker (see wpkd -h)"))
		return
	}
	send := startStream(w)
	s.log.Info("squash started", "ref", req.Ref, "expert", req.Expert)
	res, err := squash.Squash(r.Context(), s.store, s.worker, req.Ref, squash.Options{Expert: req.Expert, Tags: req.Tags}, send)
	if err != nil {
		err = worker.Explain(err, s.worker.Socket())
		s.log.Info("squash failed", "ref", req.Ref, "err", err)
		send(api.Event{Error: err.Error()})
		return
	}
	s.log.Info("squash finished", "ref", req.Ref, "digest", res.Manifest.Digest, "new_bytes", res.NewBytes)
	send(api.Event{Result: res})
}

// export streams progress like build; the last line says what it wrote.
func (s *server) export(w http.ResponseWriter, r *http.Request) {
	var req api.ExportRequest
	if !decode(w, r, &req) {
		return
	}
	if req.Format != "" && req.Format != "hf" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("export format %q is not supported; only hf is so far", req.Format))
		return
	}
	if !filepath.IsAbs(req.Dir) {
		writeError(w, http.StatusBadRequest, fmt.Errorf("export directory must be an absolute path, got %q", req.Dir))
		return
	}
	send := startStream(w)
	s.log.Info("export started", "ref", req.Ref, "dir", req.Dir)
	res, err := export.Export(r.Context(), s.store, req.Ref, export.Options{Dir: req.Dir, MaxShardSize: req.MaxShardSize}, send)
	if err != nil {
		s.log.Info("export failed", "ref", req.Ref, "err", err)
		send(api.Event{Error: err.Error()})
		return
	}
	s.log.Info("export finished", "ref", req.Ref, "dir", res.Dir, "bytes", res.Bytes)
	send(api.Event{Export: res})
}

// rm untags images; their blobs stay until a prune.
func (s *server) rm(w http.ResponseWriter, r *http.Request) {
	var req api.RmRequest
	if !decode(w, r, &req) {
		return
	}
	var removed []string
	for _, ref := range req.Refs {
		if err := s.store.Untag(ref); err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, store.ErrUnknownImage) {
				status = http.StatusNotFound
			}
			writeError(w, status, err)
			return
		}
		removed = append(removed, ref)
	}
	writeJSON(w, http.StatusOK, api.RmResponse{Removed: removed})
}

// prune deletes unreachable blobs, stale build-cache entries and, if asked,
// the hf:// download cache. It holds the store alone.
func (s *server) prune(w http.ResponseWriter, r *http.Request) {
	var req api.PruneRequest
	if !decode(w, r, &req) {
		return
	}
	s.ops.Lock()
	defer s.ops.Unlock()
	res, err := s.store.Prune(req.DryRun)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	out := api.PruneResponse{Blobs: res.Blobs, Bytes: res.Bytes, KeptBlobs: res.KeptBlobs, KeptBytes: res.KeptBytes}
	if !req.DryRun {
		if out.CacheEntries, err = build.PruneCache(s.store.Root(), s.store); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
	}
	if req.Downloads && s.hf != nil && s.hf.CacheDir != "" {
		out.DownloadBytes = dirSize(s.hf.CacheDir)
		if !req.DryRun {
			if err := os.RemoveAll(s.hf.CacheDir); err != nil {
				writeError(w, http.StatusInternalServerError, err)
				return
			}
		}
	}
	s.log.Info("prune", "dry_run", req.DryRun, "blobs", out.Blobs, "bytes", out.Bytes, "downloads", out.DownloadBytes)
	writeJSON(w, http.StatusOK, out)
}

func dirSize(dir string) int64 {
	var n int64
	_ = filepath.WalkDir(dir, func(_ string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}

func (s *server) tag(w http.ResponseWriter, r *http.Request) {
	var req api.TagRequest
	if !decode(w, r, &req) {
		return
	}
	d, err := s.store.Resolve(req.Source)
	if errors.Is(err, store.ErrUnknownImage) {
		writeError(w, http.StatusNotFound, err)
		return
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	name, err := s.store.Tag(req.Target, d)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, api.TagResponse{Ref: name, Digest: d.Digest.String()})
}

func (s *server) images(w http.ResponseWriter, _ *http.Request) {
	descs, err := s.store.Images()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	out := []api.Image{}
	for _, d := range descs {
		m, c, err := s.load(d.Digest.String())
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		img := api.Image{
			Ref: d.Annotations[ocispec.AnnotationRefName], Digest: d.Digest.String(),
			Size: imageSize(m, d.Size), Layers: len(c.Layers), Arch: c.Arch.Family,
		}
		out = append(out, img)
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *server) inspect(w http.ResponseWriter, r *http.Request) {
	ref := r.URL.Query().Get("ref")
	if ref == "" {
		writeError(w, http.StatusBadRequest, errors.New("missing ref"))
		return
	}
	d, err := s.store.Resolve(ref)
	if errors.Is(err, store.ErrUnknownImage) {
		writeError(w, http.StatusNotFound, err)
		return
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	m, c, err := s.load(d.Digest.String())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, api.ImageDetail{
		Ref: d.Annotations[ocispec.AnnotationRefName], Digest: d.Digest.String(),
		Manifest: *m, Config: *c,
	})
}

// load reads a manifest and its config from the store.
func (s *server) load(manifestDigest string) (*ocispec.Manifest, *spec.Config, error) {
	var m ocispec.Manifest
	if err := s.store.ReadJSON(manifestDigest, &m); err != nil {
		return nil, nil, err
	}
	var c spec.Config
	if err := s.store.ReadJSON(m.Config.Digest.String(), &c); err != nil {
		return nil, nil, err
	}
	return &m, &c, nil
}

// imageSize adds up an image's distinct blobs, config and manifest.
func imageSize(m *ocispec.Manifest, manifestSize int64) int64 {
	seen := map[string]bool{}
	total := manifestSize
	for _, d := range append([]ocispec.Descriptor{m.Config}, m.Layers...) {
		if !seen[d.Digest.String()] {
			seen[d.Digest.String()] = true
			total += d.Size
		}
	}
	return total
}
