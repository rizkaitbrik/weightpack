// Package daemon is wpkd's HTTP API over the local store. See package api
// for the endpoints and wire types.
package daemon

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"path/filepath"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

const maxRequestBody = 1 << 20

type server struct {
	store   *store.Store
	version string
	log     *slog.Logger
}

// NewHandler returns the daemon's API over s.
func NewHandler(s *store.Store, version string, logger *slog.Logger) http.Handler {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	srv := &server{store: s, version: version, log: logger}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", srv.health)
	mux.HandleFunc("GET /v1/blobs/{digest}", srv.blob)
	mux.HandleFunc("POST /v1/build", srv.build)
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

// build streams progress as NDJSON. The last line has a result or an error.
// A client that disconnects cancels the build.
func (s *server) build(w http.ResponseWriter, r *http.Request) {
	var req api.BuildRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("bad build request: %w", err))
		return
	}
	if !filepath.IsAbs(req.Weightfile) {
		writeError(w, http.StatusBadRequest, fmt.Errorf("weightfile must be an absolute path, got %q", req.Weightfile))
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	send := func(e api.Event) {
		_ = enc.Encode(e)
		if flusher != nil {
			flusher.Flush()
		}
	}

	s.log.Info("build started", "weightfile", req.Weightfile, "tags", req.Tags)
	res, err := build.Build(r.Context(), s.store, build.Options{Weightfile: req.Weightfile, Tags: req.Tags}, send)
	if err != nil {
		s.log.Info("build failed", "weightfile", req.Weightfile, "err", err)
		send(api.Event{Error: err.Error()})
		return
	}
	s.log.Info("build finished", "digest", res.Manifest.Digest, "new_bytes", res.NewBytes)
	send(api.Event{Result: res})
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
	raw, err := s.store.ReadBlob(manifestDigest)
	if err != nil {
		return nil, nil, err
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, nil, fmt.Errorf("manifest %s: %w", manifestDigest, err)
	}
	raw, err = s.store.ReadBlob(m.Config.Digest.String())
	if err != nil {
		return nil, nil, err
	}
	var c spec.Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, nil, fmt.Errorf("config %s: %w", m.Config.Digest, err)
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
