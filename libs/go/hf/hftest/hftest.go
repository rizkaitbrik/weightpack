// Package hftest serves a fake Hugging Face Hub for tests: the revision API
// (with sizes, git blob ids and SHA-256s, as the real one lists them) and
// file downloads.
package hftest

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// Hub is a fake Hub. Files ending in .safetensors are listed as large-file
// storage (with a SHA-256), the rest with a git blob id.
type Hub struct {
	*httptest.Server
	// Token, if set, must come as a Bearer token on every request.
	Token string
	// Downloads counts file downloads.
	Downloads atomic.Int64

	mu      sync.Mutex
	repos   map[string]repo
	corrupt map[string]bool // repo + "/" + path
}

type repo struct {
	sha   string
	files map[string][]byte
}

// New starts a fake Hub.
func New(t testing.TB) *Hub {
	t.Helper()
	h := &Hub{repos: map[string]repo{}, corrupt: map[string]bool{}}
	h.Server = httptest.NewServer(http.HandlerFunc(h.serve))
	t.Cleanup(h.Close)
	return h
}

// Add publishes files under repo ("org/name") and returns the commit.
func (h *Hub) Add(id string, files map[string][]byte) string {
	names := make([]string, 0, len(files))
	for n := range files {
		names = append(names, n)
	}
	sort.Strings(names)
	sum := sha1.New()
	for _, n := range names {
		fmt.Fprintf(sum, "%s\x00%x\x00", n, sha256.Sum256(files[n]))
	}
	sha := hex.EncodeToString(sum.Sum(nil))
	h.mu.Lock()
	h.repos[id] = repo{sha: sha, files: files}
	h.mu.Unlock()
	return sha
}

// AddDir publishes every file under dir.
func (h *Hub) AddDir(t testing.TB, id, dir string) string {
	t.Helper()
	files := map[string][]byte{}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		b, err := os.ReadFile(p)
		files[filepath.ToSlash(rel)] = b
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return h.Add(id, files)
}

// Corrupt makes downloads of one file serve wrong bytes.
func (h *Hub) Corrupt(id, path string) {
	h.mu.Lock()
	h.corrupt[id+"/"+path] = true
	h.mu.Unlock()
}

func gitBlobID(b []byte) string {
	s := sha1.New()
	fmt.Fprintf(s, "blob %d\x00", len(b))
	s.Write(b)
	return hex.EncodeToString(s.Sum(nil))
}

func (h *Hub) serve(w http.ResponseWriter, r *http.Request) {
	if h.Token != "" && r.Header.Get("Authorization") != "Bearer "+h.Token {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if rest, ok := strings.CutPrefix(r.URL.Path, "/api/models/"); ok {
		parts := strings.Split(rest, "/")
		if len(parts) != 4 || parts[2] != "revision" {
			http.NotFound(w, r)
			return
		}
		rp, ok := h.repos[parts[0]+"/"+parts[1]]
		if !ok || (parts[3] != "main" && parts[3] != rp.sha) {
			http.NotFound(w, r)
			return
		}
		type lfs struct {
			SHA256 string `json:"sha256"`
		}
		type sibling struct {
			Name   string `json:"rfilename"`
			Size   int    `json:"size"`
			BlobID string `json:"blobId"`
			LFS    *lfs   `json:"lfs,omitempty"`
		}
		out := struct {
			SHA      string    `json:"sha"`
			Siblings []sibling `json:"siblings"`
		}{SHA: rp.sha}
		for name, b := range rp.files {
			s := sibling{Name: name, Size: len(b), BlobID: gitBlobID(b)}
			if strings.HasSuffix(name, ".safetensors") {
				sum := sha256.Sum256(b)
				s.LFS = &lfs{SHA256: hex.EncodeToString(sum[:])}
			}
			out.Siblings = append(out.Siblings, s)
		}
		_ = json.NewEncoder(w).Encode(out)
		return
	}
	// /org/name/resolve/<sha>/<path...>
	parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 5)
	if len(parts) != 5 || parts[2] != "resolve" {
		http.NotFound(w, r)
		return
	}
	id := parts[0] + "/" + parts[1]
	rp, ok := h.repos[id]
	b, found := rp.files[parts[4]]
	if !ok || parts[3] != rp.sha || !found {
		http.NotFound(w, r)
		return
	}
	h.Downloads.Add(1)
	if h.corrupt[id+"/"+parts[4]] {
		b = append([]byte{}, b...)
		b[len(b)/2] ^= 0xFF
	}
	_, _ = w.Write(b)
}
