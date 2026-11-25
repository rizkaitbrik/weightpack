// Package hf fetches files from the Hugging Face Hub for builds.
//
// A revision (a branch, tag or commit) is resolved to a commit first, so
// what a build reads is pinned. Files are downloaded once into a cache keyed
// by repository and commit, and each is checked against what the Hub
// publishes for it: the SHA-256 of a large-file-storage file, or the git
// blob id of a small one. A file that does not match is never kept.
//
// Configuration follows huggingface_hub: HF_ENDPOINT points at a mirror,
// and HF_TOKEN, or the token `huggingface-cli login` saved, opens gated and
// private repositories.
package hf

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"
)

// DefaultEndpoint is the public Hub.
const DefaultEndpoint = "https://huggingface.co"

// Client talks to a Hub and keeps a download cache.
type Client struct {
	Endpoint string
	Token    string
	CacheDir string
	HTTP     *http.Client
	// Parallel is how many files download at once (default 4).
	Parallel int
}

// NewFromEnv returns a client configured the way huggingface_hub is, with
// its cache in cacheDir.
func NewFromEnv(cacheDir string) *Client {
	c := &Client{Endpoint: DefaultEndpoint, CacheDir: cacheDir, HTTP: http.DefaultClient}
	if e := os.Getenv("HF_ENDPOINT"); e != "" {
		c.Endpoint = strings.TrimRight(e, "/")
	}
	c.Token = os.Getenv("HF_TOKEN")
	if c.Token == "" {
		home := os.Getenv("HF_HOME")
		if home == "" {
			if h, err := os.UserHomeDir(); err == nil {
				home = filepath.Join(h, ".cache", "huggingface")
			}
		}
		if b, err := os.ReadFile(filepath.Join(home, "token")); err == nil {
			c.Token = strings.TrimSpace(string(b))
		}
	}
	return c
}

// File is one file of a repository at a commit.
type File struct {
	Path   string
	Size   int64
	SHA256 string // large-file-storage files
	BlobID string // git blob id (SHA-1), for the others
}

// Repo is a repository resolved to a commit.
type Repo struct {
	ID    string
	SHA   string
	Files map[string]File
}

// Has reports whether the repository has a file at path.
func (r *Repo) Has(p string) bool { _, ok := r.Files[p]; return ok }

type siblings struct {
	SHA      string `json:"sha"`
	Siblings []struct {
		Name   string `json:"rfilename"`
		Size   int64  `json:"size"`
		BlobID string `json:"blobId"`
		LFS    *struct {
			SHA256 string `json:"sha256"`
		} `json:"lfs"`
	} `json:"siblings"`
}

func (c *Client) request(ctx context.Context, rawURL string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	req.Header.Set("User-Agent", "weightpack")
	hc := c.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	return hc.Do(req)
}

func statusError(resp *http.Response, what string) error {
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%s is gated or private: set HF_TOKEN (or run huggingface-cli login) with an account that has access", what)
	case http.StatusNotFound:
		return fmt.Errorf("%s not found on the Hub", what)
	default:
		return fmt.Errorf("%s: Hub answered %s", what, resp.Status)
	}
}

func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}

// Resolve pins a repository revision to a commit and lists its files.
func (c *Client) Resolve(ctx context.Context, repo, rev string) (*Repo, error) {
	if rev == "" {
		rev = "main"
	}
	u := fmt.Sprintf("%s/api/models/%s/revision/%s?blobs=true", c.Endpoint, escapePath(repo), url.PathEscape(rev))
	resp, err := c.request(ctx, u)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, statusError(resp, fmt.Sprintf("hf://%s@%s", repo, rev))
	}
	var s siblings
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&s); err != nil {
		return nil, fmt.Errorf("hf://%s@%s: %w", repo, rev, err)
	}
	if len(s.SHA) != 40 {
		return nil, fmt.Errorf("hf://%s@%s: the Hub gave no commit", repo, rev)
	}
	r := &Repo{ID: repo, SHA: s.SHA, Files: map[string]File{}}
	for _, f := range s.Siblings {
		file := File{Path: f.Name, Size: f.Size, BlobID: f.BlobID}
		if f.LFS != nil {
			file.SHA256 = f.LFS.SHA256
		}
		r.Files[f.Name] = file
	}
	return r, nil
}

// Dir is where a repository's files at its commit live in the cache.
func (c *Client) Dir(r *Repo) string {
	return filepath.Join(c.CacheDir, strings.ReplaceAll(r.ID, "/", "--"), r.SHA)
}

// Fetch downloads the named files of r into the cache, skipping those
// already there, and returns the repository's directory in the cache.
// progress, if not nil, gets the bytes done and the total to download.
func (c *Client) Fetch(ctx context.Context, r *Repo, paths []string, progress func(done, total int64)) (string, error) {
	dir := c.Dir(r)
	var todo []File
	var total int64
	for _, p := range paths {
		f, ok := r.Files[p]
		if !ok {
			return "", fmt.Errorf("hf://%s has no file %s", r.ID, p)
		}
		if !safePath(p) {
			return "", fmt.Errorf("hf://%s: unsafe file path %q", r.ID, p)
		}
		if info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(p))); err == nil && info.Size() == f.Size {
			continue
		}
		todo = append(todo, f)
		total += f.Size
	}
	if len(todo) == 0 {
		return dir, nil
	}
	var mu sync.Mutex
	var done int64
	// Downloads run in parallel; the callback runs under the lock, so it is
	// never called concurrently.
	report := func(n int64) {
		if progress == nil {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		done += n
		progress(done, total)
	}
	g, gctx := errgroup.WithContext(ctx)
	parallel := c.Parallel
	if parallel <= 0 {
		parallel = 4
	}
	g.SetLimit(parallel)
	for _, f := range todo {
		g.Go(func() error { return c.download(gctx, r, f, dir, report) })
	}
	return dir, g.Wait()
}

// safePath accepts a relative path with no .. or absolute parts.
func safePath(p string) bool {
	clean := path.Clean(p)
	return p != "" && clean == p && !path.IsAbs(p) && !strings.HasPrefix(clean, "../") && clean != ".."
}

func (c *Client) download(ctx context.Context, r *Repo, f File, dir string, report func(int64)) error {
	u := fmt.Sprintf("%s/%s/resolve/%s/%s", c.Endpoint, escapePath(r.ID), r.SHA, escapePath(f.Path))
	resp, err := c.request(ctx, u)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return statusError(resp, fmt.Sprintf("hf://%s file %s", r.ID, f.Path))
	}
	final := filepath.Join(dir, filepath.FromSlash(f.Path))
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(final), ".download-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	var h hash.Hash
	var want string
	switch {
	case f.SHA256 != "":
		h, want = sha256.New(), f.SHA256
	case f.BlobID != "":
		h, want = sha1.New(), f.BlobID
		fmt.Fprintf(h, "blob %d\x00", f.Size)
	}
	w := io.Writer(tmp)
	if h != nil {
		w = io.MultiWriter(tmp, h)
	}
	n, err := io.Copy(w, &counting{r: resp.Body, report: report})
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("hf://%s file %s: %w", r.ID, f.Path, err)
	}
	if n != f.Size {
		return fmt.Errorf("hf://%s file %s: got %d bytes, the Hub lists %d", r.ID, f.Path, n, f.Size)
	}
	if h != nil {
		if got := hex.EncodeToString(h.Sum(nil)); got != want {
			return fmt.Errorf("hf://%s file %s does not match the hash the Hub publishes for it", r.ID, f.Path)
		}
	}
	return os.Rename(tmp.Name(), final)
}

type counting struct {
	r      io.Reader
	report func(int64)
}

func (c *counting) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	if n > 0 {
		c.report(int64(n))
	}
	return n, err
}
