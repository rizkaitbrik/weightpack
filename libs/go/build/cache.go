package build

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

// CacheDir is where the build cache lives inside a store.
func CacheDir(root string) string { return filepath.Join(root, "cache", "build") }

// buildCache maps what a blob was made from to the blob it became, so a
// rebuild of unchanged sources neither reads nor hashes them again.
//
// A source file is identified by its device, inode, size and its
// modification and change times, as git's index does: the change time
// moves on any write and cannot be set back, so an edited file never looks
// unchanged. An entry only counts while its blob is still in the store.
type buildCache struct {
	dir   string // "" disables the cache
	files map[string]string
	hits  int
}

func newBuildCache(dir string) *buildCache {
	return &buildCache{dir: dir, files: map[string]string{}}
}

// fileKey identifies a file's current contents without reading it, or ""
// when it cannot (then nothing is cached).
func (c *buildCache) fileKey(path string) string {
	if k, ok := c.files[path]; ok {
		return k
	}
	k := ""
	if info, err := os.Stat(path); err == nil {
		if st, ok := info.Sys().(*syscall.Stat_t); ok {
			k = fmt.Sprintf("%d:%d:%d:%d:%d", st.Dev, st.Ino, info.Size(), info.ModTime().UnixNano(), ctimeNanos(st))
		}
	}
	c.files[path] = k
	return k
}

func hashKey(parts ...string) string {
	sum := sha256.Sum256([]byte(strings.Join(parts, "\n")))
	return hex.EncodeToString(sum[:])
}

// tensorsKey is the key of a blob made of these tensors, or "".
func (c *buildCache) tensorsKey(refs []tensorRef, mediaType string) string {
	if c.dir == "" {
		return ""
	}
	sorted := make([]tensorRef, len(refs))
	copy(sorted, refs)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].name < sorted[j].name })
	parts := []string{"tensors", mediaType}
	for _, r := range sorted {
		fk := c.fileKey(r.file.Path)
		if fk == "" {
			return ""
		}
		parts = append(parts, fmt.Sprintf("%s|%s|%v|%s|%d|%d", r.name, r.t.DType, r.t.Shape, fk, r.t.Offset, r.t.Size))
	}
	return hashKey(parts...)
}

// fileBlobKey is the key of a blob that is a file's bytes, or "".
func (c *buildCache) fileBlobKey(path string) string {
	if c.dir == "" {
		return ""
	}
	if fk := c.fileKey(path); fk != "" {
		return hashKey("file", fk)
	}
	return ""
}

type cachedBlob struct {
	Digest string `json:"digest"`
	Size   int64  `json:"size"`
}

func (c *buildCache) path(key string) string {
	return filepath.Join(c.dir, key[:2], key+".json")
}

// load reads an entry into v, reporting whether there was one.
func (c *buildCache) load(key string, v any) bool {
	if key == "" {
		return false
	}
	raw, err := os.ReadFile(c.path(key))
	return err == nil && json.Unmarshal(raw, v) == nil
}

// store writes an entry atomically; failures only cost a later rebuild.
func (c *buildCache) store(key string, v any) {
	if key == "" {
		return
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return
	}
	p := c.path(key)
	if os.MkdirAll(filepath.Dir(p), 0o755) != nil {
		return
	}
	tmp, err := os.CreateTemp(filepath.Dir(p), ".entry-*")
	if err != nil {
		return
	}
	_, werr := tmp.Write(raw)
	if cerr := tmp.Close(); werr != nil || cerr != nil {
		_ = os.Remove(tmp.Name())
		return
	}
	if os.Rename(tmp.Name(), p) != nil {
		_ = os.Remove(tmp.Name())
	}
}

// get returns the blob a key became, if the store still has it.
func (c *buildCache) get(s *store.Store, key string) (ocispec.Descriptor, bool) {
	var e cachedBlob
	if !c.load(key, &e) {
		return ocispec.Descriptor{}, false
	}
	if ok, err := s.Has(e.Digest); err != nil || !ok {
		return ocispec.Descriptor{}, false
	}
	c.hits++
	return ocispec.Descriptor{Digest: digest.Digest(e.Digest), Size: e.Size}, true
}

func (c *buildCache) set(key string, d ocispec.Descriptor) {
	c.store(key, cachedBlob{Digest: d.Digest.String(), Size: d.Size})
}

type cachedTensor struct {
	Tensor string `json:"tensor"`
	Digest string `json:"digest"`
	Size   int64  `json:"size"`
}

// cachedXOR is what an XOR import gave: its delta blobs and counts.
type cachedXOR struct {
	Tensors   []cachedTensor `json:"tensors"`
	Unchanged int64          `json:"unchanged"`
	Raw       int64          `json:"raw"`
}

// xorKey is the key of the deltas these fine-tune files give over a parent
// chain, or "".
func (c *buildCache) xorKey(parentChain string, files []string) string {
	if c.dir == "" {
		return ""
	}
	sorted := append([]string{}, files...)
	sort.Strings(sorted)
	parts := []string{"xor", parentChain}
	for _, f := range sorted {
		fk := c.fileKey(f)
		if fk == "" {
			return ""
		}
		parts = append(parts, filepath.Base(f)+"|"+fk)
	}
	return hashKey(parts...)
}

// getXOR returns a cached XOR import if the store has all its blobs.
func (c *buildCache) getXOR(s *store.Store, key string) (cachedXOR, bool) {
	var e cachedXOR
	if !c.load(key, &e) || e.Tensors == nil {
		return cachedXOR{}, false
	}
	for _, t := range e.Tensors {
		if ok, err := s.Has(t.Digest); err != nil || !ok {
			return cachedXOR{}, false
		}
	}
	c.hits += len(e.Tensors)
	return e, true
}

// PruneCache removes cache entries whose blobs the store no longer has, and
// returns how many it removed.
func PruneCache(root string, s *store.Store) (int, error) {
	removed := 0
	err := filepath.WalkDir(CacheDir(root), func(p string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return filepath.SkipDir
			}
			return err
		}
		if d.IsDir() || filepath.Ext(p) != ".json" {
			return nil
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		var digests []string
		var single cachedBlob
		var multi cachedXOR
		switch {
		case json.Unmarshal(raw, &multi) == nil && multi.Tensors != nil:
			for _, t := range multi.Tensors {
				digests = append(digests, t.Digest)
			}
		case json.Unmarshal(raw, &single) == nil && single.Digest != "":
			digests = []string{single.Digest}
		}
		for _, dg := range digests {
			if ok, err := s.Has(dg); err != nil || !ok {
				if os.Remove(p) == nil {
					removed++
				}
				break
			}
		}
		return nil
	})
	return removed, err
}
