package build

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/weightfile"
)

// fetchKind says which files of a source a build reads.
type fetchKind int

const (
	// fetchModel: safetensors weights and the base meta files (FROM, and
	// codec=xor fine-tunes).
	fetchModel fetchKind = iota
	// fetchAdapter: a PEFT adapter's config and safetensors weights.
	fetchAdapter
)

// sourceDir returns the directory a FROM, LAYER or EXPERT reads: a local
// path, or a Hugging Face repository (or a folder in it) resolved to a
// commit and fetched into the cache, only the files the build needs.
func (b *builder) sourceDir(step string, line int, src weightfile.Source, kind fetchKind) (string, error) {
	if src.Kind != weightfile.SourceHF {
		return src.Path(b.dir), nil
	}
	fail := func(format string, args ...any) error {
		return &weightfile.Error{File: b.file, Line: line, Msg: step + ": " + fmt.Sprintf(format, args...)}
	}
	if b.hf == nil {
		return "", fail("hf:// sources need wpkd's Hugging Face client")
	}
	repo, folder, rev := src.HF()
	r, err := b.hf.Resolve(b.ctx, repo, rev)
	if err != nil {
		return "", fail("%v", err)
	}
	prefix := ""
	if folder != "" {
		prefix = folder + "/"
	}
	has := func(name string) bool { return r.Has(prefix + name) }
	// level lists the files directly in the folder (not in subfolders).
	var level []string
	for p := range r.Files {
		if rest, ok := strings.CutPrefix(p, prefix); ok && !strings.Contains(rest, "/") {
			level = append(level, rest)
		}
	}
	sort.Strings(level)

	var want []string
	switch kind {
	case fetchAdapter:
		switch {
		case !has("adapter_config.json"):
			return "", fail("%s has no adapter_config.json; is it a PEFT adapter?", src)
		case !has("adapter_model.safetensors") && has("adapter_model.bin"):
			return "", fail("%s has only adapter_model.bin: pickle weights can run code when loaded, so use a safetensors adapter", src)
		case !has("adapter_model.safetensors"):
			return "", fail("%s has no adapter_model.safetensors", src)
		}
		want = []string{"adapter_config.json", "adapter_model.safetensors"}
	case fetchModel:
		var shards []string
		if has("model.safetensors.index.json") {
			// Fetch the index first: it names the shards.
			dir, err := b.hf.Fetch(b.ctx, r, []string{prefix + "model.safetensors.index.json"}, nil)
			if err != nil {
				return "", fail("%v", err)
			}
			shards, err = indexShards(filepath.Join(dir, filepath.FromSlash(prefix), "model.safetensors.index.json"))
			if err != nil {
				return "", fail("%v", err)
			}
			for _, s := range shards {
				if !has(s) {
					return "", fail("%s: the index names %s, which the repository lacks", src, s)
				}
			}
			shards = append(shards, "model.safetensors.index.json")
		} else {
			for _, f := range level {
				if strings.HasSuffix(f, ".safetensors") {
					shards = append(shards, f)
				}
			}
		}
		if len(shards) == 0 {
			for _, f := range level {
				if slices.Contains(pickleExts, path.Ext(f)) {
					return "", fail("%s has %s but no safetensors: pickle weights can run code when loaded", src, f)
				}
			}
			return "", fail("%s has no safetensors weights", src)
		}
		want = shards
		for _, m := range baseMetaFiles {
			if has(m) {
				want = append(want, m)
			}
		}
	}

	var total int64
	for i, w := range want {
		want[i] = prefix + w
		total += r.Files[want[i]].Size
	}
	b.progress(api.Event{Step: step, Message: fmt.Sprintf("%s is commit %s; %d files, %s", src, r.SHA[:12], len(want), api.FormatSize(total))})
	next := max(total/10, 256<<20)
	dir, err := b.hf.Fetch(b.ctx, r, want, func(done, total int64) {
		if done >= next && done < total {
			b.progress(api.Event{Step: step, Message: fmt.Sprintf("downloaded %s of %s", api.FormatSize(done), api.FormatSize(total))})
			next = done + max(total/10, 256<<20)
		}
	})
	if err != nil {
		return "", fail("%v", err)
	}
	return filepath.Join(dir, filepath.FromSlash(folder)), nil
}

// indexShards reads the shard file names a model.safetensors.index.json
// names.
func indexShards(p string) ([]string, error) {
	raw, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("model.safetensors.index.json: %w", err)
	}
	set := map[string]bool{}
	for _, f := range idx.WeightMap {
		if f != path.Base(f) || path.Ext(f) != ".safetensors" {
			return nil, fmt.Errorf("model.safetensors.index.json names %q, not a safetensors file beside it", f)
		}
		set[f] = true
	}
	out := make([]string, 0, len(set))
	for f := range set {
		out = append(out, f)
	}
	sort.Strings(out)
	return out, nil
}
