package build

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

func cacheHits(events []api.Event) string {
	for _, e := range events {
		if e.Step == "cache" {
			return e.Message
		}
	}
	return ""
}

func (f *fixture) buildOpts(t *testing.T, name, body string, opts Options) *api.BuildResult {
	t.Helper()
	opts.Weightfile = f.weightfile(t, name, body)
	f.events = nil
	res, err := Build(context.Background(), f.store, opts, func(e api.Event) { f.events = append(f.events, e) })
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestBuildCacheReusesAndInvalidates(t *testing.T) {
	f := newFixture(t)
	body := "FROM ./base\nLAYER french ./adapters/french\n"
	first := f.buildOpts(t, "W", body, Options{})
	if hits := cacheHits(f.events); hits != "" {
		t.Errorf("a cold build reported cache hits: %s", hits)
	}

	// A rebuild reuses every blob: 17 base blobs and 4 LoRA modules.
	again := f.buildOpts(t, "W", body, Options{})
	if again.Manifest.Digest != first.Manifest.Digest {
		t.Fatal("a cached rebuild changed the digest")
	}
	if hits := cacheHits(f.events); !strings.Contains(hits, "reused 21 blobs") {
		t.Errorf("rebuild: %q", hits)
	}

	// NoCache reads everything again, to the same result.
	if nc := f.buildOpts(t, "W", body, Options{NoCache: true}); nc.Manifest.Digest != first.Manifest.Digest || cacheHits(f.events) != "" {
		t.Error("NoCache build differed or reported hits")
	}

	// Rewriting the adapter with other values of the same size, and setting
	// its modification time back, still misses: the change time moved.
	weights := filepath.Join(f.dir, "adapters/french/adapter_model.safetensors")
	info, _ := os.Stat(weights)
	buildtest.WriteAdapter(t, filepath.Join(f.dir, "adapters/french"), buildtest.Adapter{Rank: 8, Alpha: 16, Modules: buildtest.Attn, Blocks: []int{0, 1}, Seed: 99})
	if err := os.Chtimes(weights, time.Now(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if after, _ := os.Stat(weights); after.Size() != info.Size() {
		t.Fatal("test needs a same-size rewrite")
	}
	changed := f.buildOpts(t, "W", body, Options{})
	fresh := f.buildOpts(t, "W", body, Options{NoCache: true})
	if changed.Manifest.Digest == first.Manifest.Digest || changed.Manifest.Digest != fresh.Manifest.Digest {
		t.Errorf("edited adapter: cached build %s, uncached %s, before %s", changed.Manifest.Digest, fresh.Manifest.Digest, first.Manifest.Digest)
	}

	// A blob gone from the store invalidates its entry.
	m, _ := f.load(t, changed.Manifest.Digest.String())
	var victim string
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] == "base" && d.MediaType == spec.MediaTypeTensor {
			victim = d.Digest.String()
			break
		}
	}
	if err := f.store.Delete(victim); err != nil {
		t.Fatal(err)
	}
	if rebuilt := f.buildOpts(t, "W", body, Options{}); rebuilt.Manifest.Digest != changed.Manifest.Digest {
		t.Error("rebuild after a deleted blob changed the digest")
	}
	if ok, _ := f.store.Has(victim); !ok {
		t.Error("the deleted blob was not written again")
	}
	if n, err := PruneCache(f.store.Root(), f.store); err != nil || n != 0 {
		t.Errorf("PruneCache on a whole store removed %d entries: %v", n, err)
	}
}

func TestBuildCacheSkipsTheWorkerForXOR(t *testing.T) {
	f := newFixture(t)
	writeFinetune(t, filepath.Join(f.dir, "finance"), 7)
	body := "FROM ./base\nLAYER finance ./finance codec=xor\n"
	w := &xorWorker{}
	first := f.buildOpts(t, "W", body, Options{Worker: w})
	w.req = nil
	again := f.buildOpts(t, "W", body, Options{Worker: w})
	if w.req != nil {
		t.Error("a cached XOR import called the worker")
	}
	if again.Manifest.Digest != first.Manifest.Digest {
		t.Error("a cached XOR import changed the digest")
	}
}

func TestPruneCacheDropsStaleEntries(t *testing.T) {
	f := newFixture(t)
	res := f.buildOpts(t, "W", "FROM ./base\n", Options{})
	m, _ := f.load(t, res.Manifest.Digest.String())
	for _, d := range m.Layers {
		_ = f.store.Delete(d.Digest.String())
	}
	n, err := PruneCache(f.store.Root(), f.store)
	if err != nil || n == 0 {
		t.Errorf("PruneCache removed %d entries: %v", n, err)
	}
}
