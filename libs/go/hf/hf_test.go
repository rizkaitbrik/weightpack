package hf_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rizkaitbrik/weightpack/libs/go/hf"
	"github.com/rizkaitbrik/weightpack/libs/go/hf/hftest"
)

func client(hub *hftest.Hub, t *testing.T) *hf.Client {
	return &hf.Client{Endpoint: hub.URL, CacheDir: t.TempDir(), HTTP: hub.Client()}
}

func TestResolveAndFetch(t *testing.T) {
	hub := hftest.New(t)
	sha := hub.Add("acme/tiny", map[string][]byte{
		"config.json":            []byte(`{"model_type":"llama"}`),
		"model.safetensors":      []byte("weights-bytes"),
		"adapter/adapter_config": []byte("{}"),
	})
	c := client(hub, t)
	ctx := context.Background()

	r, err := c.Resolve(ctx, "acme/tiny", "main")
	if err != nil {
		t.Fatal(err)
	}
	if r.SHA != sha || !r.Has("model.safetensors") || r.Files["model.safetensors"].SHA256 == "" || r.Files["config.json"].BlobID == "" {
		t.Fatalf("repo = %+v", r)
	}
	var lastDone, lastTotal int64
	dir, err := c.Fetch(ctx, r, []string{"config.json", "model.safetensors", "adapter/adapter_config"}, func(d, t int64) { lastDone, lastTotal = d, t })
	if err != nil {
		t.Fatal(err)
	}
	if lastDone != lastTotal || lastTotal == 0 {
		t.Errorf("progress ended at %d of %d", lastDone, lastTotal)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "adapter", "adapter_config")); string(b) != "{}" {
		t.Errorf("nested file = %q", b)
	}
	// A second fetch comes from the cache.
	before := hub.Downloads.Load()
	if _, err := c.Fetch(ctx, r, []string{"config.json", "model.safetensors"}, nil); err != nil {
		t.Fatal(err)
	}
	if hub.Downloads.Load() != before {
		t.Error("cached files were downloaded again")
	}
	// Resolving by commit works as well as by branch.
	if r2, err := c.Resolve(ctx, "acme/tiny", sha); err != nil || r2.SHA != sha {
		t.Errorf("resolve by commit: %v", err)
	}
}

func TestFetchRejectsCorruptFiles(t *testing.T) {
	hub := hftest.New(t)
	hub.Add("acme/tiny", map[string][]byte{"model.safetensors": []byte("big weights"), "config.json": []byte("{}")})
	for _, p := range []string{"model.safetensors", "config.json"} {
		hub.Corrupt("acme/tiny", p)
	}
	c := client(hub, t)
	r, err := c.Resolve(context.Background(), "acme/tiny", "")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"model.safetensors", "config.json"} {
		_, err := c.Fetch(context.Background(), r, []string{p}, nil)
		if err == nil || !strings.Contains(err.Error(), "does not match the hash") {
			t.Errorf("%s: err = %v", p, err)
		}
		if _, err := os.Stat(filepath.Join(c.Dir(r), p)); !os.IsNotExist(err) {
			t.Errorf("%s: a corrupt download was kept", p)
		}
	}
}

func TestTokensAndErrors(t *testing.T) {
	hub := hftest.New(t)
	hub.Token = "secret"
	hub.Add("acme/gated", map[string][]byte{"config.json": []byte("{}")})
	c := client(hub, t)
	ctx := context.Background()
	if _, err := c.Resolve(ctx, "acme/gated", ""); err == nil || !strings.Contains(err.Error(), "gated or private") {
		t.Errorf("no token: %v", err)
	}
	c.Token = "secret"
	if _, err := c.Resolve(ctx, "acme/gated", ""); err != nil {
		t.Errorf("with token: %v", err)
	}
	if _, err := c.Resolve(ctx, "acme/missing", ""); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("missing repo: %v", err)
	}
	r, _ := c.Resolve(ctx, "acme/gated", "")
	if _, err := c.Fetch(ctx, r, []string{"nope.json"}, nil); err == nil {
		t.Error("fetching a file the repo lacks succeeded")
	}
}

func TestNewFromEnv(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "token"), []byte("hf_saved\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_HOME", home)
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_ENDPOINT", "https://mirror.example.com/")
	c := hf.NewFromEnv("/cache")
	if c.Token != "hf_saved" || c.Endpoint != "https://mirror.example.com" || c.CacheDir != "/cache" {
		t.Errorf("client = %+v", c)
	}
	t.Setenv("HF_TOKEN", "hf_env")
	if hf.NewFromEnv("").Token != "hf_env" {
		t.Error("HF_TOKEN should win over the saved token")
	}
}
