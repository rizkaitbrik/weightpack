package registry

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ggcr "github.com/google/go-containerregistry/pkg/registry"
	"github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2/registry/remote"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

// newRegistry starts an in-memory OCI registry and returns its host:port.
// Docker credentials are pointed at an empty directory so the tests never
// read the user's.
func newRegistry(t *testing.T) string {
	t.Helper()
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	srv := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(srv.Close)
	return strings.TrimPrefix(srv.URL, "http://")
}

func openStore(t *testing.T) *store.Store {
	t.Helper()
	s, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func buildImage(t *testing.T, s *store.Store, weightfile string, tags ...string) ocispec.Descriptor {
	t.Helper()
	res, err := build.Build(context.Background(), s, build.Options{Weightfile: weightfile, Tags: tags}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res.Manifest
}

func checkStored(t *testing.T, s *store.Store, ref string) {
	t.Helper()
	d, err := s.Resolve(ref)
	if err != nil {
		t.Fatal(err)
	}
	var m ocispec.Manifest
	var c spec.Config
	if err := s.ReadJSON(d.Digest.String(), &m); err != nil {
		t.Fatal(err)
	}
	if err := s.ReadJSON(m.Config.Digest.String(), &c); err != nil {
		t.Fatal(err)
	}
	if err := spec.CheckImage(&m, &c); err != nil {
		t.Fatalf("%s: %v", ref, err)
	}
	for _, l := range m.Layers {
		if ok, _ := s.Has(l.Digest.String()); !ok {
			t.Fatalf("%s: blob %s missing after pull", ref, l.Digest)
		}
	}
}

func TestPushPullRoundTrip(t *testing.T) {
	host := newRegistry(t)
	ctx := context.Background()
	dir := t.TempDir()
	wf := buildtest.Project(t, dir)
	buildtest.WriteAdapter(t, filepath.Join(dir, "adapters/support"), buildtest.Adapter{Rank: 4, Alpha: 4, Modules: []string{"self_attn.o_proj"}, Blocks: []int{0, 1}, Seed: 9})
	supportWF := filepath.Join(dir, "Support.Weightfile")
	if err := os.WriteFile(supportWF, []byte("FROM ./base\nLAYER support ./adapters/support\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	assistant := host + "/team/assistant:v1"
	support := host + "/team/support:v1"
	a := openStore(t)
	built := buildImage(t, a, wf, assistant)
	buildImage(t, a, supportWF, support)

	pushed, err := Push(ctx, a, assistant, Options{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pushed.Digest != built.Digest.String() || pushed.Moved != pushed.Blobs || pushed.MovedBytes != pushed.Size {
		t.Errorf("first push = %+v", pushed)
	}
	again, err := Push(ctx, a, assistant, Options{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if again.Moved != 0 {
		t.Errorf("second push sent %d blobs", again.Moved)
	}
	// The registry already holds the base, so only support's two LoRA
	// modules, its config and its manifest go up.
	sup, err := Push(ctx, a, support, Options{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sup.Moved != 4 {
		t.Errorf("support push sent %d blobs, want 4", sup.Moved)
	}

	b := openStore(t)
	var events []api.Event
	pulled, err := Pull(ctx, b, assistant, Options{Exact: true}, func(e api.Event) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	if pulled.Digest != built.Digest.String() || pulled.Moved != pulled.Blobs || pulled.Tag != assistant {
		t.Errorf("pull = %+v", pulled)
	}
	if len(events) < 2 || !strings.Contains(events[0].Message, "llama base") || !strings.Contains(events[len(events)-1].Message, "fetched") {
		t.Errorf("pull events = %+v", events)
	}
	checkStored(t, b, assistant)

	sp, err := Pull(ctx, b, support, Options{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sp.Moved != 4 {
		t.Errorf("support pull fetched %d blobs, want 4", sp.Moved)
	}
	if re, err := Pull(ctx, b, assistant, Options{}, nil); err != nil || re.Moved != 0 {
		t.Errorf("re-pull = %+v, %v", re, err)
	}

	// Pulling by digest stores the image but tags nothing.
	c := openStore(t)
	byDigest := host + "/team/assistant@" + built.Digest.String()
	pd, err := Pull(ctx, c, byDigest, Options{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pd.Tag != "" {
		t.Errorf("digest pull tagged %q", pd.Tag)
	}
	checkStored(t, c, built.Digest.String())
	if images, _ := c.Images(); len(images) != 0 {
		t.Errorf("digest pull left %d tags", len(images))
	}
}

// handmade pushes a small image built by hand: a base with one tensor and a
// one-blob layer of the given kind and media type. mutate runs after the
// digests and chain IDs are set, to break the image in a chosen way.
func handmade(t *testing.T, host, ref string, kind spec.LayerKind, mediaType string, mutate func(*ocispec.Manifest, *spec.Config)) []digest.Digest {
	t.Helper()
	ctx := context.Background()
	name, tag, _ := strings.Cut(ref, ":")
	repo, err := remote.NewRepository(host + "/" + name)
	if err != nil {
		t.Fatal(err)
	}
	repo.PlainHTTP = true
	push := func(mt string, b []byte) ocispec.Descriptor {
		d := ocispec.Descriptor{MediaType: mt, Digest: digest.FromBytes(b), Size: int64(len(b))}
		if err := repo.Push(ctx, d, bytes.NewReader(b)); err != nil {
			t.Fatal(err)
		}
		return d
	}
	baseBlob := push(spec.MediaTypeTensor, []byte("base tensor bytes"))
	baseBlob.Annotations = map[string]string{spec.AnnotationLayer: "base", spec.AnnotationTensor: "w"}
	topBlob := push(mediaType, []byte("delta bytes"))
	topBlob.Annotations = map[string]string{spec.AnnotationLayer: "top", spec.AnnotationTensor: "w"}

	cfg := spec.Config{Schema: spec.SchemaV1, Arch: spec.Arch{Family: "llama", Fingerprint: "sha256:00"},
		Layers: []spec.Layer{{ID: "base", Kind: spec.KindBase}, {ID: "top", Kind: kind, Parent: "base", Rank: 1, Alpha: 1}}}
	cfg.Layers[0].Digest = spec.LayerDigest(cfg.Layers[0], []spec.BlobRef{spec.BlobRefFromDescriptor(baseBlob)}, nil)
	cfg.Layers[0].Chain = spec.ChainID("", cfg.Layers[0].Digest)
	cfg.Layers[1].Digest = spec.LayerDigest(cfg.Layers[1], []spec.BlobRef{spec.BlobRefFromDescriptor(topBlob)}, nil)
	cfg.Layers[1].Chain = spec.ChainID(cfg.Layers[0].Chain, cfg.Layers[1].Digest)

	m := ocispec.Manifest{Versioned: specs.Versioned{SchemaVersion: 2}, MediaType: ocispec.MediaTypeImageManifest,
		ArtifactType: spec.ArtifactType, Layers: []ocispec.Descriptor{baseBlob, topBlob}}
	if mutate != nil {
		mutate(&m, &cfg)
	}
	raw, _ := json.Marshal(cfg)
	m.Config = push(spec.MediaTypeConfig, raw)
	raw, _ = json.Marshal(m)
	md := ocispec.Descriptor{MediaType: m.MediaType, Digest: digest.FromBytes(raw), Size: int64(len(raw))}
	if err := repo.PushReference(ctx, md, bytes.NewReader(raw), tag); err != nil {
		t.Fatal(err)
	}
	return []digest.Digest{baseBlob.Digest, topBlob.Digest}
}

func TestPullRefusesBadImages(t *testing.T) {
	host := newRegistry(t)
	ctx := context.Background()

	cases := []struct {
		name   string
		ref    string
		kind   spec.LayerKind
		media  string
		mutate func(*ocispec.Manifest, *spec.Config)
		opts   Options
		want   string
	}{
		{name: "other artifact", ref: "other:1", kind: spec.KindLoRA, media: spec.MediaTypeLoRA,
			mutate: func(m *ocispec.Manifest, _ *spec.Config) { m.ArtifactType = "application/vnd.example.other" },
			want:   "not a Weightpack image"},
		{name: "inconsistent", ref: "tampered:1", kind: spec.KindLoRA, media: spec.MediaTypeLoRA,
			mutate: func(_ *ocispec.Manifest, c *spec.Config) { c.Layers[1].Alpha = 64 },
			want:   "inconsistent image"},
		{name: "path-like layer id", ref: "pathy:1", kind: spec.KindLoRA, media: spec.MediaTypeLoRA,
			mutate: func(m *ocispec.Manifest, c *spec.Config) {
				c.Layers[1].ID = "../escaped"
				m.Layers[1].Annotations[spec.AnnotationLayer] = "../escaped"
			},
			want: "bad layer id"},
		{name: "lossy under exact", ref: "lossy:1", kind: spec.KindDeltaBit1, media: spec.MediaTypeDeltaBit1,
			opts: Options{Exact: true}, want: "lossy blob"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			blobs := handmade(t, host, c.ref, c.kind, c.media, c.mutate)
			s := openStore(t)
			_, err := Pull(ctx, s, host+"/"+c.ref, c.opts, nil)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want %q", err, c.want)
			}
			for _, d := range blobs {
				if ok, _ := s.Has(d.String()); ok {
					t.Errorf("refused pull still fetched blob %s", d)
				}
			}
		})
	}

	// Without --exact the lossy image is accepted.
	s := openStore(t)
	if _, err := Pull(ctx, s, host+"/lossy:1", Options{}, nil); err != nil {
		t.Fatalf("lossy pull without exact: %v", err)
	}
}

func TestReferencesNeedARegistry(t *testing.T) {
	s := openStore(t)
	for _, f := range []func() error{
		func() error { _, err := Push(context.Background(), s, "assistant:v1", Options{}, nil); return err },
		func() error { _, err := Pull(context.Background(), s, "assistant", Options{}, nil); return err },
	} {
		if err := f(); err == nil || !strings.Contains(err.Error(), "names no registry") {
			t.Errorf("err = %v", err)
		}
	}
}

func TestPullUnknownTag(t *testing.T) {
	host := newRegistry(t)
	if _, err := Pull(context.Background(), openStore(t), host+"/nothing:here", Options{}, nil); err == nil {
		t.Fatal("pulled a tag that does not exist")
	}
}

func TestIsLoopback(t *testing.T) {
	for host, want := range map[string]bool{
		"localhost": true, "localhost:5000": true, "127.0.0.1:41234": true,
		"registry.example.com": false, "10.0.0.1:5000": false, "ghcr.io": false,
	} {
		if got := isLoopback(host); got != want {
			t.Errorf("isLoopback(%q) = %v", host, got)
		}
	}
}

// A reference without a tag means latest on both sides: the local image
// tagged :latest is pushed under :latest, so pulling the same reference
// finds it.
func TestUntaggedPushMeansLatest(t *testing.T) {
	host := newRegistry(t)
	ctx := context.Background()
	ref := host + "/team/model"
	a := openStore(t)
	built := buildImage(t, a, buildtest.Project(t, t.TempDir()), ref)

	pushed, err := Push(ctx, a, ref, Options{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pushed.Ref != ref+":latest" {
		t.Errorf("push ref = %q, want %q", pushed.Ref, ref+":latest")
	}
	pulled, err := Pull(ctx, openStore(t), ref, Options{}, nil)
	if err != nil {
		t.Fatalf("pull of the pushed reference: %v", err)
	}
	if pulled.Digest != built.Digest.String() {
		t.Errorf("pulled %s, pushed %s", pulled.Digest, built.Digest)
	}
}
