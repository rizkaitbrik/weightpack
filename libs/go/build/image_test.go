package build

import (
	"context"
	"io"
	"log"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ggcr "github.com/google/go-containerregistry/pkg/registry"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/build/buildtest"
	"github.com/rizkaitbrik/weightpack/libs/go/registry"
	"github.com/rizkaitbrik/weightpack/libs/go/spec"
	"github.com/rizkaitbrik/weightpack/libs/go/store"
)

// world is a project directory, a publishing store A, a registry and a
// fresh store B that builds from what A pushed.
type world struct {
	dir  string
	host string
	a, b *store.Store
}

func newWorld(t *testing.T) *world {
	t.Helper()
	t.Setenv("DOCKER_CONFIG", t.TempDir())
	srv := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	buildtest.Project(t, dir)
	a, err := store.Open(filepath.Join(dir, "a"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := store.Open(filepath.Join(dir, "b"))
	if err != nil {
		t.Fatal(err)
	}
	return &world{dir: dir, host: strings.TrimPrefix(srv.URL, "http://"), a: a, b: b}
}

func pullInto(s *store.Store) PullFunc {
	return func(ctx context.Context, ref string, progress func(api.Event)) error {
		_, err := registry.Pull(ctx, s, ref, registry.Options{}, progress)
		return err
	}
}

// build writes a Weightfile into the project directory and builds it into
// s, pulling missing images into s.
func (w *world) build(t *testing.T, s *store.Store, name, body string, tags ...string) (*api.BuildResult, []api.Event, error) {
	t.Helper()
	p := filepath.Join(w.dir, name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var events []api.Event
	res, err := Build(context.Background(), s, Options{Weightfile: p, Tags: tags, Pull: pullInto(s)}, func(e api.Event) {
		events = append(events, e)
	})
	return res, events, err
}

// publish builds body in A, tags it host/team/<name>:v1 and pushes it.
func (w *world) publish(t *testing.T, name, body string) string {
	t.Helper()
	ref := w.host + "/team/" + name + ":v1"
	if _, _, err := w.build(t, w.a, name+".Weightfile", body, ref); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Push(context.Background(), w.a, ref, registry.Options{}, nil); err != nil {
		t.Fatal(err)
	}
	return ref
}

func mustBuild(t *testing.T, res *api.BuildResult, err error) *api.BuildResult {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func layerByID(c *spec.Config, id string) spec.Layer {
	for _, l := range c.Layers {
		if l.ID == id {
			return l
		}
	}
	return spec.Layer{}
}

func TestFromRegistryImageMatchesALocalBuild(t *testing.T) {
	w := newWorld(t)
	local, _, err := w.build(t, w.a, "Local.Weightfile", "FROM ./base\nLAYER french ./adapters/french\n")
	local = mustBuild(t, local, err)
	base := w.publish(t, "base", "FROM ./base\n")

	for _, from := range []string{base, "oci://" + base} {
		res, events, err := w.build(t, w.b, "Remote.Weightfile", "FROM "+from+"\nLAYER french ./adapters/french\n")
		res = mustBuild(t, res, err)
		if res.Manifest.Digest != local.Manifest.Digest {
			t.Errorf("FROM %s: digest %s, a local FROM gives %s", from, res.Manifest.Digest, local.Manifest.Digest)
		}
		pulled := false
		for _, e := range events {
			pulled = pulled || strings.HasPrefix(e.Message, "pulling ")
		}
		if from == base && !pulled {
			t.Error("the first build did not pull the base")
		}
		if from != base && pulled {
			t.Error("the second build pulled a base the store already had")
		}
	}
}

func TestLayerFromImageChecksTheChain(t *testing.T) {
	w := newWorld(t)
	base := w.publish(t, "base", "FROM ./base\n")
	french := w.publish(t, "french", "FROM ./base\nLAYER french ./adapters/french\n")
	legal := w.publish(t, "legal", "FROM ./base\nLAYER french ./adapters/french\nLAYER legal ./adapters/legal\n")
	_, ac := loadImage(t, w.a, legal)

	// legal was trained on base → french: stacking it on base alone is refused.
	_, _, err := w.build(t, w.b, "Wrong.Weightfile", "FROM "+base+"\nLAYER legal oci://"+legal+"\n")
	if err == nil || !strings.Contains(err.Error(), "was built on base → french") || !strings.Contains(err.Error(), "here it would sit on base (chain") {
		t.Fatalf("wrong chain: err = %v", err)
	}

	// On the chain it was built on, it gives back the very same image.
	res, _, err := w.build(t, w.b, "Right.Weightfile", "FROM "+french+"\nLAYER legal oci://"+legal+"\n")
	res = mustBuild(t, res, err)
	if res.Manifest.Digest.String() != mustResolve(t, w.a, legal) {
		t.Errorf("rebuilt legal image %s differs from the published one", res.Manifest.Digest)
	}

	// Under another id the layer keeps its digest and chain ID.
	res, _, err = w.build(t, w.b, "Renamed.Weightfile", "FROM "+french+"\nLAYER law oci://"+legal+"\n")
	res = mustBuild(t, res, err)
	law, want := layerByID(res.Config, "law"), layerByID(ac, "legal")
	if law.Digest != want.Digest || law.Chain != want.Chain || law.Rank != want.Rank || law.Alpha != want.Alpha {
		t.Errorf("law = %+v, want legal's digest, chain, rank and alpha %+v", law, want)
	}
}

func TestExpertsFromImages(t *testing.T) {
	w := newWorld(t)
	french := w.publish(t, "french", "FROM ./base\nLAYER french ./adapters/french\n")
	legal := w.publish(t, "legal", "FROM ./base\nLAYER french ./adapters/french\nLAYER legal ./adapters/legal\n")
	res, _, err := w.build(t, w.b, "Mix.Weightfile", "FROM "+french+"\nMIX domain targets=mlp blocks=1 top_k=1\n"+
		"  EXPERT legal oci://"+legal+"\n  EXPERT medical ./adapters/medical\n")
	res = mustBuild(t, res, err)

	local, _, err := w.build(t, w.a, "LocalMix.Weightfile", "FROM ./base\nLAYER french ./adapters/french\n"+
		"MIX domain targets=mlp blocks=1 top_k=1\n  EXPERT legal ./adapters/legal\n  EXPERT medical ./adapters/medical\n")
	local = mustBuild(t, local, err)
	if res.Manifest.Digest != local.Manifest.Digest {
		t.Errorf("mix with an expert from an image %s differs from the all-local mix %s", res.Manifest.Digest, local.Manifest.Digest)
	}
}

func TestImageSourcesRefuse(t *testing.T) {
	w := newWorld(t)
	base := w.publish(t, "base", "FROM ./base\n")
	french := w.publish(t, "french", "FROM ./base\nLAYER french ./adapters/french\n")
	assistant := w.host + "/team/assistant:v1"
	if _, _, err := w.build(t, w.a, "Weightfile", buildtest.Assistant, assistant); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Push(context.Background(), w.a, assistant, registry.Options{}, nil); err != nil {
		t.Fatal(err)
	}
	cases := map[string]struct{ body, want string }{
		"from a mix":     {"FROM " + assistant + "\nLAYER x ./adapters/legal\n", "ends in MIX domain"},
		"layer is a mix": {"FROM " + french + "\nLAYER x oci://" + assistant + "\n", "ends in MIX domain"},
		"layer is base":  {"FROM " + base + "\nLAYER x oci://" + base + "\n", "only a base"},
		"id clash":       {"FROM " + french + "\nLAYER french ./adapters/french\n", "already has a layer french"},
		"no registry":    {"FROM nothing:here\n", "names no registry"},
		"unknown tag":    {"FROM " + w.host + "/team/nothing:v1\n", "FROM"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, err := w.build(t, w.b, "Bad.Weightfile", c.body, "bad")
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want %q", err, c.want)
			}
			if _, err := w.b.Resolve("bad"); err == nil {
				t.Error("a failed build tagged an image")
			}
		})
	}
}

func TestFromImageInheritsTemplateAndLicense(t *testing.T) {
	w := newWorld(t)
	// A local tag works as FROM without any registry.
	parent, _, err := w.build(t, w.a, "Parent.Weightfile", "FROM ./base\nTEMPLATE ./chat.jinja\nLICENSE MIT\nLABEL team=nlp\n", "parent")
	parent = mustBuild(t, parent, err)
	pm, _ := loadImage(t, w.a, "parent")

	child, _, err := w.build(t, w.a, "Child.Weightfile", "FROM parent\nLAYER french ./adapters/french\n", "child")
	mustBuild(t, child, err)
	cm, _ := loadImage(t, w.a, "child")
	if cm.Annotations[ocispec.AnnotationLicenses] != "MIT" || cm.Annotations["team"] != "" {
		t.Errorf("child annotations = %v, want the license but not the label", cm.Annotations)
	}
	if templates(cm) != templates(pm) || templates(cm) == "" {
		t.Errorf("child template %q, parent %q", templates(cm), templates(pm))
	}

	own, _, err := w.build(t, w.a, "Own.Weightfile", "FROM parent\nLICENSE Apache-2.0\n", "own")
	mustBuild(t, own, err)
	om, _ := loadImage(t, w.a, "own")
	if om.Annotations[ocispec.AnnotationLicenses] != "Apache-2.0" {
		t.Errorf("own license = %v", om.Annotations)
	}
}

func templates(m *ocispec.Manifest) string {
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationTitle] == titleTemplate && d.Annotations[spec.AnnotationLayer] == "" {
			return d.Digest.String()
		}
	}
	return ""
}

func mustResolve(t *testing.T, s *store.Store, ref string) string {
	t.Helper()
	d, err := s.Resolve(ref)
	if err != nil {
		t.Fatal(err)
	}
	return d.Digest.String()
}

func loadImage(t *testing.T, s *store.Store, ref string) (*ocispec.Manifest, *spec.Config) {
	t.Helper()
	var m ocispec.Manifest
	var c spec.Config
	if err := s.ReadJSON(mustResolve(t, s, ref), &m); err != nil {
		t.Fatal(err)
	}
	if err := s.ReadJSON(m.Config.Digest.String(), &c); err != nil {
		t.Fatal(err)
	}
	return &m, &c
}

// addLayer stores a copy of ref with an extra layer "other": a copy of
// layer copyID placed on parentID. The result passes CheckImage but is not
// one stack (with a mix on top).
func addLayer(t *testing.T, s *store.Store, ref, tag, copyID, parentID string) {
	t.Helper()
	m, c := loadImage(t, s, ref)
	other := layerByID(c, copyID)
	other.ID, other.Parent = "other", parentID
	other.Chain = spec.ChainID(layerByID(c, parentID).Chain, other.Digest)
	c.Layers = append(c.Layers, other)
	for _, d := range m.Layers {
		if d.Annotations[spec.AnnotationLayer] == copyID {
			nd := d
			nd.Annotations = map[string]string{spec.AnnotationLayer: "other", spec.AnnotationTensor: d.Annotations[spec.AnnotationTensor]}
			m.Layers = append(m.Layers, nd)
		}
	}
	if err := spec.CheckImage(m, c); err != nil {
		t.Fatalf("the altered image should pass CheckImage: %v", err)
	}
	cd, _, err := s.PutJSON(spec.MediaTypeConfig, c)
	if err != nil {
		t.Fatal(err)
	}
	m.Config = cd
	md, _, err := s.PutJSON(ocispec.MediaTypeImageManifest, m)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Tag(tag, md); err != nil {
		t.Fatal(err)
	}
}

func TestBranchedImagesAreRefused(t *testing.T) {
	w := newWorld(t)
	if _, _, err := w.build(t, w.a, "French.Weightfile", "FROM ./base\nLAYER french ./adapters/french\n", "french"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := w.build(t, w.a, "Weightfile", buildtest.Assistant, "assistant"); err != nil {
		t.Fatal(err)
	}
	addLayer(t, w.a, "french", "branched", "french", "base")       // beside french
	addLayer(t, w.a, "assistant", "on-expert", "medical", "legal") // above an expert
	cases := map[string]struct{ body, want string }{
		"from a branch":       {"FROM branched\nLAYER legal ./adapters/legal\n", "french and other both sit on base"},
		"layer from a branch": {"FROM ./base\nLAYER x oci://branched\n", "french and other both sit on base"},
		"above an expert":     {"FROM ./base\nLAYER french ./adapters/french\nLAYER x oci://on-expert\n", "layers other sit above a MIX or an expert"},
	}
	for name, c := range cases {
		_, _, err := w.build(t, w.a, "Bad.Weightfile", c.body, "bad")
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
	}
}
