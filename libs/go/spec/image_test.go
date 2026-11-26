package spec

import (
	"strings"
	"testing"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func dg(c string) digest.Digest { return digest.Digest("sha256:" + strings.Repeat(c, 64)) }

func blob(layer, tensor, mediaType string, d digest.Digest) ocispec.Descriptor {
	a := map[string]string{AnnotationLayer: layer}
	if tensor != "" {
		a[AnnotationTensor] = tensor
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: d, Size: 1, Annotations: a}
}

// builtImage returns a consistent manifest and config: a base, a stacked
// LoRA and a mix of two LoRA experts.
func builtImage() (*ocispec.Manifest, *Config) {
	m := &ocispec.Manifest{
		ArtifactType: ArtifactType,
		Config:       ocispec.Descriptor{MediaType: MediaTypeConfig, Digest: dg("c"), Size: 1},
		Layers: []ocispec.Descriptor{
			blob("base", "w", MediaTypeTensor, dg("1")),
			blob("base", "", MediaTypeTensor, dg("2")),
			{MediaType: MediaTypeMeta, Digest: dg("3"), Size: 1, Annotations: map[string]string{AnnotationLayer: "base", AnnotationTitle: "tokenizer.json"}},
			blob("french", "w", MediaTypeLoRA, dg("4")),
			blob("legal", "w", MediaTypeLoRA, dg("5")),
			blob("medical", "w", MediaTypeLoRA, dg("6")),
			{MediaType: MediaTypeMeta, Digest: dg("7"), Size: 1, Annotations: map[string]string{AnnotationTitle: "chat_template.jinja"}},
		},
	}
	c := &Config{
		Schema: SchemaV1,
		Layers: []Layer{
			{ID: "base", Kind: KindBase},
			{ID: "french", Kind: KindLoRA, Parent: "base", Rank: 8, Alpha: 16},
			{ID: "domain", Kind: KindMix, Parent: "french", TopK: 1, Router: "arrow", Experts: []string{"legal", "medical"}},
			{ID: "legal", Kind: KindLoRA, Parent: "french", Rank: 4, Alpha: 8},
			{ID: "medical", Kind: KindLoRA, Parent: "french", Rank: 4, Alpha: 8, RSLoRA: true},
		},
	}
	seal(m, c)
	return m, c
}

// seal fills in layer digests and chain IDs the way a build does.
func seal(m *ocispec.Manifest, c *Config) {
	blobs := map[string][]BlobRef{}
	for _, d := range m.Layers {
		if id := d.Annotations[AnnotationLayer]; id != "" {
			blobs[id] = append(blobs[id], BlobRefFromDescriptor(d))
		}
	}
	idx := map[string]int{}
	for i, l := range c.Layers {
		idx[l.ID] = i
	}
	var fill func(i int)
	fill = func(i int) {
		l := &c.Layers[i]
		if l.Chain != "" {
			return
		}
		var experts []string
		for _, e := range l.Experts {
			fill(idx[e])
			experts = append(experts, c.Layers[idx[e]].Digest)
		}
		l.Digest = LayerDigest(*l, blobs[l.ID], experts)
		parentChain := ""
		if l.Parent != "" {
			fill(idx[l.Parent])
			parentChain = c.Layers[idx[l.Parent]].Chain
		}
		l.Chain = ChainID(parentChain, l.Digest)
	}
	for i := range c.Layers {
		fill(i)
	}
}

func TestCheckImageAcceptsBuiltImage(t *testing.T) {
	m, c := builtImage()
	if err := CheckImage(m, c); err != nil {
		t.Fatal(err)
	}
	if c.Layers[0].Chain != c.Layers[0].Digest {
		t.Error("base chain ID must equal its digest")
	}
}

// A trained mix holds its router, sealed into its digest; an Arrow mix
// holds nothing.
func TestCheckImageRouter(t *testing.T) {
	trained := func(router string, blobs int) (*ocispec.Manifest, *Config) {
		m, c := builtImage()
		for i := range c.Layers {
			c.Layers[i].Digest, c.Layers[i].Chain = "", ""
		}
		c.Layers[2].Router = router
		for i := 0; i < blobs; i++ {
			d := ocispec.Descriptor{MediaType: MediaTypeRouter, Digest: dg(string(rune('a' + i))), Size: 1,
				Annotations: map[string]string{AnnotationLayer: "domain", AnnotationTitle: RouterFile}}
			m.Layers = append(m.Layers, d)
		}
		seal(m, c)
		return m, c
	}
	if err := CheckImage(trained(RouterTrained, 1)); err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]struct {
		router string
		blobs  int
	}{
		"trained without its router": {RouterTrained, 0},
		"two routers":                {RouterTrained, 2},
		"arrow with a router":        {RouterArrow, 1},
	} {
		if err := CheckImage(trained(c.router, c.blobs)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	// The router is part of the mix: another router is another mix.
	m, c := trained(RouterTrained, 1)
	m.Layers[len(m.Layers)-1].Digest = dg("f")
	if err := CheckImage(m, c); err == nil {
		t.Error("swapped router: no error")
	}
}

func TestCheckImageRejectsTampering(t *testing.T) {
	cases := map[string]func(m *ocispec.Manifest, c *Config){
		"artifact type":     func(m *ocispec.Manifest, _ *Config) { m.ArtifactType = "application/x" },
		"config media type": func(m *ocispec.Manifest, _ *Config) { m.Config.MediaType = "application/x" },
		"swapped blob":      func(m *ocispec.Manifest, _ *Config) { m.Layers[3].Digest = dg("9") },
		"dropped blob":      func(m *ocispec.Manifest, _ *Config) { m.Layers = m.Layers[1:] },
		"changed alpha":     func(_ *ocispec.Manifest, c *Config) { c.Layers[1].Alpha = 32 },
		"changed top_k":     func(_ *ocispec.Manifest, c *Config) { c.Layers[2].TopK = 2 },
		"unknown layer":     func(m *ocispec.Manifest, _ *Config) { m.Layers[3].Annotations[AnnotationLayer] = "nope" },
		"blob in mix":       func(m *ocispec.Manifest, _ *Config) { m.Layers[3].Annotations[AnnotationLayer] = "domain" },
		"unannotated":       func(m *ocispec.Manifest, _ *Config) { delete(m.Layers[0].Annotations, AnnotationLayer) },
		"missing chain":     func(_ *ocispec.Manifest, c *Config) { c.Layers[1].Chain = "" },
		"chain off parent":  func(_ *ocispec.Manifest, c *Config) { c.Layers[1].Chain = ChainID("sha256:other", c.Layers[1].Digest) },
		"expert swapped": func(m *ocispec.Manifest, c *Config) {
			// legal's blob changes and its own digest and chain are recomputed,
			// but the mix still pins the old expert digest.
			m.Layers[4].Digest = dg("8")
			c.Layers[3].Digest = LayerDigest(c.Layers[3], []BlobRef{BlobRefFromDescriptor(m.Layers[4])}, nil)
			c.Layers[3].Chain = ChainID(c.Layers[1].Chain, c.Layers[3].Digest)
		},
	}
	for name, tamper := range cases {
		m, c := builtImage()
		tamper(m, c)
		if err := CheckImage(m, c); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestLayerDigest(t *testing.T) {
	l := Layer{ID: "a", Kind: KindLoRA, Rank: 8, Alpha: 16}
	b1 := BlobRef{Tensor: "x", MediaType: MediaTypeLoRA, Digest: dg("1").String()}
	b2 := BlobRef{Tensor: "y", MediaType: MediaTypeLoRA, Digest: dg("2").String()}
	d := LayerDigest(l, []BlobRef{b1, b2}, nil)
	if LayerDigest(l, []BlobRef{b2, b1}, nil) != d {
		t.Error("digest depends on blob order")
	}
	renamed := l
	renamed.ID, renamed.Parent = "b", "elsewhere"
	if LayerDigest(renamed, []BlobRef{b1, b2}, nil) != d {
		t.Error("digest depends on the layer id or parent; the chain ID covers the parent")
	}
	scaled := l
	scaled.Alpha = 32
	if LayerDigest(scaled, []BlobRef{b1, b2}, nil) == d {
		t.Error("digest ignores alpha")
	}
	b1.MediaType = MediaTypeDeltaXOR
	if LayerDigest(l, []BlobRef{b1, b2}, nil) == d {
		t.Error("digest ignores media types")
	}
}
