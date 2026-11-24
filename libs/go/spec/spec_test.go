package spec

import (
	"strings"
	"testing"
)

func TestBaseMediaTypeAndLossless(t *testing.T) {
	cases := map[string]bool{
		MediaTypeLoRA + "+zipnn":     true,
		MediaTypeDeltaXOR + "+zstd":  true,
		MediaTypeDeltaBit1:           false,
		"application/vnd.other.v1":   false,
		MediaTypeTensor:              true,
		MediaTypeDeltaSparse + "+zs": false,
	}
	for mt, want := range cases {
		if got := IsLossless(mt); got != want {
			t.Errorf("IsLossless(%q) = %v, want %v", mt, got, want)
		}
	}
	if got := BaseMediaType(MediaTypeLoRA + "+zipnn"); got != MediaTypeLoRA {
		t.Errorf("BaseMediaType = %q", got)
	}
}

func TestChainID(t *testing.T) {
	base := "sha256:aaaa"
	if got := ChainID("", base); got != base {
		t.Fatalf("base chain id = %q, want %q", got, base)
	}
	c1 := ChainID(base, "sha256:bbbb")
	c2 := ChainID(base, "sha256:bbbb")
	if c1 != c2 || !strings.HasPrefix(c1, "sha256:") {
		t.Fatalf("chain id not deterministic: %q %q", c1, c2)
	}
	if ChainID("sha256:other", "sha256:bbbb") == c1 {
		t.Fatal("chain id must depend on the parent")
	}
}

func TestArchFingerprintIsOrderIndependent(t *testing.T) {
	a := []TensorInfo{
		{Name: "model.embed_tokens.weight", DType: "BF16", Shape: []int64{128256, 4096}},
		{Name: "model.norm.weight", DType: "BF16", Shape: []int64{4096}},
	}
	b := []TensorInfo{a[1], a[0]}
	if ArchFingerprint(a, "sha256:tok") != ArchFingerprint(b, "sha256:tok") {
		t.Fatal("fingerprint depends on tensor order")
	}
	if ArchFingerprint(a, "sha256:tok") == ArchFingerprint(a, "sha256:other") {
		t.Fatal("fingerprint ignores the tokenizer")
	}
}

func validConfig() Config {
	return Config{
		Schema: SchemaV1,
		Arch:   Arch{Family: "llama", Fingerprint: "sha256:9f1c"},
		Layers: []Layer{
			{ID: "base", Kind: KindBase},
			{ID: "french", Kind: KindLoRA, Parent: "base", Rank: 32},
			{ID: "domain", Kind: KindMix, Parent: "french", TopK: 2, Router: "arrow", Experts: []string{"legal", "medical"}},
			{ID: "legal", Kind: KindLoRA, Parent: "french", Rank: 16},
			{ID: "medical", Kind: KindLoRA, Parent: "french", Rank: 16},
		},
	}
}

func TestValidate(t *testing.T) {
	c := validConfig()
	if err := c.Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}

	bad := validConfig()
	bad.Layers[4].Parent = "base"
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "expert") {
		t.Fatalf("expert with wrong parent accepted: %v", err)
	}

	bad = validConfig()
	bad.Layers[2].TopK = 3
	if err := bad.Validate(); err == nil {
		t.Fatal("top_k larger than expert count accepted")
	}

	bad = validConfig()
	bad.Layers = append(bad.Layers, Layer{ID: "base2", Kind: KindBase})
	if err := bad.Validate(); err == nil {
		t.Fatal("two base layers accepted")
	}
}

func TestValidateRejectsBadLayerIDs(t *testing.T) {
	for _, id := range []string{"../escaped", "a/b", "Upper", ".hidden", "-x", strings.Repeat("a", 65)} {
		c := validConfig()
		c.Layers[1].ID = id
		for i := range c.Layers {
			if c.Layers[i].Parent == "french" {
				c.Layers[i].Parent = id
			}
		}
		if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "bad layer id") {
			t.Errorf("id %q: err = %v", id, err)
		}
	}
	if !ValidLayerID("legal-v2.1_x") {
		t.Error("a well-formed id was rejected")
	}
}
