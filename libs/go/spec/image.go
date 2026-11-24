package spec

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// AnnotationTitle names a meta blob's file, such as "tokenizer.json". It is
// the standard OCI title annotation.
const AnnotationTitle = ocispec.AnnotationTitle

// BlobRef is one blob of a layer, as its layer digest sees it.
type BlobRef struct {
	// Tensor is the tensor a single-tensor blob holds, or for a lora.v1 blob
	// the base tensor it modifies. It is empty for a packed blob, whose
	// tensor names are in its own safetensors header.
	Tensor string `json:"tensor,omitempty"`
	// Title is the file name of a meta blob.
	Title     string `json:"title,omitempty"`
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
}

// BlobRefFromDescriptor reads a BlobRef from a manifest layer descriptor.
func BlobRefFromDescriptor(d ocispec.Descriptor) BlobRef {
	return BlobRef{
		Tensor:    d.Annotations[AnnotationTensor],
		Title:     d.Annotations[AnnotationTitle],
		MediaType: d.MediaType,
		Digest:    d.Digest.String(),
	}
}

// layerContent is what a layer digest is computed over: the layer's kind,
// the parameters that change how its tensors apply, and its blobs. The layer
// id is left out, so the same content under another name is the same layer,
// as in Docker. Fidelity records are left out too: they describe how a layer
// was made, not what it is.
type layerContent struct {
	Kind    LayerKind `json:"kind"`
	Rank    int       `json:"rank,omitempty"`
	Alpha   float64   `json:"alpha,omitempty"`
	RSLoRA  bool      `json:"rslora,omitempty"`
	Targets []string  `json:"targets,omitempty"`
	Blocks  string    `json:"blocks,omitempty"`
	TopK    int       `json:"top_k,omitempty"`
	Router  string    `json:"router,omitempty"`
	Experts []string  `json:"experts,omitempty"`
	Blobs   []BlobRef `json:"blobs,omitempty"`
}

// LayerDigest returns the digest that identifies a layer: sha256 over the
// compact JSON of its kind, rank, alpha, rslora, targets, blocks, top_k,
// router, expert layer digests (for a mix, in mix order) and blobs (sorted by
// tensor, title, then digest).
func LayerDigest(l Layer, blobs []BlobRef, expertDigests []string) string {
	sorted := make([]BlobRef, len(blobs))
	copy(sorted, blobs)
	sort.Slice(sorted, func(i, j int) bool {
		a, b := sorted[i], sorted[j]
		if a.Tensor != b.Tensor {
			return a.Tensor < b.Tensor
		}
		if a.Title != b.Title {
			return a.Title < b.Title
		}
		return a.Digest < b.Digest
	})
	c := layerContent{
		Kind: l.Kind, Rank: l.Rank, Alpha: l.Alpha, RSLoRA: l.RSLoRA,
		Targets: l.Targets, Blocks: l.Blocks, TopK: l.TopK, Router: l.Router,
		Experts: expertDigests, Blobs: sorted,
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(c); err != nil {
		// Every field is a plain string, number or slice of them.
		panic(err)
	}
	sum := sha256.Sum256(bytes.TrimRight(buf.Bytes(), "\n"))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// isMeta reports whether a blob may appear without a layer annotation:
// image-level files such as a chat template or license set by the Weightfile.
func isMeta(mediaType string) bool { return BaseMediaType(mediaType) == MediaTypeMeta }

// CheckImage verifies that a manifest and its config agree, which is what
// makes an image safe to compose and serve:
//
//   - the manifest has the Weightpack artifact and config media types;
//   - the config passes Validate;
//   - every layer blob names a layer of the config (meta blobs may name
//     none, which makes them image-level);
//   - each layer's digest matches its kind, parameters, blobs and experts;
//   - each chain ID matches its parent's chain ID and its own digest.
func CheckImage(m *ocispec.Manifest, c *Config) error {
	if m.ArtifactType != ArtifactType {
		return fmt.Errorf("manifest artifactType %q, want %q", m.ArtifactType, ArtifactType)
	}
	if m.Config.MediaType != MediaTypeConfig {
		return fmt.Errorf("config media type %q, want %q", m.Config.MediaType, MediaTypeConfig)
	}
	if err := c.Validate(); err != nil {
		return err
	}
	byID := make(map[string]Layer, len(c.Layers))
	for _, l := range c.Layers {
		byID[l.ID] = l
	}
	blobs := map[string][]BlobRef{}
	for _, d := range m.Layers {
		id, ok := d.Annotations[AnnotationLayer]
		if !ok {
			if !isMeta(d.MediaType) {
				return fmt.Errorf("blob %s (%s) has no %s annotation", d.Digest, d.MediaType, AnnotationLayer)
			}
			continue
		}
		l, known := byID[id]
		if !known {
			return fmt.Errorf("blob %s names unknown layer %q", d.Digest, id)
		}
		if l.Kind == KindMix {
			return fmt.Errorf("blob %s belongs to mix %q, which holds no blobs", d.Digest, id)
		}
		blobs[id] = append(blobs[id], BlobRefFromDescriptor(d))
	}
	for _, l := range c.Layers {
		if l.Digest == "" || l.Chain == "" {
			return fmt.Errorf("layer %q has no digest or chain ID", l.ID)
		}
		var experts []string
		for _, e := range l.Experts {
			experts = append(experts, byID[e].Digest)
		}
		if got := LayerDigest(l, blobs[l.ID], experts); got != l.Digest {
			return fmt.Errorf("layer %q: digest %s does not match its content (%s)", l.ID, l.Digest, got)
		}
		want := ChainID(byID[l.Parent].Chain, l.Digest)
		if l.Kind == KindBase {
			want = ChainID("", l.Digest)
		}
		if l.Chain != want {
			return fmt.Errorf("layer %q: chain ID %s, want %s", l.ID, l.Chain, want)
		}
	}
	return nil
}
