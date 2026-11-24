// Package spec defines the Weightpack image format: OCI media types, the
// image config schema, chain IDs and the architecture fingerprint.
package spec

// ArtifactType is the OCI artifactType of a Weightpack image manifest.
const ArtifactType = "application/vnd.weightpack.image.v1+json"

// MediaTypeConfig is the media type of the image config blob.
const MediaTypeConfig = "application/vnd.weightpack.config.v1+json"

// Layer blob media types. An entropy coder is appended as a suffix,
// for example MediaTypeLoRA + "+zipnn".
const (
	MediaTypeTensor      = "application/vnd.weightpack.tensor.v1"
	MediaTypeLoRA        = "application/vnd.weightpack.lora.v1"
	MediaTypeDeltaXOR    = "application/vnd.weightpack.delta.xor.v1"
	MediaTypeDeltaBit1   = "application/vnd.weightpack.delta.bit1.v1"
	MediaTypeDeltaSparse = "application/vnd.weightpack.delta.sparse.v1"
	MediaTypeRouter      = "application/vnd.weightpack.router.v1"
	MediaTypeMeta        = "application/vnd.weightpack.meta.v1"
)

// Manifest annotation keys set on every layer descriptor.
const (
	AnnotationLayer  = "wpk.layer"
	AnnotationTensor = "wpk.tensor"
)

// lossless reports which base media types restore the source bit for bit.
var lossless = map[string]bool{
	MediaTypeTensor:      true,
	MediaTypeLoRA:        true,
	MediaTypeDeltaXOR:    true,
	MediaTypeDeltaBit1:   false,
	MediaTypeDeltaSparse: false,
	MediaTypeRouter:      true,
	MediaTypeMeta:        true,
}

// IsLossless reports whether a blob of the given media type (with or without
// an entropy-coder suffix) can be decoded back to its source exactly.
// Unknown media types are treated as lossy.
func IsLossless(mediaType string) bool {
	return lossless[BaseMediaType(mediaType)]
}

// BaseMediaType strips an entropy-coder suffix such as "+zipnn" or "+zstd".
func BaseMediaType(mediaType string) string {
	for i := 0; i < len(mediaType); i++ {
		if mediaType[i] == '+' {
			return mediaType[:i]
		}
	}
	return mediaType
}
