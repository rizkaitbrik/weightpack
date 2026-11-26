// Package trust signs Weightpack images and checks their signatures with
// Sigstore, in the format cosign v3 uses for container images, so cosign
// verifies what wpk signs and wpk verifies what cosign signs.
//
// A signature is a Sigstore bundle (BundleMediaType) holding a DSSE
// envelope over an in-toto statement: the image manifest's digest as the
// subject, PredicateType as the predicate type, an empty predicate. The
// registry stores the bundle as an OCI referrer of the manifest. Signing the
// manifest signs every layer, since the manifest names each blob by digest.
//
// A bundle is signed either with a long-lived key (cosign's key format;
// GenerateKeyPair writes one) or keyless: a short-lived Fulcio certificate
// for an OIDC identity, logged in Rekor. Verification takes a Requirement,
// trusted keys or trusted identities, and a policy (api.TrustPolicy) says
// which requirement applies to which registry or repository.
package trust

const (
	// BundleMediaType is a Sigstore bundle, v0.3: the referrer's artifact
	// type and its one layer's media type.
	BundleMediaType = "application/vnd.dev.sigstore.bundle.v0.3+json"
	// PredicateType marks an in-toto statement as an image signature, as
	// cosign sign writes it.
	PredicateType = "https://sigstore.dev/cosign/sign/v1"

	inTotoPayloadType = "application/vnd.in-toto+json"
)
