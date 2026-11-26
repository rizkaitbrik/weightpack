package trust

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	protobundle "github.com/sigstore/protobuf-specs/gen/pb-go/bundle/v1"
	"github.com/sigstore/sigstore-go/pkg/bundle"
	"google.golang.org/protobuf/encoding/protojson"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/registry"
)

// Repository is what signatures need from a registry repository; oras's
// remote.Repository is one.
type Repository interface {
	oras.Target
	registry.ReferrerLister
}

const (
	// maxBundleSize bounds a bundle wpk reads. Bundles hold a signature, a
	// certificate chain and log proofs: a few kilobytes.
	maxBundleSize = 1 << 20
	// maxManifestSize bounds a referrer manifest.
	maxManifestSize = 64 << 10
	// maxSignatures bounds how many signatures of one image are checked.
	maxSignatures = 32
)

// Attach stores bundle, which must sign subject (see Signer.Sign), in repo
// as cosign v3 does: an OCI 1.1 manifest of artifact type BundleMediaType
// with the bundle as its one layer and subject as its subject. Registries
// without the referrers API get the fallback tag oras maintains. It
// returns the descriptor of that manifest.
func Attach(ctx context.Context, repo Repository, subject ocispec.Descriptor, bundleJSON []byte) (ocispec.Descriptor, error) {
	b, err := parseBundle(bundleJSON)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if err := signsManifest(b, subject.Digest); err != nil {
		return ocispec.Descriptor{}, err
	}
	layer, err := oras.PushBytes(ctx, repo, BundleMediaType, bundleJSON)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("upload the signature: %w", err)
	}
	layer.Annotations = map[string]string{
		ocispec.AnnotationTitle: fmt.Sprintf("%s-%s.sigstore.json", layer.Digest.Algorithm(), layer.Digest.Encoded()),
	}
	desc, err := oras.PackManifest(ctx, repo, oras.PackManifestVersion1_1, BundleMediaType, oras.PackManifestOptions{
		Subject: &ocispec.Descriptor{MediaType: subject.MediaType, Digest: subject.Digest, Size: subject.Size},
		Layers:  []ocispec.Descriptor{layer},
		ManifestAnnotations: map[string]string{
			"dev.sigstore.bundle.content":       "dsse-envelope",
			"dev.sigstore.bundle.predicateType": PredicateType,
		},
	})
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("store the signature: %w", err)
	}
	return desc, nil
}

// Bundles returns the Sigstore bundles attached to subject in repo, at most
// maxSignatures of them, skipping referrers that hold something else.
func Bundles(ctx context.Context, repo Repository, subject ocispec.Descriptor) ([]*bundle.Bundle, error) {
	var refs []ocispec.Descriptor
	errEnough := errors.New("enough")
	err := repo.Referrers(ctx, subject, BundleMediaType, func(page []ocispec.Descriptor) error {
		refs = append(refs, page...)
		if len(refs) >= maxSignatures {
			refs = refs[:maxSignatures]
			return errEnough
		}
		return nil
	})
	if err != nil && !errors.Is(err, errEnough) {
		return nil, fmt.Errorf("list the signatures of %s: %w", subject.Digest, err)
	}
	var out []*bundle.Bundle
	for _, r := range refs {
		b, err := fetchBundle(ctx, repo, r)
		if err != nil {
			continue // not a bundle wpk can read; another tool's artifact
		}
		out = append(out, b)
	}
	return out, nil
}

func fetchBundle(ctx context.Context, repo Repository, ref ocispec.Descriptor) (*bundle.Bundle, error) {
	if ref.MediaType != ocispec.MediaTypeImageManifest || ref.Size > maxManifestSize {
		return nil, errors.New("not a bundle manifest")
	}
	raw, err := content.FetchAll(ctx, repo, ref)
	if err != nil {
		return nil, err
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	if len(m.Layers) != 1 || !strings.HasPrefix(m.Layers[0].MediaType, "application/vnd.dev.sigstore.bundle") || m.Layers[0].Size > maxBundleSize {
		return nil, errors.New("not a bundle manifest")
	}
	data, err := content.FetchAll(ctx, repo, m.Layers[0])
	if err != nil {
		return nil, err
	}
	return parseBundle(data)
}

func parseBundle(data []byte) (*bundle.Bundle, error) {
	if len(data) > maxBundleSize {
		return nil, fmt.Errorf("signature bundle of %d bytes is too large", len(data))
	}
	var pb protobundle.Bundle
	if err := protojson.Unmarshal(data, &pb); err != nil {
		return nil, fmt.Errorf("not a Sigstore bundle: %w", err)
	}
	b, err := bundle.NewBundle(&pb)
	if err != nil {
		return nil, fmt.Errorf("not a valid Sigstore bundle: %w", err)
	}
	return b, nil
}

// signsManifest checks, without verifying the signature, that a bundle
// is an image signature for d: a DSSE envelope whose in-toto statement has
// cosign's signing predicate type and d among its subjects.
func signsManifest(b *bundle.Bundle, d digest.Digest) error {
	env, err := b.Envelope()
	if err != nil {
		return errors.New("the bundle holds no DSSE envelope; image signatures are in-toto statements")
	}
	st, err := env.Statement()
	if err != nil {
		return fmt.Errorf("the bundle's envelope holds no in-toto statement: %w", err)
	}
	if st.GetPredicateType() != PredicateType {
		return fmt.Errorf("the bundle is a %q attestation, not an image signature", st.GetPredicateType())
	}
	for _, s := range st.GetSubject() {
		if s.GetDigest()[d.Algorithm().String()] == d.Encoded() {
			return nil
		}
	}
	return fmt.Errorf("the bundle does not sign %s", d)
}
