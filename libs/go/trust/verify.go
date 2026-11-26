package trust

import (
	"context"
	"crypto"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sigstore/sigstore-go/pkg/root"
	"github.com/sigstore/sigstore-go/pkg/verify"
	"github.com/sigstore/sigstore/pkg/signature"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
)

// Requirement is whose signature an image needs: any one of Keys, or any
// one of Identities.
type Requirement struct {
	Keys       []crypto.PublicKey
	Identities []api.Identity
}

// NewRequirement parses PEM public keys and checks identities.
func NewRequirement(keys []string, identities []api.Identity) (Requirement, error) {
	var req Requirement
	for _, k := range keys {
		pub, err := ParsePublicKey([]byte(k))
		if err != nil {
			return Requirement{}, err
		}
		req.Keys = append(req.Keys, pub)
	}
	for _, id := range identities {
		if _, err := certificateIdentity(id); err != nil {
			return Requirement{}, err
		}
	}
	req.Identities = identities
	if len(req.Keys) == 0 && len(req.Identities) == 0 {
		return Requirement{}, errors.New("name at least one trusted key or keyless identity")
	}
	return req, nil
}

// Verifier checks image signatures.
type Verifier struct {
	// Sigstore provides the trusted root keyless signatures chain to. Key
	// signatures do not need it.
	Sigstore *Sigstore

	// skipSCT drops the check that Fulcio logged the certificate in a
	// certificate transparency log, for test CAs that do not.
	skipSCT bool
}

// Result says which signature satisfied a requirement.
type Result struct {
	// Signer is "key SHA256:…" or the certificate's subject and issuer.
	Signer string
	// Signed is when a log or timestamp authority saw the signature, for
	// keyless signatures.
	Signed time.Time
	// Signatures counts the bundles attached to the image.
	Signatures int
}

// ErrUnsigned means the image carries no signature at all.
var ErrUnsigned = errNoSignatures

// Verify checks that one of the Sigstore bundles attached to subject in
// repo signs it and was made by a signer req trusts.
func (v *Verifier) Verify(ctx context.Context, repo Repository, subject ocispec.Descriptor, req Requirement) (*Result, error) {
	bundles, err := Bundles(ctx, repo, subject)
	if err != nil {
		return nil, err
	}
	if len(bundles) == 0 {
		return nil, fmt.Errorf("%w to %s", ErrUnsigned, subject.Digest)
	}
	var errs []string
	for _, b := range bundles {
		res, err := v.Check(b, subject.Digest, req)
		if err == nil {
			res.Signatures = len(bundles)
			return res, nil
		}
		errs = append(errs, err.Error())
	}
	count := "1 signature"
	if len(bundles) > 1 {
		count = fmt.Sprintf("%d signatures", len(bundles))
	}
	return nil, fmt.Errorf("%s has %s, none from a trusted signer: %s",
		subject.Digest, count, strings.Join(dedupe(errs), "; "))
}

// Check verifies one signed entity (a parsed bundle) against req.
func (v *Verifier) Check(entity verify.SignedEntity, manifest digest.Digest, req Requirement) (*Result, error) {
	raw, err := hex.DecodeString(manifest.Encoded())
	if err != nil {
		return nil, err
	}
	artifact := verify.WithArtifactDigest(manifest.Algorithm().String(), raw)
	content, err := entity.VerificationContent()
	if err != nil {
		return nil, err
	}

	if content.Certificate() == nil {
		if len(req.Keys) == 0 {
			return nil, errors.New("signed with a key, and only keyless identities are trusted")
		}
		for _, k := range req.Keys {
			res, err := checkKey(entity, artifact, k)
			if err == nil {
				return &Result{Signer: "key " + Fingerprint(k), Signed: res}, nil
			}
		}
		return nil, errors.New("signed with a key that is not one of the trusted keys")
	}

	if len(req.Identities) == 0 {
		return nil, fmt.Errorf("signed keyless as %s, and only keys are trusted", certSubject(content))
	}
	if v.Sigstore == nil {
		return nil, errors.New("keyless signatures need a Sigstore trusted root, and none is configured")
	}
	tm, err := v.Sigstore.TrustedMaterial()
	if err != nil {
		return nil, err
	}
	opts := []verify.VerifierOption{verify.WithTransparencyLog(1), verify.WithObserverTimestamps(1)}
	if !v.skipSCT {
		opts = append(opts, verify.WithSignedCertificateTimestamps(1))
	}
	sev, err := verify.NewVerifier(tm, opts...)
	if err != nil {
		return nil, err
	}
	policy := []verify.PolicyOption{}
	for _, id := range req.Identities {
		ci, err := certificateIdentity(id)
		if err != nil {
			return nil, err
		}
		policy = append(policy, verify.WithCertificateIdentity(ci))
	}
	res, err := sev.Verify(entity, verify.NewPolicy(artifact, policy...))
	if err != nil {
		return nil, fmt.Errorf("signed keyless as %s: %w", certSubject(content), err)
	}
	if err := checkPredicate(res); err != nil {
		return nil, err
	}
	out := &Result{Signer: fmt.Sprintf("%s (issuer %s)", res.Signature.Certificate.SubjectAlternativeName, res.Signature.Certificate.Issuer)}
	for _, ts := range res.VerifiedTimestamps {
		if out.Signed.IsZero() || ts.Timestamp.Before(out.Signed) {
			out.Signed = ts.Timestamp
		}
	}
	return out, nil
}

// checkKey verifies a key-signed entity with one public key.
func checkKey(entity verify.SignedEntity, artifact verify.ArtifactPolicyOption, pub crypto.PublicKey) (time.Time, error) {
	sv, err := signature.LoadDefaultVerifier(pub)
	if err != nil {
		return time.Time{}, err
	}
	key := root.NewExpiringKey(sv, time.Time{}, time.Time{})
	tm := root.NewTrustedPublicKeyMaterial(func(string) (root.TimeConstrainedVerifier, error) { return key, nil })
	sev, err := verify.NewVerifier(tm, verify.WithNoObserverTimestamps())
	if err != nil {
		return time.Time{}, err
	}
	res, err := sev.Verify(entity, verify.NewPolicy(artifact, verify.WithKey()))
	if err != nil {
		return time.Time{}, err
	}
	return time.Time{}, checkPredicate(res)
}

// checkPredicate refuses attestations: an image signature is an in-toto
// statement with cosign's signing predicate type.
func checkPredicate(res *verify.VerificationResult) error {
	if res.Statement == nil {
		return errors.New("the bundle signs a bare message, not an image")
	}
	if pt := res.Statement.GetPredicateType(); pt != PredicateType {
		return fmt.Errorf("a %q attestation, not an image signature", pt)
	}
	return nil
}

// certificateIdentity builds sigstore-go's matcher for an identity. Regular
// expressions are anchored, so they must match the whole value: an
// unanchored "https://github.com/acme/" would also match a URI that merely
// contains it.
func certificateIdentity(id api.Identity) (verify.CertificateIdentity, error) {
	if id.Subject == "" && id.SubjectRegexp == "" {
		return verify.CertificateIdentity{}, errors.New("a keyless identity needs a subject or a subject regexp")
	}
	if id.Issuer == "" && id.IssuerRegexp == "" {
		return verify.CertificateIdentity{}, errors.New("a keyless identity needs an issuer or an issuer regexp")
	}
	ci, err := verify.NewShortCertificateIdentity(id.Issuer, anchor(id.IssuerRegexp), id.Subject, anchor(id.SubjectRegexp))
	if err != nil {
		return verify.CertificateIdentity{}, fmt.Errorf("identity %s: %w", id, err)
	}
	return ci, nil
}

func anchor(re string) string {
	if re == "" {
		return ""
	}
	return "^(?:" + re + ")$"
}

// certSubject reads who a certificate names, before any verification, for
// error messages.
func certSubject(content verify.VerificationContent) string {
	cert := content.Certificate()
	if cert == nil {
		return "a key"
	}
	for _, e := range cert.EmailAddresses {
		return e
	}
	for _, u := range cert.URIs {
		return u.String()
	}
	return cert.Subject.String()
}

func dedupe(s []string) []string {
	seen := map[string]bool{}
	out := s[:0]
	for _, x := range s {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}
