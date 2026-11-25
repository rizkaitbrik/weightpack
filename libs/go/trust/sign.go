package trust

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/opencontainers/go-digest"
	"github.com/sigstore/sigstore-go/pkg/root"
	"github.com/sigstore/sigstore-go/pkg/sign"
	"github.com/sigstore/sigstore/pkg/oauthflow"
	"google.golang.org/protobuf/encoding/protojson"
)

// Signer makes Sigstore bundles that sign image manifests.
type Signer struct {
	keypair sign.Keypair
	opts    sign.BundleOptions
	// Who describes the signer: a key fingerprint, or the identity in the
	// OIDC token.
	Who string
}

// NewKeySigner signs with a long-lived key (see LoadKey). Its bundles hold
// a hint of the key, not a certificate, and are not logged in Rekor, so
// signing needs no network and an image's name never leaves the registry.
func NewKeySigner(privatePEM, password []byte) (*Signer, error) {
	kp, err := LoadKey(privatePEM, password)
	if err != nil {
		return nil, err
	}
	return &Signer{keypair: kp, Who: "key " + Fingerprint(kp.GetPublicKey())}, nil
}

// NewKeylessSigner signs with a fresh key that Fulcio certifies for the
// identity in idToken, an OIDC ID token, and records the signature in the
// instance's Rekor log (and with its timestamp authorities). Verifiers then
// trust the identity, not a key. The log is public: it shows the identity
// and the manifest digest.
func NewKeylessSigner(sg *Sigstore, idToken string) (*Signer, error) {
	sc, err := sg.SigningConfig()
	if err != nil {
		return nil, err
	}
	now := time.Now()
	fulcio, err := root.SelectService(sc.FulcioCertificateAuthorityURLs(), sign.FulcioAPIVersions, now)
	if err != nil {
		return nil, fmt.Errorf("pick a Fulcio instance: %w", err)
	}
	opts := sign.BundleOptions{
		CertificateProvider:        sign.NewFulcio(&sign.FulcioOptions{BaseURL: fulcio.URL, Timeout: 30 * time.Second, Retries: 1}),
		CertificateProviderOptions: &sign.CertificateProviderOptions{IDToken: idToken},
	}
	rekors, err := root.SelectServices(sc.RekorLogURLs(), sc.RekorLogURLsConfig(), sign.RekorAPIVersions, now)
	if err != nil {
		return nil, fmt.Errorf("pick a Rekor log: %w", err)
	}
	for _, r := range rekors {
		opts.TransparencyLogs = append(opts.TransparencyLogs, sign.NewRekor(&sign.RekorOptions{
			BaseURL: r.URL, Version: r.MajorAPIVersion, Timeout: 90 * time.Second, Retries: 1,
		}))
	}
	if len(sc.TimestampAuthorityURLs()) > 0 {
		tsas, err := root.SelectServices(sc.TimestampAuthorityURLs(), sc.TimestampAuthorityURLsConfig(), sign.TimestampAuthorityAPIVersions, now)
		if err != nil {
			return nil, fmt.Errorf("pick a timestamp authority: %w", err)
		}
		for _, t := range tsas {
			opts.TimestampAuthorities = append(opts.TimestampAuthorities, sign.NewTimestampAuthority(&sign.TimestampAuthorityOptions{
				URL: t.URL, Timeout: 30 * time.Second, Retries: 1,
			}))
		}
	}
	// sigstore-go checks the finished bundle against the trusted root.
	tm, err := sg.TrustedMaterial()
	if err != nil {
		return nil, err
	}
	opts.TrustedRoot = tm
	kp, err := sign.NewEphemeralKeypair(nil)
	if err != nil {
		return nil, err
	}
	who := "keyless"
	if subject, err := oauthflow.SubjectFromUnverifiedToken([]byte(idToken)); err == nil && subject != "" {
		who = subject
	}
	return &Signer{keypair: kp, opts: opts, Who: who}, nil
}

// statement is the in-toto statement cosign signs for an image: the
// manifest digest as its only subject, cosign's signing predicate type and
// an empty predicate.
type statement struct {
	Type          string           `json:"_type"`
	Subject       []statementEntry `json:"subject"`
	PredicateType string           `json:"predicateType"`
	Predicate     struct{}         `json:"predicate"`
}

type statementEntry struct {
	Digest map[string]string `json:"digest"`
}

// Sign returns a Sigstore bundle (JSON, media type BundleMediaType) whose
// DSSE envelope signs manifest the way cosign sign does.
func (s *Signer) Sign(ctx context.Context, manifest digest.Digest) ([]byte, error) {
	if err := manifest.Validate(); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(statement{
		Type:          "https://in-toto.io/Statement/v1",
		Subject:       []statementEntry{{Digest: map[string]string{manifest.Algorithm().String(): manifest.Encoded()}}},
		PredicateType: PredicateType,
	})
	if err != nil {
		return nil, err
	}
	opts := s.opts
	opts.Context = ctx
	b, err := sign.Bundle(&sign.DSSEData{Data: payload, PayloadType: inTotoPayloadType}, s.keypair, opts)
	if err != nil {
		return nil, fmt.Errorf("sign %s: %w", manifest, err)
	}
	return protojson.Marshal(b)
}
