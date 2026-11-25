package trust

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http/httptest"
	"strings"
	"testing"

	ggcr "github.com/google/go-containerregistry/pkg/registry"
	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sigstore/sigstore-go/pkg/testing/ca"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/registry/remote"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
)

// newRepo starts an in-memory registry and returns a repository on it with
// one manifest pushed, the image to sign.
func newRepo(t *testing.T) (*remote.Repository, ocispec.Descriptor) {
	t.Helper()
	srv := httptest.NewServer(ggcr.New(ggcr.Logger(log.New(io.Discard, "", 0))))
	t.Cleanup(srv.Close)
	repo, err := remote.NewRepository(strings.TrimPrefix(srv.URL, "http://") + "/team/assistant")
	if err != nil {
		t.Fatal(err)
	}
	repo.PlainHTTP = true
	ctx := context.Background()
	layer, err := oras.PushBytes(ctx, repo, "application/octet-stream", []byte("weights"))
	if err != nil {
		t.Fatal(err)
	}
	image, err := oras.PackManifest(ctx, repo, oras.PackManifestVersion1_1, "application/vnd.example", oras.PackManifestOptions{Layers: []ocispec.Descriptor{layer}})
	if err != nil {
		t.Fatal(err)
	}
	return repo, image
}

func keyPair(t *testing.T, password string) (priv, pub []byte) {
	t.Helper()
	priv, pub, err := GenerateKeyPair([]byte(password))
	if err != nil {
		t.Fatal(err)
	}
	return priv, pub
}

func requirement(t *testing.T, keys [][]byte, ids ...api.Identity) Requirement {
	t.Helper()
	var pems []string
	for _, k := range keys {
		pems = append(pems, string(k))
	}
	req, err := NewRequirement(pems, ids)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func TestKeySignAttachVerify(t *testing.T) {
	ctx := context.Background()
	repo, image := newRepo(t)
	priv, pub := keyPair(t, "hunter2")
	_, otherPub := keyPair(t, "")

	if !KeyNeedsPassword(priv) {
		t.Error("a generated key is not encrypted")
	}
	if _, err := NewKeySigner(priv, []byte("wrong")); err == nil || !strings.Contains(err.Error(), "wrong password") {
		t.Errorf("wrong password: %v", err)
	}
	signer, err := NewKeySigner(priv, []byte("hunter2"))
	if err != nil {
		t.Fatal(err)
	}
	pk, _ := ParsePublicKey(pub)
	if signer.Who != "key "+Fingerprint(pk) {
		t.Errorf("signer = %q", signer.Who)
	}

	v := &Verifier{}
	if _, err := v.Verify(ctx, repo, image, requirement(t, [][]byte{pub})); !errors.Is(err, ErrUnsigned) {
		t.Fatalf("unsigned image: %v", err)
	}

	b, err := signer.Sign(ctx, image.Digest)
	if err != nil {
		t.Fatal(err)
	}
	// The bundle holds what cosign v3 writes for an image.
	var shape struct {
		MediaType    string `json:"mediaType"`
		DsseEnvelope struct {
			PayloadType string `json:"payloadType"`
			Payload     []byte `json:"payload"`
		} `json:"dsseEnvelope"`
	}
	if err := json.Unmarshal(b, &shape); err != nil {
		t.Fatal(err)
	}
	var st statement
	_ = json.Unmarshal(shape.DsseEnvelope.Payload, &st)
	if shape.MediaType != BundleMediaType || shape.DsseEnvelope.PayloadType != inTotoPayloadType ||
		st.PredicateType != PredicateType || st.Subject[0].Digest["sha256"] != image.Digest.Encoded() {
		t.Fatalf("bundle = %s", b)
	}

	other, _ := oras.PushBytes(ctx, repo, ocispec.MediaTypeImageManifest, []byte(`{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{"mediaType":"application/vnd.oci.empty.v1+json","digest":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","size":2},"layers":[]}`))
	if _, err := Attach(ctx, repo, other, b); err == nil || !strings.Contains(err.Error(), "does not sign") {
		t.Errorf("attach to another manifest: %v", err)
	}
	sig, err := Attach(ctx, repo, image, b)
	if err != nil {
		t.Fatal(err)
	}

	// The referrer is laid out as cosign reads it.
	var m ocispec.Manifest
	rc, err := repo.Fetch(ctx, sig)
	if err != nil {
		t.Fatal(err)
	}
	_ = json.NewDecoder(rc).Decode(&m)
	rc.Close()
	if m.ArtifactType != BundleMediaType || m.Subject == nil || m.Subject.Digest != image.Digest ||
		len(m.Layers) != 1 || m.Layers[0].MediaType != BundleMediaType || m.Config.MediaType != ocispec.MediaTypeEmptyJSON ||
		m.Annotations["dev.sigstore.bundle.predicateType"] != PredicateType {
		t.Errorf("referrer manifest = %+v", m)
	}

	res, err := v.Verify(ctx, repo, image, requirement(t, [][]byte{otherPub, pub}))
	if err != nil {
		t.Fatal(err)
	}
	if res.Signer != signer.Who || res.Signatures != 1 {
		t.Errorf("verified = %+v", res)
	}
	if _, err := v.Verify(ctx, repo, image, requirement(t, [][]byte{otherPub})); err == nil || !strings.Contains(err.Error(), "not one of the trusted keys") {
		t.Errorf("untrusted key: %v", err)
	}
	if _, err := v.Verify(ctx, repo, image, requirement(t, nil, api.Identity{Subject: "a@example.com", Issuer: "https://issuer"})); err == nil || !strings.Contains(err.Error(), "only keyless identities") {
		t.Errorf("key signature against identities: %v", err)
	}
}

func TestUnencryptedKeysAndCosignHeaders(t *testing.T) {
	// cosign before 2.0 wrote ENCRYPTED COSIGN PRIVATE KEY; the body is the same.
	priv, pub := keyPair(t, "pw")
	old := []byte(strings.ReplaceAll(string(priv), "ENCRYPTED SIGSTORE PRIVATE KEY", "ENCRYPTED COSIGN PRIVATE KEY"))
	s, err := NewKeySigner(old, []byte("pw"))
	if err != nil {
		t.Fatal(err)
	}
	pk, _ := ParsePublicKey(pub)
	if s.Who != "key "+Fingerprint(pk) {
		t.Errorf("signer = %q", s.Who)
	}
	if _, err := ParsePublicKey([]byte("not a key")); err == nil {
		t.Error("parsed garbage as a public key")
	}
}

func TestKeylessIdentities(t *testing.T) {
	vs, err := ca.NewVirtualSigstore()
	if err != nil {
		t.Fatal(err)
	}
	v := &Verifier{Sigstore: &Sigstore{Material: vs}, skipSCT: true}
	d := digest.FromString("manifest")
	const workflow = "https://github.com/acme/models/.github/workflows/release.yml@refs/heads/main"
	const actions = "https://token.actions.githubusercontent.com"
	payload := func(pt string, d digest.Digest) []byte {
		b, _ := json.Marshal(statement{Type: "https://in-toto.io/Statement/v1", PredicateType: pt,
			Subject: []statementEntry{{Digest: map[string]string{"sha256": d.Encoded()}}}})
		return b
	}
	entity, err := vs.Attest(workflow, actions, payload(PredicateType, d))
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name string
		id   api.Identity
		ok   bool
	}{
		{"exact", api.Identity{Subject: workflow, Issuer: actions}, true},
		{"regexp", api.Identity{SubjectRegexp: `https://github\.com/acme/.*`, Issuer: actions}, true},
		{"anchored", api.Identity{SubjectRegexp: `github\.com/acme/`, Issuer: actions}, false},
		{"other subject", api.Identity{Subject: "https://github.com/evil/x", Issuer: actions}, false},
		{"other issuer", api.Identity{Subject: workflow, Issuer: "https://accounts.google.com"}, false},
	}
	for _, c := range cases {
		res, err := v.Check(entity, d, requirement(t, nil, c.id))
		if (err == nil) != c.ok {
			t.Errorf("%s: %+v, %v", c.name, res, err)
			continue
		}
		if c.ok && (res.Signer != workflow+" (issuer "+actions+")" || res.Signed.IsZero()) {
			t.Errorf("%s: %+v", c.name, res)
		}
	}

	_, pub := keyPair(t, "")
	if _, err := v.Check(entity, d, requirement(t, [][]byte{pub})); err == nil || !strings.Contains(err.Error(), "only keys are trusted") {
		t.Errorf("keyless signature against keys: %v", err)
	}
	if _, err := v.Check(entity, digest.FromString("other"), requirement(t, nil, cases[0].id)); err == nil {
		t.Error("a signature of another manifest verified")
	}
	att, _ := vs.Attest(workflow, actions, payload("https://slsa.dev/provenance/v1", d))
	if _, err := v.Check(att, d, requirement(t, nil, cases[0].id)); err == nil || !strings.Contains(err.Error(), "attestation") {
		t.Errorf("provenance attestation as a signature: %v", err)
	}
	// Production verifiers insist on a certificate transparency proof.
	strict := &Verifier{Sigstore: &Sigstore{Material: vs}}
	if _, err := strict.Check(entity, d, requirement(t, nil, cases[0].id)); err == nil {
		t.Error("a certificate without an SCT verified")
	}
}

func TestPolicy(t *testing.T) {
	_, pub := keyPair(t, "")
	p := &api.TrustPolicy{Rules: []api.TrustRule{
		{Scope: "*", Action: api.TrustReject},
		{Scope: "GHCR.io/", Action: api.TrustAccept},
		{Scope: "ghcr.io/acme", Action: api.TrustSigned, Keys: []string{string(pub)}},
		{Scope: "ghcr.io/acme/scratch", Action: api.TrustAccept},
		{Scope: "localhost:5000", Action: api.TrustAccept},
	}}
	if err := CheckPolicy(p); err != nil {
		t.Fatal(err)
	}
	if p.Rules[1].Scope != "ghcr.io" {
		t.Errorf("scope not normalized: %q", p.Rules[1].Scope)
	}
	for ref, want := range map[string]string{
		"ghcr.io/acme/assistant:v1":                                "ghcr.io/acme",
		"ghcr.io/acme:v1":                                          "ghcr.io/acme",
		"ghcr.io/acmecorp/assistant:v1":                            "ghcr.io",
		"ghcr.io/acme/scratch/x@sha256:" + strings.Repeat("a", 64): "ghcr.io/acme/scratch",
		"registry.example.com/x":                                   "*",
		"localhost:5000/x":                                         "localhost:5000",
	} {
		r, err := reference.Parse(ref)
		if err != nil {
			t.Fatal(err)
		}
		if got := Match(p, r); got == nil || got.Scope != want {
			t.Errorf("%s matched %+v, want %s", ref, got, want)
		}
	}
	if Match(&api.TrustPolicy{}, reference.Reference{Host: "ghcr.io", Path: "x"}) != nil {
		t.Error("an empty policy matched")
	}

	for _, bad := range []api.TrustRule{
		{Scope: "ghcr.io/acme", Action: api.TrustSigned},
		{Scope: "ghcr.io/acme", Action: "maybe"},
		{Scope: "acme", Action: api.TrustAccept},
		{Scope: "ghcr.io/acme:v1", Action: api.TrustAccept},
		{Scope: "ghcr.io", Action: api.TrustAccept, Keys: []string{string(pub)}},
		{Scope: "ghcr.io", Action: api.TrustSigned, Keys: []string{"garbage"}},
		{Scope: "ghcr.io", Action: api.TrustSigned, Identities: []api.Identity{{Subject: "x"}}},
		{Scope: "ghcr.io", Action: api.TrustSigned, Identities: []api.Identity{{SubjectRegexp: "(", Issuer: "i"}}},
	} {
		if err := CheckPolicy(&api.TrustPolicy{Rules: []api.TrustRule{bad}}); err == nil {
			t.Errorf("accepted %+v", bad)
		}
	}
	if err := CheckPolicy(&api.TrustPolicy{Rules: []api.TrustRule{{Scope: "ghcr.io", Action: api.TrustAccept}, {Scope: "ghcr.io/", Action: api.TrustReject}}}); err == nil {
		t.Error("accepted two rules for one scope")
	}

	path := t.TempDir() + "/policy.json"
	if got, err := LoadPolicy(path); err != nil || len(got.Rules) != 0 {
		t.Fatalf("missing policy = %+v, %v", got, err)
	}
	if err := SavePolicy(path, p); err != nil {
		t.Fatal(err)
	}
	got, err := LoadPolicy(path)
	if err != nil || len(got.Rules) != len(p.Rules) || got.Rules[2].Keys[0] != string(pub) {
		t.Fatalf("reloaded = %+v, %v", got, err)
	}
}
