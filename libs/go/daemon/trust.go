package daemon

import (
	"context"
	"fmt"
	"net/http"
	"path/filepath"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2/registry/remote"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
	"github.com/rizkaitbrik/weightpack/libs/go/registry"
	"github.com/rizkaitbrik/weightpack/libs/go/serving"
	"github.com/rizkaitbrik/weightpack/libs/go/trust"
)

// WithTrust sets the Sigstore instance keyless signatures chain to. Without
// it, wpkd uses the public-good instance, caching its TUF repository under
// the store root.
func WithTrust(sg *trust.Sigstore) Option { return func(s *server) { s.sigstore = sg } }

// WithPolicyFile reads the trust policy from path instead of
// <root>/policy.json, as from a mounted Kubernetes ConfigMap. wpk trust set
// fails when the file is read-only.
func WithPolicyFile(path string) Option { return func(s *server) { s.policyFile = path } }

// policyPath is the trust policy's file, next to the store by default.
func (s *server) policyPath() string {
	if s.policyFile != "" {
		return s.policyFile
	}
	return filepath.Join(s.store.Root(), "policy.json")
}

func (s *server) verifier() *trust.Verifier { return &trust.Verifier{Sigstore: s.sigstore} }

// pullOptions are the registry options of every pull: the trust policy
// checks each image before its tensors are fetched.
func (s *server) pullOptions(plainHTTP, exact bool) registry.Options {
	opts := s.registryOptions(plainHTTP)
	opts.Exact, opts.Verify = exact, s.checkPolicy
	return opts
}

// WithInsecureRegistries names registries wpkd always reaches over plain
// HTTP (wpkd --insecure-registries).
func WithInsecureRegistries(hosts []string) Option {
	return func(s *server) { s.insecure = hosts }
}

// registryOptions are the options of every registry operation.
func (s *server) registryOptions(plainHTTP bool) registry.Options {
	return registry.Options{PlainHTTP: plainHTTP, InsecureHosts: s.insecure}
}

// checkPolicy applies the trust policy's rule for r to the manifest root.
func (s *server) checkPolicy(ctx context.Context, r reference.Reference, repo *remote.Repository, root ocispec.Descriptor) (string, error) {
	policy, err := trust.LoadPolicy(s.policyPath())
	if err != nil {
		return "", err
	}
	rule := trust.Match(policy, r)
	if rule == nil {
		return "", nil
	}
	switch rule.Action {
	case api.TrustAccept:
		return "", nil
	case api.TrustReject:
		return "", fmt.Errorf("the trust policy rejects images from %s (wpk trust ls shows the rules)", rule.Scope)
	}
	req, err := trust.NewRequirement(rule.Keys, rule.Identities)
	if err != nil {
		return "", err
	}
	res, err := s.verifier().Verify(ctx, repo, root, req)
	if err != nil {
		return "", fmt.Errorf("the trust policy for %s requires a trusted signature: %w", rule.Scope, err)
	}
	s.log.Info("signature verified", "ref", r.String(), "digest", root.Digest, "signer", res.Signer, "rule", rule.Scope)
	return res.Signer, nil
}

func (s *server) resolve(w http.ResponseWriter, r *http.Request) {
	var req api.ResolveRequest
	if !decode(w, r, &req) {
		return
	}
	ref, repo, root, err := registry.Resolve(r.Context(), req.Ref, "resolve", s.registryOptions(req.PlainHTTP))
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	out := api.ResolveResponse{Ref: ref.String(), Digest: root.Digest.String(), MediaType: root.MediaType, Size: root.Size}
	if req.Placement {
		// What a pull checks first: the trust policy, then the manifest and
		// config. A control plane then never places an image the node
		// would refuse.
		if out.Signer, err = s.checkPolicy(r.Context(), ref, repo, root); err != nil {
			writeError(w, http.StatusForbidden, fmt.Errorf("%s: %w", ref, err))
			return
		}
		m, cfg, err := registry.Describe(r.Context(), repo, ref, root)
		if err != nil {
			writeError(w, http.StatusBadGateway, err)
			return
		}
		if out.ServingBase, err = serving.ServingBase(cfg); err != nil {
			writeError(w, http.StatusUnprocessableEntity, fmt.Errorf("%s cannot be served: %w", ref, err))
			return
		}
		if len(cfg.Layers) > 0 {
			out.Base = cfg.Layers[0].Chain
		}
		out.Bytes = root.Size + m.Config.Size
		for _, l := range m.Layers {
			out.Bytes += l.Size
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// attach stores a bundle wpk made as a referrer of the manifest it signs.
// The manifest must be in the repository.
func (s *server) attach(w http.ResponseWriter, r *http.Request) {
	var req api.AttachRequest
	if !decode(w, r, &req) {
		return
	}
	d, err := digest.Parse(req.Digest)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("digest: %w", err))
		return
	}
	ref, repo, err := registry.Open(req.Ref, "sign", s.registryOptions(req.PlainHTTP))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	subject, err := repo.Resolve(r.Context(), d.String())
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("sign %s: find %s: %w", ref.Name(), d, err))
		return
	}
	sig, err := trust.Attach(r.Context(), repo, subject, req.Bundle)
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("sign %s@%s: %w", ref.Name(), d, err))
		return
	}
	s.log.Info("signature attached", "repository", ref.Name(), "digest", d, "signature", sig.Digest)
	writeJSON(w, http.StatusOK, api.AttachResponse{Ref: ref.Name() + "@" + d.String(), Digest: d.String(), Signature: sig.Digest.String()})
}

// verify checks a remote image against the keys and identities the request
// names, or else against the trust policy.
func (s *server) verify(w http.ResponseWriter, r *http.Request) {
	var req api.VerifyRequest
	if !decode(w, r, &req) {
		return
	}
	ref, repo, root, err := registry.Resolve(r.Context(), req.Ref, "verify", s.registryOptions(req.PlainHTTP))
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	out := api.VerifyResponse{Ref: ref.String(), Digest: root.Digest.String()}
	if len(req.Keys) == 0 && len(req.Identities) == 0 {
		policy, err := trust.LoadPolicy(s.policyPath())
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		rule := trust.Match(policy, ref)
		if rule == nil || rule.Action != api.TrustSigned {
			if rule != nil {
				out.Rule, out.Action = rule.Scope, rule.Action
			} else {
				out.Action = api.TrustAccept
			}
			bundles, err := trust.Bundles(r.Context(), repo, root)
			if err != nil {
				writeError(w, http.StatusBadGateway, err)
				return
			}
			out.Signatures = len(bundles)
			writeJSON(w, http.StatusOK, out)
			return
		}
		out.Rule, out.Action = rule.Scope, rule.Action
		req.Keys, req.Identities = rule.Keys, rule.Identities
	}
	requirement, err := trust.NewRequirement(req.Keys, req.Identities)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	res, err := s.verifier().Verify(r.Context(), repo, root, requirement)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, fmt.Errorf("verify %s: %w", ref, err))
		return
	}
	out.Signer, out.Signatures = res.Signer, res.Signatures
	writeJSON(w, http.StatusOK, out)
}

func (s *server) getTrust(w http.ResponseWriter, _ *http.Request) {
	p, err := trust.LoadPolicy(s.policyPath())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if p.Rules == nil {
		p.Rules = []api.TrustRule{}
	}
	writeJSON(w, http.StatusOK, p)
}

func (s *server) putTrust(w http.ResponseWriter, r *http.Request) {
	var p api.TrustPolicy
	if !decode(w, r, &p) {
		return
	}
	s.trustMu.Lock()
	defer s.trustMu.Unlock()
	if err := trust.SavePolicy(s.policyPath(), &p); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.log.Info("trust policy changed", "rules", len(p.Rules))
	if p.Rules == nil {
		p.Rules = []api.TrustRule{}
	}
	writeJSON(w, http.StatusOK, p)
}
