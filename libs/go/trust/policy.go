package trust

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/api"
	"github.com/rizkaitbrik/weightpack/libs/go/reference"
)

// LoadPolicy reads a trust policy file. A missing file is an empty policy,
// which pulls every image.
func LoadPolicy(path string) (*api.TrustPolicy, error) {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return &api.TrustPolicy{}, nil
	}
	if err != nil {
		return nil, err
	}
	var p api.TrustPolicy
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return nil, fmt.Errorf("trust policy %s: %w", path, err)
	}
	if err := CheckPolicy(&p); err != nil {
		return nil, fmt.Errorf("trust policy %s: %w", path, err)
	}
	return &p, nil
}

// SavePolicy checks a policy and writes it to path, replacing the file in
// one step so a pull never reads half of it.
func SavePolicy(path string, p *api.TrustPolicy) error {
	if err := CheckPolicy(p); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".policy-*.json")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// CheckPolicy validates every rule: a known action, a well-formed scope
// named once, and for signed rules at least one parseable key or complete
// identity.
func CheckPolicy(p *api.TrustPolicy) error {
	seen := map[string]bool{}
	for i := range p.Rules {
		r := &p.Rules[i]
		scope, err := NormalizeScope(r.Scope)
		if err != nil {
			return err
		}
		r.Scope = scope
		if seen[scope] {
			return fmt.Errorf("scope %s has two rules", scope)
		}
		seen[scope] = true
		switch r.Action {
		case api.TrustSigned:
			if _, err := NewRequirement(r.Keys, r.Identities); err != nil {
				return fmt.Errorf("scope %s: %w", scope, err)
			}
		case api.TrustAccept, api.TrustReject:
			if len(r.Keys) > 0 || len(r.Identities) > 0 {
				return fmt.Errorf("scope %s: only signed rules name keys or identities", scope)
			}
		default:
			return fmt.Errorf("scope %s: action must be %s, %s or %s, not %q", scope, api.TrustSigned, api.TrustAccept, api.TrustReject, r.Action)
		}
	}
	return nil
}

// NormalizeScope checks a scope and lowercases its host: "*", a registry
// host, or a host and repository path (prefix).
func NormalizeScope(scope string) (string, error) {
	if scope == "*" {
		return scope, nil
	}
	s := strings.TrimSuffix(scope, "/")
	host, path, _ := strings.Cut(s, "/")
	host = strings.ToLower(host)
	probe := host + "/x"
	if path != "" {
		probe = host + "/" + path
	}
	r, err := reference.Parse(probe)
	if err != nil || r.Host == "" || r.Tag != "" || r.Digest != "" {
		return "", fmt.Errorf("scope %q: want *, a registry host (ghcr.io) or a repository prefix (ghcr.io/acme)", scope)
	}
	if path == "" {
		return host, nil
	}
	return host + "/" + path, nil
}

// Match returns the rule for an image: the one whose scope is the longest
// match for its repository, or nil when none matches.
func Match(p *api.TrustPolicy, r reference.Reference) *api.TrustRule {
	name := strings.ToLower(r.Host) + "/" + r.Path
	var best *api.TrustRule
	bestLen := -1
	for i := range p.Rules {
		rule := &p.Rules[i]
		n := 0
		switch {
		case rule.Scope == "*":
		case name == rule.Scope || strings.HasPrefix(name, rule.Scope+"/"):
			n = len(rule.Scope)
		default:
			continue
		}
		if n > bestLen {
			best, bestLen = rule, n
		}
	}
	return best
}
