// Package reference parses OCI image references of the form
// [host[:port]/]path[:tag][@sha256:<hex>].
//
// The grammar is the OCI distribution one, slightly simplified: a registry
// host is only recognised when the first component has a '.' or ':' or is
// "localhost", as Docker does. Names are kept as written: there is no
// docker.io or library/ expansion.
package reference

import (
	"fmt"
	"regexp"
	"strings"
)

// DefaultTag is the tag a reference gets when it has neither tag nor digest.
const DefaultTag = "latest"

var (
	component = regexp.MustCompile(`^[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*$`)
	host      = regexp.MustCompile(`^(?:localhost|[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*)(?::[0-9]+)?$`)
	tag       = regexp.MustCompile(`^[\w][\w.-]{0,127}$`)
	digest    = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
)

// Reference is a parsed image reference.
type Reference struct {
	Host   string // registry host, "" if none was written
	Path   string // repository path, such as "experts/medical"
	Tag    string // "" if none was written
	Digest string // "sha256:<hex>", "" if none was written
}

// Name returns the repository name: [host/]path.
func (r Reference) Name() string {
	if r.Host == "" {
		return r.Path
	}
	return r.Host + "/" + r.Path
}

// String returns the reference in the form Parse accepts.
func (r Reference) String() string {
	s := r.Name()
	if r.Tag != "" {
		s += ":" + r.Tag
	}
	if r.Digest != "" {
		s += "@" + r.Digest
	}
	return s
}

// Parse validates and splits a reference.
func Parse(ref string) (Reference, error) {
	var out Reference
	name, dg, hasDigest := strings.Cut(ref, "@")
	if hasDigest {
		if !digest.MatchString(dg) {
			return Reference{}, fmt.Errorf("reference %q: digest must be sha256:<64 hex>", ref)
		}
		out.Digest = dg
	}
	if i := strings.LastIndex(name, ":"); i > strings.LastIndex(name, "/") {
		out.Tag = name[i+1:]
		name = name[:i]
		if !tag.MatchString(out.Tag) {
			return Reference{}, fmt.Errorf("reference %q: bad tag %q", ref, out.Tag)
		}
	}
	if name == "" {
		return Reference{}, fmt.Errorf("reference %q: missing repository", ref)
	}
	parts := strings.Split(name, "/")
	if len(parts) > 1 && (strings.ContainsAny(parts[0], ".:") || parts[0] == "localhost") {
		if !host.MatchString(parts[0]) {
			return Reference{}, fmt.Errorf("reference %q: bad registry host %q", ref, parts[0])
		}
		out.Host = parts[0]
		parts = parts[1:]
	}
	for _, p := range parts {
		if !component.MatchString(p) {
			return Reference{}, fmt.Errorf("reference %q: bad path component %q", ref, p)
		}
	}
	out.Path = strings.Join(parts, "/")
	return out, nil
}

// ParseNamed parses a reference that names an image by tag, as `wpk build -t`
// does: a digest is not allowed, and a missing tag becomes DefaultTag.
func ParseNamed(ref string) (Reference, error) {
	r, err := Parse(ref)
	if err != nil {
		return Reference{}, err
	}
	if r.Digest != "" {
		return Reference{}, fmt.Errorf("reference %q: a tag cannot include a digest", ref)
	}
	if r.Tag == "" {
		r.Tag = DefaultTag
	}
	return r, nil
}

// IsDigest reports whether s is a bare "sha256:<64 hex>" digest.
func IsDigest(s string) bool { return digest.MatchString(s) }
