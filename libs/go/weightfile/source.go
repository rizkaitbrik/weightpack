package weightfile

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// SourceKind says where a FROM, LAYER or EXPERT source comes from.
type SourceKind string

const (
	// SourceLocal is a path on disk, relative to the Weightfile's directory
	// unless absolute.
	SourceLocal SourceKind = "local"
	// SourceOCI is an image or layer in an OCI registry.
	SourceOCI SourceKind = "oci"
	// SourceHF is a Hugging Face Hub repository, optionally at a revision.
	SourceHF SourceKind = "hf"
)

// Source is a parsed source reference.
type Source struct {
	Kind SourceKind
	// Ref is the reference without its scheme: a path, an OCI reference such
	// as registry.example.com/llama@sha256:…, or an HF repo such as
	// acme/llama-fin[@revision].
	Ref string
}

// String returns the source in the form a Weightfile accepts.
func (s Source) String() string {
	switch s.Kind {
	case SourceHF:
		return "hf://" + s.Ref
	case SourceOCI:
		return "oci://" + s.Ref
	default:
		return s.Ref
	}
}

// Path resolves a local source against the Weightfile's directory. It returns
// "" for other kinds.
func (s Source) Path(dir string) string {
	if s.Kind != SourceLocal {
		return ""
	}
	if filepath.IsAbs(s.Ref) {
		return filepath.Clean(s.Ref)
	}
	return filepath.Join(dir, s.Ref)
}

var (
	// OCI distribution reference grammar, slightly simplified: a registry
	// host is only recognised when the first component has a '.' or ':' or
	// is "localhost", as Docker does.
	ociComponent = regexp.MustCompile(`^[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*$`)
	ociHost      = regexp.MustCompile(`^(?:localhost|[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*)(?::[0-9]+)?$`)
	ociTag       = regexp.MustCompile(`^[\w][\w.-]{0,127}$`)
	ociDigest    = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
	hfRepo       = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
	hfRevision   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*$`)
)

// ParseSource classifies a source string:
//
//   - "hf://org/name[@revision]" is a Hugging Face repo;
//   - "oci://ref" is an OCI reference;
//   - ".", or a string starting with "./", "../" or "/", is a local path;
//   - anything else is read as an OCI reference, as Docker reads FROM.
func ParseSource(s string) (Source, error) {
	switch {
	case s == "":
		return Source{}, fmt.Errorf("empty source")
	case strings.HasPrefix(s, "hf://"):
		ref := strings.TrimPrefix(s, "hf://")
		if err := checkHF(ref); err != nil {
			return Source{}, err
		}
		return Source{Kind: SourceHF, Ref: ref}, nil
	case strings.HasPrefix(s, "oci://"):
		ref := strings.TrimPrefix(s, "oci://")
		if err := CheckOCIReference(ref); err != nil {
			return Source{}, err
		}
		return Source{Kind: SourceOCI, Ref: ref}, nil
	case s == "." || s == ".." || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") || strings.HasPrefix(s, "/"):
		return Source{Kind: SourceLocal, Ref: s}, nil
	case strings.HasPrefix(s, "~"):
		return Source{}, fmt.Errorf("source %q: '~' is not expanded; use a path relative to the Weightfile", s)
	case strings.Contains(s, "://"):
		return Source{}, fmt.Errorf("source %q: unknown scheme (want hf://, oci:// or a local path)", s)
	}
	if err := CheckOCIReference(s); err != nil {
		return Source{}, fmt.Errorf("%w (local paths start with ./ or /)", err)
	}
	return Source{Kind: SourceOCI, Ref: s}, nil
}

func checkHF(ref string) error {
	repo, rev, hasRev := strings.Cut(ref, "@")
	if !hfRepo.MatchString(repo) {
		return fmt.Errorf("hf source %q: want hf://<org>/<name>[@revision]", ref)
	}
	if hasRev && !hfRevision.MatchString(rev) {
		return fmt.Errorf("hf source %q: bad revision %q", ref, rev)
	}
	return nil
}

// CheckOCIReference validates a reference of the form
// [host[:port]/]path[:tag][@sha256:<hex>].
func CheckOCIReference(ref string) error {
	name, digest, hasDigest := strings.Cut(ref, "@")
	if hasDigest && !ociDigest.MatchString(digest) {
		return fmt.Errorf("reference %q: digest must be sha256:<64 hex>", ref)
	}
	if i := strings.LastIndex(name, ":"); i > strings.LastIndex(name, "/") {
		tag := name[i+1:]
		name = name[:i]
		if !ociTag.MatchString(tag) {
			return fmt.Errorf("reference %q: bad tag %q", ref, tag)
		}
	}
	if name == "" {
		return fmt.Errorf("reference %q: missing repository", ref)
	}
	parts := strings.Split(name, "/")
	if len(parts) > 1 && (strings.ContainsAny(parts[0], ".:") || parts[0] == "localhost") {
		if !ociHost.MatchString(parts[0]) {
			return fmt.Errorf("reference %q: bad registry host %q", ref, parts[0])
		}
		parts = parts[1:]
	}
	for _, p := range parts {
		if !ociComponent.MatchString(p) {
			return fmt.Errorf("reference %q: bad path component %q", ref, p)
		}
	}
	return nil
}
