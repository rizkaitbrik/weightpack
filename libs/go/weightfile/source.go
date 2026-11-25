package weightfile

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/rizkaitbrik/weightpack/libs/go/reference"
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

// HF splits a Hugging Face source into its repository ("org/name"), the
// folder inside it ("" for the root) and the revision ("main" when none is
// given). It returns empty strings for other kinds.
func (s Source) HF() (repo, folder, rev string) {
	if s.Kind != SourceHF {
		return "", "", ""
	}
	path, rev, _ := strings.Cut(s.Ref, "@")
	if rev == "" {
		rev = "main"
	}
	parts := strings.SplitN(path, "/", 3)
	repo = parts[0] + "/" + parts[1]
	if len(parts) == 3 {
		folder = parts[2]
	}
	return repo, folder, rev
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
	// org/name, then optionally a folder inside the repository.
	hfRepo     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9._-]+)*$`)
	hfRevision = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*$`)
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
	case isLocalPath(s):
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
		return fmt.Errorf("hf source %q: want hf://<org>/<name>[/<folder>][@revision]", ref)
	}
	for _, part := range strings.Split(repo, "/") {
		if part == "." || part == ".." {
			return fmt.Errorf("hf source %q: a folder cannot be %q", ref, part)
		}
	}
	if hasRev && !hfRevision.MatchString(rev) {
		return fmt.Errorf("hf source %q: bad revision %q", ref, rev)
	}
	return nil
}

// CheckOCIReference validates a reference of the form
// [host[:port]/]path[:tag][@sha256:<hex>].
func CheckOCIReference(ref string) error {
	_, err := reference.Parse(ref)
	return err
}

// isLocalPath reports whether s is written as a local path: ".", "..", or
// starting with "./", "../" or "/".
func isLocalPath(s string) bool {
	return s == "." || s == ".." || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") || strings.HasPrefix(s, "/")
}
