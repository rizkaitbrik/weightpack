// Package api holds the HTTP API between wpk and wpkd: request and response
// types, and a client that talks to the daemon over its Unix socket.
//
// Endpoints (all under the daemon's socket):
//
//	GET  /healthz                  daemon status
//	POST /v1/build                 BuildRequest -> NDJSON stream of Event
//	GET  /v1/images                -> []Image
//	GET  /v1/images/inspect?ref=R  -> ImageDetail
package api

import (
	"fmt"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

// BuildRequest asks the daemon to build a Weightfile. Paths are absolute:
// the daemon runs on the same machine and reads the build context in place.
type BuildRequest struct {
	Weightfile string   `json:"weightfile"`
	Tags       []string `json:"tags,omitempty"`
}

// Event is one line of a build's NDJSON stream. The last line carries either
// Result or Error.
type Event struct {
	Step    string       `json:"step,omitempty"`
	Message string       `json:"message,omitempty"`
	Warning bool         `json:"warning,omitempty"`
	Result  *BuildResult `json:"result,omitempty"`
	Error   string       `json:"error,omitempty"`
}

// BuildResult describes a finished build.
type BuildResult struct {
	Manifest ocispec.Descriptor `json:"manifest"`
	Config   *spec.Config       `json:"config"`
	Tags     []string           `json:"tags,omitempty"`
	// Blobs and Size count the image's distinct blobs, config and manifest
	// included; NewBlobs and NewBytes count those the store did not hold.
	Blobs    int   `json:"blobs"`
	Size     int64 `json:"size"`
	NewBlobs int   `json:"new_blobs"`
	NewBytes int64 `json:"new_bytes"`
}

// Image is one tagged image in the local store.
type Image struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
	// Size is the total of the image's blobs. Images that share a base share
	// its blobs, so sizes do not add up to the store's disk use.
	Size   int64  `json:"size"`
	Layers int    `json:"layers"`
	Arch   string `json:"arch,omitempty"`
}

// ImageDetail is what `wpk inspect` prints.
type ImageDetail struct {
	Ref      string           `json:"ref,omitempty"`
	Digest   string           `json:"digest"`
	Manifest ocispec.Manifest `json:"manifest"`
	Config   spec.Config      `json:"config"`
}

// FormatSize prints a byte count in decimal units, as the design doc does
// ("16.1 GB").
func FormatSize(n int64) string {
	const unit = 1000
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "kMGTPE"[exp])
}
