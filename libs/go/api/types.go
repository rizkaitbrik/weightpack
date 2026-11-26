// Package api holds the HTTP API between wpk and wpkd: request and response
// types, and a client that talks to the daemon over its Unix socket.
//
// Endpoints (all under the daemon's socket):
//
//	GET  /healthz                  daemon status
//	POST /v1/build                 BuildRequest -> NDJSON stream of Event
//	POST /v1/push                  PushRequest -> NDJSON stream of Event
//	POST /v1/pull                  PullRequest -> NDJSON stream of Event
//	POST /v1/tag                   TagRequest -> TagResponse
//	POST /v1/squash                SquashRequest -> NDJSON stream of Event
//	POST /v1/export                ExportRequest -> NDJSON stream of Event
//	POST /v1/rm                    RmRequest -> RmResponse
//	POST /v1/prune                 PruneRequest -> PruneResponse
//	GET  /v1/images                -> []Image
//	GET  /v1/images/inspect?ref=R  -> ImageDetail
//	POST /v1/run                   RunRequest -> NDJSON stream of Event
//	GET  /v1/containers            -> []Container
//	POST /v1/stop                  StopRequest -> StopResponse
//	GET  /v1/containers/logs?name=N[&follow=1]  -> the engine's log, as text
//
// Containers answer on wpkd's gateway, a separate TCP listener that speaks
// the OpenAI API and routes each request by its model field.
package api

import (
	"fmt"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"

	"github.com/rizkaitbrik/weightpack/libs/go/spec"
)

// BuildRequest asks the daemon to build a Weightfile. Paths are absolute:
// the daemon runs on the same machine and reads the build context in place.
type BuildRequest struct {
	Weightfile string   `json:"weightfile"`
	Tags       []string `json:"tags,omitempty"`
	// NoCache reads and hashes every source again.
	NoCache bool `json:"no_cache,omitempty"`
}

// RmRequest untags images.
type RmRequest struct {
	Refs []string `json:"refs"`
}

// RmResponse lists the tags removed.
type RmResponse struct {
	Removed []string `json:"removed"`
}

// PruneRequest asks the daemon to delete blobs no tagged image reaches.
type PruneRequest struct {
	DryRun bool `json:"dry_run,omitempty"`
	// Downloads also removes the hf:// download cache.
	Downloads bool `json:"downloads,omitempty"`
}

// PruneResponse says what a prune removed, or would remove.
type PruneResponse struct {
	Blobs         int   `json:"blobs"`
	Bytes         int64 `json:"bytes"`
	KeptBlobs     int   `json:"kept_blobs"`
	KeptBytes     int64 `json:"kept_bytes"`
	CacheEntries  int   `json:"cache_entries"`
	DownloadBytes int64 `json:"download_bytes"`
	// ServingBytes counts serving files (exported bases and adapters) that
	// no running engine uses; the next wpk run of their images makes them
	// again.
	ServingBytes int64 `json:"serving_bytes"`
}

// RunRequest asks the daemon to serve an image.
type RunRequest struct {
	Ref string `json:"ref"`
	// Name is the model name clients send; it defaults to Ref. An image
	// with a MIX answers to Name/<expert> for each expert.
	Name string `json:"name,omitempty"`
	// Engine is "reference" (transformers and PEFT: cuda, mps or cpu) or
	// "vllm" (cuda). "" picks vllm on NVIDIA GPUs when it is installed and
	// the reference engine otherwise.
	Engine string `json:"engine,omitempty"`
	// Device is "cuda", "mps" or "cpu"; "" detects the best one.
	Device string `json:"device,omitempty"`
}

// StopRequest stops containers by name.
type StopRequest struct {
	Names []string `json:"names"`
}

// StopResponse lists the containers stopped.
type StopResponse struct {
	Stopped []string `json:"stopped"`
}

// Container is a running image: the model names the gateway routes to it
// and the engine that serves them. Containers on the same serving base
// share one engine and one copy of the base weights.
type Container struct {
	Name   string `json:"name"`
	Image  string `json:"image"`
	Digest string `json:"digest"`
	// Models are the names requests use: Name, or Name/<expert> for each
	// expert of a MIX.
	Models []string `json:"models"`
	Engine string   `json:"engine"`
	Device string   `json:"device"`
	// EngineID names the engine process; containers that share it share
	// its log.
	EngineID string `json:"engine_id"`
	// Base is the chain ID of the serving base: the image's base with
	// every layer below its adapters baked in.
	Base string `json:"base"`
	// Adapters are the layers served as LoRA adapters on that base.
	Adapters []string `json:"adapters,omitempty"`
	// Status is "running", or "exited" when the engine died (Error says
	// why; wpk logs shows its output).
	Status  string    `json:"status"`
	Error   string    `json:"error,omitempty"`
	Created time.Time `json:"created"`
	// Endpoint is the gateway's OpenAI base URL.
	Endpoint string `json:"endpoint,omitempty"`
}

// PushRequest asks the daemon to push a local image. Ref must name a
// registry host, such as registry.example.com/assistant:v1.
type PushRequest struct {
	Ref string `json:"ref"`
	// PlainHTTP talks to the registry over HTTP instead of HTTPS. Registries
	// on localhost and 127.0.0.1 always use HTTP.
	PlainHTTP bool `json:"plain_http,omitempty"`
}

// PullRequest asks the daemon to pull an image into its store.
type PullRequest struct {
	Ref       string `json:"ref"`
	PlainHTTP bool   `json:"plain_http,omitempty"`
	// Exact refuses images with lossy layers (delta.bit1, delta.sparse).
	Exact bool `json:"exact,omitempty"`
}

// SquashRequest asks the daemon to bake an image's LoRA layers into its base
// and store the result as a new image with a single base layer.
type SquashRequest struct {
	Ref string `json:"ref"`
	// Expert picks the one expert to bake in when the image has a MIX.
	Expert string   `json:"expert,omitempty"`
	Tags   []string `json:"tags,omitempty"`
}

// ExportRequest asks the daemon to write an image out as files: a Hugging
// Face model directory for the base, a PEFT adapter directory per LoRA
// layer, and a Weightfile that rebuilds the image. Dir is absolute and must
// not exist yet, or be empty.
type ExportRequest struct {
	Ref string `json:"ref"`
	Dir string `json:"dir"`
	// Format is "hf", the only one so far ("" means hf).
	Format string `json:"format,omitempty"`
	// MaxShardSize caps each base weight file, in bytes, as Hugging Face's
	// max_shard_size does. 0 means 5 GB.
	MaxShardSize int64 `json:"max_shard_size,omitempty"`
}

// ExportResult describes a finished export.
type ExportResult struct {
	Dir   string `json:"dir"`
	Files int    `json:"files"`
	Bytes int64  `json:"bytes"`
}

// TagRequest points Target at the image Source names.
type TagRequest struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

// TagResponse is the result of a tag.
type TagResponse struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
}

// Event is one line of a build, squash, push, pull, export or run NDJSON
// stream. The last line carries Result (build, squash), Transfer (push,
// pull), Export, Container (run) or Error.
type Event struct {
	Step      string          `json:"step,omitempty"`
	Message   string          `json:"message,omitempty"`
	Warning   bool            `json:"warning,omitempty"`
	Result    *BuildResult    `json:"result,omitempty"`
	Transfer  *TransferResult `json:"transfer,omitempty"`
	Export    *ExportResult   `json:"export,omitempty"`
	Container *Container      `json:"container,omitempty"`
	Error     string          `json:"error,omitempty"`
}

// TransferResult describes a finished push or pull.
type TransferResult struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
	// Blobs and Size count the image's distinct blobs, config and manifest
	// included; Moved and MovedBytes count those sent or fetched. The other
	// side already held the rest.
	Blobs      int   `json:"blobs"`
	Size       int64 `json:"size"`
	Moved      int   `json:"moved"`
	MovedBytes int64 `json:"moved_bytes"`
	// Tag is the local tag a pull set, if any.
	Tag string `json:"tag,omitempty"`
}

// BuildResult describes a finished build or squash: the image it stored.
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
