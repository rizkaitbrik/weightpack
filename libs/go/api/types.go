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
//	GET  /v1/doctor                -> DoctorReport
//	GET  /v1/node                  -> Node
//	POST /v1/resolve               ResolveRequest -> ResolveResponse
//	POST /v1/signatures            AttachRequest -> AttachResponse
//	POST /v1/verify                VerifyRequest -> VerifyResponse
//	GET  /v1/trust                 -> TrustPolicy
//	PUT  /v1/trust                 TrustPolicy -> TrustPolicy
//
// Containers answer on wpkd's gateway, a separate TCP listener that speaks
// the OpenAI API and routes each request by its model field.
package api

import (
	"encoding/json"
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
	// Owner names what started the container, such as a cluster control
	// plane, which manages only its own.
	Owner string `json:"owner,omitempty"`
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
	// Adapters are the layers served on that base at request time: LoRA
	// adapters, or a full fine-tune's bit1 layer on the reference engine.
	Adapters []string `json:"adapters,omitempty"`
	// Router is set ("arrow") when the model named Name routes each token
	// across a MIX's experts; Name/<expert> still picks one expert.
	Router string `json:"router,omitempty"`
	// Status is "running", or "exited" when the engine died (Error says
	// why; wpk logs shows its output).
	Status  string    `json:"status"`
	Error   string    `json:"error,omitempty"`
	Created time.Time `json:"created"`
	// Endpoint is the gateway's OpenAI base URL.
	Endpoint string `json:"endpoint,omitempty"`
	// Owner is the run request's Owner.
	Owner string `json:"owner,omitempty"`
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
	// Verified names who signed a pulled image, when the trust policy
	// asked for a signature.
	Verified string `json:"verified,omitempty"`
}

// ResolveRequest asks which manifest a remote reference names.
type ResolveRequest struct {
	Ref       string `json:"ref"`
	PlainHTTP bool   `json:"plain_http,omitempty"`
	// Placement also checks the image against the trust policy, as a pull
	// would, and reads its config, to say where it could run.
	Placement bool `json:"placement,omitempty"`
}

// ResolveResponse is the manifest a remote reference names.
type ResolveResponse struct {
	Ref       string `json:"ref"`
	Digest    string `json:"digest"`
	MediaType string `json:"media_type"`
	Size      int64  `json:"size"`
	// With Placement: the chain IDs of the image's base layer and of the
	// base an engine serves it on, the total size of its blobs, and who
	// signed it when the policy asks for a signature.
	Base        string `json:"base,omitempty"`
	ServingBase string `json:"serving_base,omitempty"`
	Bytes       int64  `json:"bytes,omitempty"`
	Signer      string `json:"signer,omitempty"`
}

// AttachRequest asks the daemon to store a Sigstore bundle that signs the
// manifest Digest in Ref's repository, as cosign does: an OCI referrer of
// the manifest. wpk makes the bundle, so keys and identity tokens never
// reach the daemon.
type AttachRequest struct {
	Ref       string          `json:"ref"`
	Digest    string          `json:"digest"`
	Bundle    json.RawMessage `json:"bundle"`
	PlainHTTP bool            `json:"plain_http,omitempty"`
}

// AttachResponse names the stored signature.
type AttachResponse struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
	// Signature is the digest of the referrer manifest that holds the bundle.
	Signature string `json:"signature"`
}

// VerifyRequest asks the daemon to check a remote image's signatures:
// against Keys and Identities when given, or else against the trust policy.
type VerifyRequest struct {
	Ref        string     `json:"ref"`
	PlainHTTP  bool       `json:"plain_http,omitempty"`
	Keys       []string   `json:"keys,omitempty"`
	Identities []Identity `json:"identities,omitempty"`
}

// VerifyResponse says who signed the image.
type VerifyResponse struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
	// Signer is who made the signature that satisfied the check, or "" when
	// the policy's rule accepts unsigned images.
	Signer string `json:"signer,omitempty"`
	// Rule is the scope of the policy rule that applied, if the policy did.
	Rule string `json:"rule,omitempty"`
	// Action is what that rule does with the image.
	Action string `json:"action,omitempty"`
	// Signatures counts the Sigstore signatures attached to the image.
	Signatures int `json:"signatures"`
}

// Trust rule actions.
const (
	TrustSigned = "signed" // pull only images signed by one of the rule's keys or identities
	TrustAccept = "accept" // pull images whether signed or not
	TrustReject = "reject" // pull nothing
)

// TrustPolicy says which images wpkd pulls. For each pull the rule whose
// scope names the image most closely applies; when no rule matches, and
// with no policy at all, wpkd pulls every image.
type TrustPolicy struct {
	Rules []TrustRule `json:"rules"`
}

// TrustRule is the policy for the images under one scope.
type TrustRule struct {
	// Scope is a registry host ("ghcr.io"), a repository or a prefix of
	// one at a path boundary ("ghcr.io/acme", "ghcr.io/acme/assistant"), or
	// "*" for every image.
	Scope string `json:"scope"`
	// Action is TrustSigned, TrustAccept or TrustReject.
	Action string `json:"action"`
	// Keys are PEM public keys (as cosign generate-key-pair or wpk keygen
	// write them). A signature by any one of them satisfies the rule.
	Keys []string `json:"keys,omitempty"`
	// Identities are keyless signers. A Sigstore signature whose Fulcio
	// certificate matches any one of them satisfies the rule.
	Identities []Identity `json:"identities,omitempty"`
}

// Identity is a keyless signer: who signed (the certificate's subject, an
// email or a CI workflow URI) and who vouched for it (the OIDC issuer).
// Each takes an exact value or a regular expression, which must match the
// whole value.
type Identity struct {
	Issuer        string `json:"issuer,omitempty"`
	IssuerRegexp  string `json:"issuer_regexp,omitempty"`
	Subject       string `json:"subject,omitempty"`
	SubjectRegexp string `json:"subject_regexp,omitempty"`
}

// String prints the identity as wpk trust shows it: a regular expression
// is marked with a leading ~.
func (id Identity) String() string {
	subject, issuer := id.Subject, id.Issuer
	if subject == "" {
		subject = "~" + id.SubjectRegexp
	}
	if issuer == "" {
		issuer = "~" + id.IssuerRegexp
	}
	return subject + " (issuer " + issuer + ")"
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
	// Base is the chain ID of the image's base layer. Images with the same
	// base share its blobs.
	Base string `json:"base,omitempty"`
}

// ImageDetail is what `wpk inspect` prints.
type ImageDetail struct {
	Ref      string           `json:"ref,omitempty"`
	Digest   string           `json:"digest"`
	Manifest ocispec.Manifest `json:"manifest"`
	Config   spec.Config      `json:"config"`
}

// Node describes the machine a daemon runs on, for a cluster control plane
// placing images: whether it serves, where, and how loaded it is.
type Node struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	// Serving is false when wpkd runs without a gateway.
	Serving bool `json:"serving"`
	// GatewayPort is the OpenAI gateway's port, on the same host as the API.
	GatewayPort int `json:"gateway_port,omitempty"`
	// Device is the default device containers run on; Devices are all the
	// devices they can run on, and EngineKinds the engines installed
	// ("reference", and "vllm" on an NVIDIA GPU with vLLM).
	Device      string   `json:"device,omitempty"`
	Devices     []string `json:"devices,omitempty"`
	EngineKinds []string `json:"engine_kinds,omitempty"`
	// Engines counts the engine processes running; MaxEngines caps them
	// (0: no cap). A container on a base no engine serves yet needs one.
	Engines    int `json:"engines"`
	MaxEngines int `json:"max_engines,omitempty"`
}

// Check statuses.
const (
	CheckOK   = "ok"
	CheckWarn = "warn" // works, with a limit worth knowing
	CheckFail = "fail" // something wpk needs does not work
	CheckSkip = "skip" // does not apply here
)

// Check is one line of wpk doctor.
type Check struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Detail string `json:"detail"`
	// Fix says what to do about a warn or fail.
	Fix string `json:"fix,omitempty"`
}

// DoctorReport is what wpkd finds about its machine: the build worker, the
// device, the engines, the gateway, the store and Hugging Face access.
type DoctorReport struct {
	Checks []Check `json:"checks"`
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
