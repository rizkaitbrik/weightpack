# Weightpack — Docker for neural nets

Status: draft · Last updated 2026-11-24

## Summary

Weightpack is Docker for model weights. A model ships as a base plus content-addressed delta layers, is built from a Weightfile, and is pushed to any OCI registry. It adds one thing Docker lacks: layers can sit side by side as experts behind a router, which gives you a mixture-of-experts without copying the base. A base 8B model plus 50 LoRA variants takes about 20 GB, against about 820 GB as full copies.

It has three parts:

1. **Format.** An OCI artifact with one blob per tensor. Each layer pins its parent chain and the base's architecture fingerprint.
2. **Composition.** A Weightfile with `FROM`, `LAYER` (stack), `MIX` / `EXPERT` / `ROUTER` (mix), plus `wpk squash` to bake the result into dense weights.
3. **Store and runtime.** On disk: tensor-level dedup plus delta codecs. At serve time: the base loads once, expert deltas page in, and a router picks them per request, sequence or token.

The main design decision is to make adapter-style layers (LoRA and similar) the native unit. Full fine-tunes change every weight, so dedup alone saves little on them. They come in through an import step that stores a compressed delta and records how much quality it lost.

## Problem and scope

Teams end up with dozens of variants of one base model: per domain, per client, per language. Each is stored, shipped and loaded as a full copy. Combining two of them (legal and French, say) means retraining or a lossy weight merge.

**Terms.** In this doc a *layer* is a Docker-style layer: a set of weight changes over a parent. A transformer layer is called a *block*. An *expert* is a layer mounted in a mix behind a router. An *image* is a named composition of layers, and a *container* is a running instance of one.

**Goals**

- **G1. Store only what changed.** Each variant costs its delta. Push and pull move only blobs the other side lacks.
- **G2. Compose independently built layers.** Stack them or mix them, declared in a file and reproducible by digest.
- **G3. Serve many compositions at once.** The base sits in GPU memory once, shared by every container on that GPU.
- **G4. Reuse existing infrastructure.** OCI registries, signing, safetensors, the Hugging Face Hub, existing inference engines.
- **G5. Make every lossy step visible.** Each lossy layer records its codec and its measured quality drop. Exact pulls can refuse it.

**Non-goals**

- Mixing models of different architectures or tokenizers. All layers in an image descend from one base.
- Training. Weightpack calls existing trainers (PEFT, TRL, Axolotl) and caches their output; it does not replace them.
- Guaranteeing that any combination is good. The build checks compatibility; quality is measured by eval hooks.

## Prior art and the gap

Each piece of this exists today. What is missing is one system where a layer is a typed delta with a pinned parent, layers can be stacked or mixed, and a mix is served without first being turned into a full checkpoint. Weightpack is mostly integration; the least mature piece is token-level routing over deltas inside a production engine.

| Project | What it already does | What it lacks here | What Weightpack takes |
| --- | --- | --- | --- |
| [Ollama Modelfile](https://docs.ollama.com/modelfile) and [manifest](https://pkg.go.dev/github.com/ollama/ollama@v0.12.0/server/internal/manifest) | `FROM` + `ADAPTER`; content-addressed blobs; tensors stored as manifest layers | No mixing; an adapter on the wrong base is undefined behaviour, not a build error | Build-file UX, per-tensor blobs |
| [CNCF ModelPack](https://github.com/modelpack/model-spec), [Docker Model Runner](https://docs.docker.com/reference/cli/docker/model/package/) | OCI artifact spec for models; Docker can emit ModelPack format | Weight layers are whole files, not deltas over a parent; no composition | The envelope. Weightpack should be a ModelPack extension, not a rival |
| [Hugging Face Xet](https://huggingface.co/docs/hub/xet/index.md) | Chunk-level dedup (about 64 KB chunks); only changed chunks move | Byte-level, not tensor-aware; a full fine-tune changes almost every chunk | Optional storage backend |
| [ZipLLM](https://arxiv.org/abs/2505.06252) (NSDI '26) | Tensor-level dedup plus lossless XOR delta (BitX); 54% less storage | Storage only; no composition or serving | Lossless delta tier |
| [ZipNN](https://arxiv.org/abs/2411.05239) | Lossless; compresses float exponents apart; BF16 Llama 3.1 shrinks to 66.4% | Single-model, no deltas | Entropy-coding stage |
| [BitDelta](https://arxiv.org/abs/2402.10193) | Fine-tune delta stored as 1 sign bit per weight plus a scale per matrix | Lossy; needs a calibration pass | Lossy tier for full fine-tunes |
| [S-LoRA](https://arxiv.org/abs/2311.03285) | Thousands of LoRAs on one GPU: unified paging, batched kernels | One adapter per request; no packaging | Runtime memory design |
| [BTX](https://arxiv.org/abs/2403.07816), [mergekit-moe](https://huggingface.co/blog/mlabonne/frankenmoe), [MixtureKit](https://arxiv.org/abs/2512.12121) | Build an MoE from dense fine-tunes: their FFNs become experts, the rest is averaged or taken from a base | Output is one monolithic checkpoint; nothing deduplicated | "FFN expert" layer kind, router init |
| [Arrow](https://arxiv.org/abs/2405.11157) (ICML 2024, now in PEFT) | Zero-shot routing over a LoRA library, from each LoRA's SVD; no router training | Research-grade; no packaging | Default router for independently built layers |

## Core model: stack, mix, squash

Layers combine in three ways: stacked on top of each other (the only mode Docker has), mixed side by side behind a router, or squashed into plain weights.

```mermaid
flowchart BT
  base["Base · Llama 3.1 8B Instruct · 16.1 GB, one blob per tensor<br/>(stored and loaded once)"]
  french["LAYER french · LoRA rank 32<br/>(one set of blobs used by both images)"]
  base --> french
  subgraph A["Image A · assistant"]
    direction BT
    router["Router · Arrow, top-2 per token"]
    legal["legal · LoRA r16"]
    medical["medical · LoRA r16"]
    finance["finance · 1-bit delta, eval-gated"]
    router --> legal & medical & finance
  end
  subgraph B["Image B · support"]
    support["LAYER support · LoRA r16<br/>plain stack, squashable"]
  end
  french --> A
  french --> B
```

Image B pulls nothing new except its top layer, because the base and `french` blobs are already local. In image A the router sends each token to two of the three experts.

| Layer kind | Holds | Size for an 8B base |
| --- | --- | --- |
| `base` | Every tensor of a released model | 16.1 GB in BF16 |
| `lora` | A and B matrices for its target modules | 84 MB at rank 16, all modules |
| `delta` | An encoded difference over full-shape tensors (`xor`, `bit1`, `sparse`) | About 1 GB at 1 bit per weight |
| `ffn-expert` | A replacement FFN per block, BTX-style, stored as a delta over the base FFN | Depends on the codec |
| `router` | Per-block gate weights, or Arrow prototypes | One vector per expert per module |
| `meta` | Tokenizer, chat template, license, model card | Kilobytes |

**Stack.** Deltas add up in the order of the chain:

$$W = W_0 + \Delta_1 + \Delta_2 + \dots$$

Each delta was trained on everything below it, so the chain IDs fix the order.

**Mix.** In each targeted module of the chosen blocks, the router g picks the top k experts per token:

$$y = W x + \sum_{i \in \mathrm{TopK}(g(x))} g_i(x)\, \Delta_i x$$

The base stays on for every token and acts as a shared expert. The routed experts are deltas, not copies of the FFN, which is why a mix of three experts costs little more than the base.

**Squash.** Bake a stack, or a mix with fixed weights, into one delta or into dense weights. A mix with per-token routing cannot be squashed: the routing is what makes it a mixture of experts.

`wpk squash` bakes a stack of LoRA layers, or a stack plus one chosen expert of a mix (`--expert`), into dense weights and stores the result as a new image with a single `base` layer. For every base tensor a layer modifies it computes `W + Σ scalingᵢ · (Bᵢ Aᵢ)` in float32, layers in chain order, and rounds once, to nearest even, to the tensor's dtype. That is what PEFT's `merge_and_unload` does for a float32 adapter, and the parity test (`npx nx run worker:parity`) checks it bit for bit on a BF16 Llama, for plain and rsLoRA scaling. Tensors no layer touches keep their blobs, so a squash stores only what it changed, and the tensor table, and so the architecture fingerprint, is the base's. The manifest records what was baked: `wpk.squashed` lists the layer ids, bottom first, and `wpk.squashed.from` holds the source image's manifest digest.

## System architecture

Weightpack splits into three planes, as Docker does: a build host that turns a Weightfile into blobs, a standard OCI registry that stores them, and GPU nodes that serve images through vLLM with a Weightpack plugin. A small control plane routes requests and places each image on a node where its base is already loaded.

```mermaid
flowchart TB
  subgraph build["Build host · wpk CLI + wpkd daemon"]
    direction LR
    resolver["Resolver<br/>Weightfile, chain IDs, fingerprint"] --> train["Train + import<br/>PEFT/TRL jobs or delta vs parent"]
    train --> encode["Encode + gate<br/>xor, bit1, sparse, ZipNN; eval gate"]
    encode --> store["Local store<br/>tensor CAS, SQLite, build cache"]
  end
  registry["OCI registry<br/>manifests + tensor blobs by digest<br/>Sigstore signatures as referrers"]
  store -- "wpk push" --> registry
  subgraph cp["Control plane"]
    scheduler["Scheduler<br/>sends each image to a node holding its base"]
    gateway["API gateway<br/>OpenAI-compatible; model name to digest"]
    catalogue["Catalogue + metrics"]
  end
  subgraph node["GPU node"]
    agent["Node agent (wpkd)<br/>pulls missing blobs, verifies, decodes, prefetches"]
    subgraph vllm["vLLM + Weightpack plugin"]
      loader["Composition loader"]
      pool["Expert pool"]
      rt["Router"]
      kernels["Kernels: LoRA, bit1, FFN expert"]
    end
    agent --> loader
  end
  registry -- "wpk pull, missing blobs only" --> agent
  scheduler --> agent
  gateway --> vllm
```

Blobs reach a node only through the registry, and only the ones it lacks. Requests reach the engine only through the gateway, after the scheduler has picked a node.

| Component | Runs on | Language | Owns |
| --- | --- | --- | --- |
| `wpk` CLI | Laptop, CI | Go | Parses commands; talks to the local `wpkd` |
| `wpkd` daemon | Build hosts and GPU nodes | Go | Local store, build cache, push and pull, signature checks |
| Build workers | Build hosts with GPUs | Python | Training (PEFT, TRL), delta import, codecs, eval runs |
| Weightpack plugin | Inside each vLLM process | Python, Triton, CUDA | Composition loading, expert pool, routers, kernels |
| Control plane | Kubernetes, as an operator | Go | Gateway, scheduler, catalogue, metrics |
| OCI registry | Existing: Harbor, GHCR, ECR | n/a | Manifests, blobs, signatures |

**Design decisions**

- **One daemon per machine, as with dockerd.** The CLI stays thin. `wpkd` owns the store, so concurrent builds and pulls share one CAS and one cache. On a GPU node the same daemon is the node agent.
- **Go around the tensors, Python on them.** OCI, Kubernetes and Sigstore libraries are Go-first; PEFT, vLLM and eval harnesses are Python. The two talk over a local gRPC socket (`proto/weightpack/worker/v1`). Per-weight work (squash, codecs, hashing large tensors) stays in Python on native libraries, never in the daemon. The daemon hands the worker the paths of the blobs it needs, which the worker opens read-only, and a staging directory for its output; it then checks each output file (one tensor, the expected name, dtype and shape) and stores it in canonical form itself, so only the daemon ever writes to the store. Splitting a safetensors file into tensor blobs is not per-weight work: the daemon copies the byte ranges the header names and never decodes a value, and the store hashes blobs as it writes them, as any OCI store does.
- **Store layout.** Blobs sit under `blobs/sha256/<ab>/<digest>`, compressed as pushed. Beside them is a decoded cache of memory-mappable safetensors, and SQLite holds images, chain IDs, reference counts and fidelity records. The decoded cache is disposable. Until reference counts and fidelity records arrive, tags live in an `index.json` in the OCI image-layout format, one manifest descriptor per `name:tag`.
- **Base affinity is the scheduling rule.** The scheduler groups images by base digest and sends each request to a replica that already holds that base. Cluster-level memory savings come from this rule, not from the format. A new copy of a base loads only when no replica holding it has room.
- **The engine stays stock.** The plugin uses vLLM's plugin and LoRA interfaces plus one custom module for mixes, so a vLLM upgrade does not mean maintaining a fork.
- **Nodes are disposable.** Everything a node needs is addressable by digest, so a node can be drained, rebuilt and refilled from the registry.
- **Trust boundary at pull.** `wpkd` verifies signatures before decoding. The plugin loads only safetensors from the decoded cache. Training jobs run sandboxed, without registry credentials.

## Image format

An image is a standard OCI manifest. Its blobs are tensors and its config declares the layer graph, so any OCI registry (Docker Hub, GHCR, Harbor, ECR) stores it unchanged.

- **Manifest `layers[]`:** one entry per tensor blob, annotated with its layer id and tensor name. Small tensors (norms, biases) are packed into one blob per layer. Identical tensors across images get the same digest, so they are stored once.
- **Config blob:** the architecture fingerprint, the tokenizer digest, the layer graph and the fidelity records.

```json
{
  "schema": "weightpack.config.v1",
  "arch": { "family": "llama", "fingerprint": "sha256:9f1c…", "blocks": 32, "hidden": 4096 },
  "tokenizer": "sha256:44ab…",
  "layers": [
    { "id": "base",    "kind": "base", "digest": "sha256:d203…", "chain": "sha256:d203…" },
    { "id": "french",  "kind": "lora", "parent": "base", "digest": "sha256:7e01…", "chain": "sha256:cbb5…",
      "rank": 32, "alpha": 64, "targets": ["q", "k", "v", "o", "gate", "up", "down"] },
    { "id": "domain",  "kind": "mix", "parent": "french", "blocks": "8-31",
      "targets": ["gate", "up", "down"], "top_k": 2, "router": "arrow",
      "experts": ["legal", "medical", "finance"] },
    { "id": "legal",   "kind": "lora", "parent": "french", "rank": 16 },
    { "id": "medical", "kind": "lora", "parent": "french", "rank": 16 },
    { "id": "finance", "kind": "delta.bit1", "parent": "french",
      "fidelity": { "source": "hf://acme/llama-fin@3e2a…", "eval": "finance-v2", "max_drop": 0.5 } }
  ]
}
```

A build sets `digest` and `chain` on every layer (the other layers are abbreviated above). A LoRA layer's `alpha` and `rslora` give its scaling as PEFT defines it: alpha / rank, or alpha / √rank with `rslora: true`.

Four rules make composition safe:

1. **Chain IDs.** A layer's identity is `sha256(parent chain ID + " " + layer digest)`, as Docker computes it; the base's chain ID is its own digest. A layer is valid only on the exact chain it was built on. The **layer digest** is sha256 over the compact JSON of the layer's kind, the parameters that change how it applies (`rank`, `alpha`, `rslora`, `targets`, `blocks`, `top_k`, `router`), the layer digests of its experts in mix order (for a mix), and its blobs (tensor, title, media type, digest), sorted. The layer id is left out, so the same content under another name is the same layer, and so is the fidelity record, which says how a layer was made rather than what it is. Any reader can recompute every digest and chain ID from the manifest and config, which is how a pulled image is checked.
2. **Architecture fingerprint.** A hash over the base's sorted (tensor name, shape, dtype) list plus the tokenizer digest: the digest of the `tokenizer.json` blob, or of `tokenizer.model` when there is no `tokenizer.json`. It plays the role of `linux/amd64`: pull and build refuse a mismatch.
3. **Declared targets.** Every delta lists the tensors it touches. Untouched tensors cost nothing.
4. **One parent per mix.** All experts in a `MIX` must have the mix's parent chain.

Layer ids are lowercase letters, digits, `.`, `_` and `-`, starting with a letter or digit, at most 64 characters: the same form a Weightfile accepts. Configs with other ids are refused wherever an image is checked, at pull included, so an image from a registry cannot slip path syntax into an id that ends up naming a file or directory.

| Media type (`application/vnd.weightpack.…`) | Holds | Lossless |
| --- | --- | --- |
| `tensor.v1` | One base tensor, raw safetensors | Yes |
| `lora.v1` | A and B matrices for one target module | Yes |
| `delta.xor.v1` | Bitwise XOR against the parent tensor | Yes |
| `delta.bit1.v1` | Sign bits plus one scale per matrix | No |
| `delta.sparse.v1` | Kept indices and rescaled values | No |
| `router.v1` | Router weights or Arrow prototypes, per block | Yes |
| `meta.v1` | Tokenizer, chat template, license, model card | Yes |

An entropy coder is a suffix, as OCI does with `+zstd`: `lora.v1+zipnn`. Docker keeps model layers uncompressed so they can be memory-mapped; Weightpack trades that for size and decodes once into a local cache on pull. The daemon does the entropy coding, which is byte-level like hashing: it compresses what it stores and decompresses what it hands a worker, so workers only ever see plain safetensors. Compression runs with one encoder goroutine at a fixed level, so the same tensor gives the same blob digest with a given encoder version.

**Blob layout.** Tensor blobs are safetensors files in one canonical byte layout: tensors ordered by name, data packed with no gaps, a compact JSON header with sorted keys and no `__metadata__`, padded with spaces to a multiple of 8 bytes. The same tensors under the same names therefore give the same digest whatever file, shard or writer they came from, which is what makes tensor-level dedup work.

- `tensor.v1`: one base tensor, annotated `wpk.tensor` with its name. Tensors under 64 KiB (norms, biases) share one packed blob per layer, which has no `wpk.tensor` annotation; their names are in its header.
- `lora.v1`: the `lora_A` and `lora_B` matrices of one module, named in the base's naming (`model.layers.0.self_attn.q_proj.lora_A.weight`, without PEFT's `base_model.model.` prefix), annotated `wpk.tensor` with the base weight they modify. A LoRA layer may also hold `tensor.v1` blobs: `modules_to_save` copies, each a full tensor that replaces the base tensor of the same name (annotated `wpk.tensor` with it) at that layer's place in the chain, before the layer's LoRA deltas.
- `meta.v1`: one file per blob, named by the standard `org.opencontainers.image.title` annotation. The base's `config.json`, tokenizer files, license and model card belong to the base layer. A `TEMPLATE` file and a `LICENSE` file are image-level: they carry no `wpk.layer` annotation. A `LICENSE` given as an SPDX id becomes the manifest's `org.opencontainers.image.licenses` annotation instead.

Nothing in a manifest depends on paths, file order or time, so building the same inputs twice gives the same manifest digest.

## Weightfile and CLI

The Weightfile is the Dockerfile: one line per layer, built top to bottom, each step cached by the digest of its inputs.

```dockerfile
FROM registry.example.com/llama-3.1-8b-instruct@sha256:b45e…

# Stack: applied on top of FROM; trained on FROM
LAYER french ./adapters/french

# Mix: experts side by side; each trained on FROM + french
MIX domain targets=mlp blocks=8-31 top_k=2
  EXPERT legal    ./adapters/legal
  EXPERT medical  oci://registry.example.com/experts/medical:1.4
  EXPERT finance  hf://acme/llama-fin-full  codec=bit1 eval=./evals/finance.yaml max_drop=0.5
  ROUTER arrow            # zero-shot; or: ROUTER train data=./mix.jsonl steps=2000

TEMPLATE ./chat.jinja
LICENSE inherit
```

- `LAYER` and `EXPERT` accept a local adapter, an OCI reference, a Hugging Face repo, or a `TRAIN` block that calls a trainer.
- `FROM` an image inherits its base and stack as they are. `LAYER` or `EXPERT` from an image takes its top layer, and the build refuses it unless the layer's parent chain ID equals the chain it would sit on: rule 1 below, checked at build time.
- `LABEL key=value` sets annotations on the image's manifest, as a Dockerfile's labels do.
- A full fine-tune given as an `EXPERT` (finance above) is imported: Weightpack subtracts the parent, encodes the delta with the named codec, runs the eval, and fails the build if the drop exceeds `max_drop`.
- The cache key of a step is the hash of its parent chain ID, data digest, trainer config and trainer version. Change the French data and only `french` and everything above it rebuild.

| Command | Docker analogue | What it does |
| --- | --- | --- |
| `wpk build -t name:tag .` | `docker build` | Resolves the Weightfile, imports and encodes layers, writes the image |
| `wpk push` / `wpk pull` | same | Moves only the blobs the other side lacks |
| `wpk run name:tag` | `docker run` | Starts a container: an OpenAI-compatible endpoint for that composition |
| `wpk ps` / `wpk stop` / `wpk logs` | same | Lists, stops and shows the engine log of containers |
| `wpk squash name:tag --to dense` | `docker export` | Bakes a stack, or a mix with fixed weights, into one dense checkpoint |
| `wpk diff a b` | `docker diff` | Per-tensor report: which tensors differ and by how much |
| `wpk inspect` / `wpk history` | same | Layers, parents, sizes, codecs, fidelity |
| `wpk import hf://… --parent ref` | `docker import` | Turns an existing fine-tune into a delta layer over a parent |
| `wpk export --format hf\|gguf` | `docker save` | Plain checkpoint for other tools |

**Export.** `wpk export -o DIR <ref>` writes an image out for other tools and, with its Weightfile, back into Weightpack. `DIR/base/` is a Hugging Face model directory: the base tensors in safetensors shards of at most `--max-shard-size` (5 GB by default, with `model.safetensors.index.json` when there are several), plus the base's `config.json`, tokenizer and other files. `DIR/adapters/<id>/` holds each LoRA layer as a PEFT adapter, with PEFT's tensor names and an `adapter_config.json`: `target_modules` is the module-suffix list when the adapter covers every matching base matrix, and otherwise a regular expression naming its modules exactly. Adapter configs leave `task_type` null, since the image does not record an adapter's task, so PEFT loads them with its generic wrapper on any model. The image's chat template and license file go to the export root, its manifest annotations (a squash's provenance among them) become `LABEL` lines, and `DIR/Weightfile` rebuilds the image: `wpk build DIR` gives the same manifest digest, squashed images included. A squashed image exports as a plain `DIR/base` any engine loads. Only `--format hf` exists so far.

## Storage and compression

| Scheme | Lossless | Per variant | Base + 50 variants | vs full copies |
| --- | --- | --- | --- | --- |
| 51 full copies | Yes | 16.1 GB | 819 GB | 1× |
| 51 full copies, entropy-coded ([ZipNN](https://arxiv.org/abs/2411.05239), 66.4% of size) | Yes | 10.7 GB | 544 GB | 1.5× |
| Base + 50 full fine-tunes as 1-bit deltas ([BitDelta](https://arxiv.org/abs/2402.10193)-style) | No | 1.0 GB | 66 GB | 12× |
| Base + 50 LoRA layers, rank 16 | Yes | 84 MB | 20 GB | 40× |

Chunk-level dedup barely helps the first row: a full fine-tune nudges almost every weight, so almost every chunk changes. [ZipLLM](https://arxiv.org/abs/2505.06252) reports a 54% lossless cut on Hugging Face models by combining tensor dedup with XOR deltas; expect lossless storage of full fine-tunes to land near there, not near 12×. GPU memory at serve time follows the same ratios, because the runtime keeps deltas as deltas.

Every tensor goes through the same ingest pipeline:

1. **Exact dedup.** Hash the tensor. If the digest exists, reference it. This catches frozen embeddings, untouched norms and re-uploads.
2. **Delta.** Compare with the same tensor in the parent chain: arithmetic difference for lossy codecs, bitwise XOR for lossless.
3. **Codec.** `xor` (lossless), `bit1` (sign bits plus a calibrated scale per matrix), `sparse` (drop and rescale), or `lora` (already low rank). There is no default low-rank codec for full fine-tunes: their deltas are high rank, so an SVD cut loses too much.
4. **Entropy coding.** Split float bytes into exponent and mantissa streams and code them separately. Exponents are highly skewed, so they compress well.
5. **Fidelity record.** Store the relative error per tensor and the eval score against the source model.

**Lossless import.** `codec=xor` (or `wpk import --parent <ref> <dir>`) stores a full fine-tune as a `delta.xor` layer: for each tensor the fine-tune changed, its bits XORed with the parent's, in a `delta.xor.v1+zstd` blob (one canonical single-tensor safetensors file, zstd-compressed). Tensors the fine-tune left alone are not stored. The worker computes the deltas and the daemon checks, canonicalises and compresses them. An XOR layer sits only on the base or on other XOR layers: its parent's bits are then exact, while over a LoRA layer they would come from float math whose rounding can differ between machines, and decoding would no longer be bit-exact. LoRA layers may sit on XOR layers. Squash applies XOR layers to the base's bits first, exactly, then any LoRA math. A lossless layer has no fidelity record; only lossy codecs (`bit1`, `sparse`) carry one.

**Fidelity contract.** A lossy layer carries its codec, parameters, eval suite and measured drop. Each domain has a shared default eval suite; a layer's author can add their own, and the build gates on both. `wpk pull --exact` refuses lossy layers. `wpk build` fails when the drop exceeds `max_drop`. The source digest stays in the record, so a lossless rebuild is possible while the original is still reachable.

**Garbage collection.** Blobs are reference-counted by manifests. `wpk rm` untags images and `wpk prune` removes every blob no tag reaches, as `docker image prune` does, along with stale build-cache entries and old temporary files (`--downloads` also clears the `hf://` cache, `--dry-run` only counts). The daemon holds builds, pulls, pushes, squashes, exports and tag changes off while it prunes, so nothing in use is deleted.

**Build cache.** A rebuild of unchanged sources reads nothing. The builder remembers which blob each set of source tensors (or each meta file, or each XOR import over a given parent chain) became, keyed by file identity: device, inode, size, and modification and change times, as git's index does. The change time moves on any write and cannot be set back, so an edited file never looks unchanged, and an entry only counts while its blob is still in the store. `wpk build --no-cache` reads everything again.

## Runtime

The runtime keeps the base in GPU memory once and applies each expert as a small delta beside it, so a mix of three experts costs little more memory than the base alone.

- **Shared base.** One copy per GPU, or per tensor-parallel group. Every container on that GPU references it, the way containers share image layers through the page cache.
- **Expert pool.** Deltas live in host RAM and page into a GPU pool on demand, with LRU eviction. The pool shares memory with the KV cache, as in [S-LoRA](https://arxiv.org/abs/2311.03285)'s unified paging.
- **Forward pass.** For each mixed linear module, the base product is computed once per token, whatever the number of experts:

$$y = W x + \sum_{i \in \mathrm{TopK}(g(x))} g_i(x)\, B_i A_i x$$

For a 1-bit expert the term becomes sᵢ · sign(Δᵢ) x. Expert terms run in gather-style batched kernels, so one batch can hold tokens routed to different experts and requests for different images.

| Routing mode | Who picks the expert | Router weights | Cost | Use when |
| --- | --- | --- | --- | --- |
| Request | The caller (a header) or a small classifier | None | Lowest: same as multi-LoRA serving today | One domain per request |
| Sequence | A router on the prompt's hidden state, once per sequence | Small | Low: experts stay fixed for the whole generation | Mixed traffic, one domain per conversation |
| Token | A router in each mixed block, per token (Arrow or a trained gate) | Per block | Highest: more kernel work and memory traffic per token | Prompts that mix domains inside one request |

- **Materialize.** A pure stack, or a mix with fixed weights, can be baked with `wpk squash --to dense` and served by any engine with zero overhead.
- **Engine choice.** Ship first as a vLLM plugin, which already batches many LoRAs; add one custom module for token-level mixing. Writing a new engine is not worth it.
- **Cache keys.** Prefix-cache keys must include the image digest: the same prompt gives different KV entries under different images.

**Serving today.** `wpk run` serves an image through stock engines, with per-request routing:

- **Serving base.** Engines apply one adapter per request, so everything below an image's adapters is baked into a dense serving base: its XOR layers and every stacked LoRA layer but the top one (`squash` through that layer). A stack ending in a LoRA layer serves that layer as the adapter; a MIX serves each expert as an adapter, as model `<name>/<expert>`. The base is named by the chain ID of its top layer, so every image with the same lower layers has the same one. It is exported once under `<root>/run/bases/<chain>` (adapters under `<root>/run/adapters/<chain>-<id>`); a squash it needs leaves an untagged image that `wpk prune` removes, along with serving files no container uses.
- **One engine per base and device.** The first container on a base starts an engine process on a loopback port; later containers on that base add their adapters to it at runtime (`/v1/load_lora_adapter`), so the base's weights load once. The engine stops with its last container, and with `wpkd`. Engines are vLLM on NVIDIA GPUs and `weightpack-engine` (transformers and PEFT; `cuda`, `mps` or `cpu`) elsewhere; both speak the same OpenAI API.
- **Gateway.** `wpkd` listens for the OpenAI API on `127.0.0.1:8090` and routes each request by its `model` field to the engine and the engine's name for that model, renaming the model back in replies and streams. It sends the image's `TEMPLATE` as the request's `chat_template` unless the request brings one, since images that share a base may differ in template. The gateway does no tensor work.

**FFN experts (v1).** An `ffn-expert` replaces the whole FFN path, so it cannot ride on the shared base product the way a LoRA does: the block outputs the router-weighted sum of each chosen expert's FFN, each with weights W₀ + Δᵢ. The gate and up projections still share W₀x across experts, because they see the same input. Only the down projection needs a full product per active expert. Weights stay stored and paged as deltas, so the extra cost is compute, not memory. When a full fine-tune is imported as an FFN expert, its attention changes are dropped by default; `attn=average` instead averages them into one stacked layer below the mix, as BTX does.

## Hard problems

Two of these shape the design: a layer only works on the exact chain it was trained on, and independently trained layers should be mixed, not stacked.

1. **Layers are not portable.** A layer trained on base + A is valid only on base + A. Moving it to a new base version (Llama 3.1 to 3.3) means retraining it. Weightpack exposes this as `wpk rebase`: marked lossy, eval-gated, never silent. The method is under Decisions.
2. **Stacking independent layers interferes.** Summing deltas trained separately degrades each of them. So `LAYER` is for layers trained in sequence, and independent layers go in a `MIX`. A weight merge (TIES, DARE) stays available through `wpk squash`, as a lossy, eval-gated step.
3. **Routers are not free.** Zero-shot routing (Arrow, mergekit's hidden-state gates) works without joint training. BTX still fine-tunes its router on mixed data, and that usually wins. Budget a cheap `ROUTER train` step: experts frozen, a few thousand mixed examples.
4. **Full fine-tunes only compress well lossily.** The 10× and higher wins need lossy codecs; lossless delta coding gets about 2×. Teams should train adapters when a variant is meant to live in Weightpack.
5. **Token routing costs memory traffic.** Each extra active expert adds reads per token. Cap `top_k` at 2 and mix only the blocks that need it.
6. **Experts of different shapes.** Experts can use different ranks and targets; an expert with no delta on a module adds zero there. Arrow needs a prototype per module, so it routes only among experts that touch that module.
7. **Trust and licenses.** A delta can carry a backdoor as easily as a skill. Sign every layer (Sigstore), load safetensors only, and record data sources and license per layer. The base license applies to every layer above it.

## Roadmap and decisions

Build it in four phases, each closed by a measurable gate. Phases 1 and 2 pay for themselves even if token-level mixing never ships: they already give dedup, one-command composition and multi-variant serving.

| Phase | Scope | Gate |
| --- | --- | --- |
| 1 · Format, store | Tensor store on OCI; base and LoRA layers; `FROM`, `LAYER`, `EXPERT`; build, push, squash; vLLM with per-request routing | 50 LoRA variants of one 8B base in 20 GB; squash matches a PEFT merge |
| 2 · Delta import | `wpk import`; `xor` codec (lossless); `bit1` codec (lossy); `ffn-expert` import; fidelity records and eval gate | Lossless import is bit-exact; `bit1` import stays within `max_drop` on its eval |
| 3 · Token mixing | Arrow router; vLLM module for MoE over LoRA and FFN experts; `ROUTER train`; sequence and token routing | A 3-expert mix beats each single expert on a mixed-domain eval |
| 4 · Ecosystem | ModelPack media types proposal; HF and GGUF export; Sigstore signing; `wpk rebase` | A ModelPack client pulls and runs a squashed image |

Phase 3 is the research-heavy one, so it comes after the format and codecs are stable. Phase 4 can start any time someone needs interop.

**Decisions**

- **Packaging: ModelPack-compatible now, own namespace for deltas.** Base tensors and squashed images use ModelPack's manifest, artifact type and config, so any ModelPack client can pull them. Delta, router and mix types live under `vnd.weightpack.*` and go upstream as a proposal after phase 2, once they stop changing. ModelPack is still a CNCF Sandbox project; waiting on its review cycle would slow phases 1 and 2.
- **Languages: Go for `wpk`, `wpkd` and the control plane; Python for everything that touches tensors.**
- **Engine: vLLM first.** It already batches many LoRAs per request; Weightpack adds one module for token-level mixing.
- **`ffn-expert` is in v1.** Import lands in phase 2 and serving in phase 3. Its runtime cost is described under Runtime.
- **Rebase: transfer, gate, then distil.** `wpk rebase` tries three steps in order and stops at the first that passes the eval gate. First, add the old delta to the new base unchanged: [fine-tuning transfer](https://arxiv.org/abs/2503.20110) works best when the two bases are linearly connected, so it suits releases within one model family. Second, distil: the old image teaches a fresh layer on the new base over filtered synthetic prompts, as [Trans-LoRA](https://arxiv.org/abs/2405.17258) does, so no original data is needed. Third, when the original data exists, fine-tune briefly starting from the transferred delta. Steps 1 and 2 run unattended; step 3 is opt-in.
- **Fidelity evals: a shared default suite per domain, author suites optional.** A layer's author can add a suite; the build gates on both.

## Sources

- [Ollama Modelfile reference](https://docs.ollama.com/modelfile) and [Ollama manifest format](https://pkg.go.dev/github.com/ollama/ollama@v0.12.0/server/internal/manifest)
- [CNCF ModelPack specification](https://github.com/modelpack/model-spec); [Docker and ModelPack interop](https://www.cncf.io/blog/2026/08/12/advancing-ai-model-interoperability-with-docker-and-modelpack/); [docker model package](https://docs.docker.com/reference/cli/docker/model/package/); [Why Docker chose OCI artifacts for models](https://www.docker.com/blog/oci-artifacts-for-ai-model-packaging/)
- [Hugging Face Xet storage](https://huggingface.co/docs/hub/xet/index.md)
- [ZipLLM (NSDI '26)](https://arxiv.org/abs/2505.06252); [ZipNN](https://arxiv.org/abs/2411.05239); [BitDelta](https://arxiv.org/abs/2402.10193)
- [S-LoRA](https://arxiv.org/abs/2311.03285)
- [Branch-Train-MiX](https://arxiv.org/abs/2403.07816); [MergeKit MoE guide](https://huggingface.co/blog/mlabonne/frankenmoe); [MixtureKit](https://arxiv.org/abs/2512.12121)
- [Arrow routing (ICML 2024)](https://arxiv.org/abs/2405.11157)
- [Efficient Model Development through Fine-tuning Transfer](https://arxiv.org/abs/2503.20110); [Trans-LoRA](https://arxiv.org/abs/2405.17258)
