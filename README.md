# Weightpack

**Docker for neural nets.** Ship a model as a base plus small, content-addressed weight layers. Stack layers, or mix them side by side as experts behind a router, and push the result to any OCI registry.

> **Status: early development.** `wpk build` turns a Weightfile into an image in the local store, from local safetensors bases and PEFT LoRA adapters, Hugging Face repositories (`hf://`) or images in a registry, refusing a layer built on a different chain; `images`, `inspect` and `history` read it back; `push` and `pull` move it through any OCI registry, sending only the blobs the other side lacks; `import` stores a full fine-tune losslessly as compressed XOR deltas over its parent; `squash` bakes LoRA and XOR layers into dense weights, matching PEFT's merge bit for bit; `export` writes an image out as Hugging Face and PEFT files that transformers, PEFT and vLLM load. It is checked end to end on Qwen2.5-0.5B-Instruct with public LoRA adapters (`npx nx run worker:smoke`). Serving does not exist yet. See the [design doc](docs/design.md) and the roadmap below.

## Why

Teams end up with dozens of fine-tuned variants of one base model, and each one is stored, shipped and loaded as a full copy. Weightpack stores each variant as only what it changed:

| | Base + 50 variants of an 8B model |
| --- | --- |
| 51 full copies | 819 GB |
| Base + 50 LoRA layers (rank 16) | 20 GB |

At serve time the base is loaded once per GPU, and every image built on it shares that copy.

## Concepts

- **Layer**: a set of weight changes over a parent (LoRA, XOR delta, 1-bit delta, FFN expert). Each layer is pinned to the exact chain it was built on.
- **Stack**: layers applied in order, as in Docker.
- **Mix**: layers side by side as experts; a router picks the top-k per request, sequence or token.
- **Squash**: bake a stack, or a fixed mix, into plain dense weights for any engine.

```dockerfile
FROM registry.example.com/llama-3.1-8b-instruct@sha256:b45e…

LAYER french ./adapters/french

MIX domain targets=mlp blocks=8-31 top_k=2
  EXPERT legal    ./adapters/legal
  EXPERT medical  oci://registry.example.com/experts/medical:1.4
  ROUTER arrow
```

## Repository layout

This is an [Nx](https://nx.dev) workspace with Go and Python projects.

| Path | Project | Language | What it is |
| --- | --- | --- | --- |
| `apps/wpk` | `wpk` | Go | Command-line client |
| `apps/wpkd` | `wpkd` | Go | Daemon: local store, build cache, push/pull, containers and the OpenAI gateway |
| `libs/go/spec` | `spec` | Go | Image format: media types, config schema, layer digests, chain IDs, architecture fingerprint |
| `libs/go/store` | `store` | Go | Content-addressed blob store and tags |
| `libs/go/weightfile` | `weightfile` | Go | Weightfile parser ([reference](docs/weightfile.md)) |
| `libs/go/build` | `build` | Go | Builder: Weightfile to image (base split into tensor blobs, LoRA layers checked against the base) |
| `libs/go/safetensors` | `safetensors` | Go | safetensors header reader and canonical writer; never decodes values |
| `libs/go/reference` | `reference` | Go | OCI image reference parsing |
| `libs/go/registry` | `registry` | Go | Push and pull to OCI registries; pulls are checked before any tensor is fetched |
| `libs/go/trust` | `trust` | Go | Sigstore signing and verification in cosign's format, and the trust policy pulls obey |
| `libs/go/api` | `api` | Go | `wpk` to `wpkd` wire types and client |
| `libs/go/daemon` | `daemon` | Go | `wpkd`'s HTTP API over the store |
| `libs/go/entropy` | `entropy` | Go | Entropy coders named by media-type suffixes (`+zstd`) |
| `libs/go/export` | `export` | Go | Export: Hugging Face base, PEFT adapters and a Weightfile that rebuilds the image |
| `libs/go/squash` | `squash` | Go | Squash: picks the layers, hands blobs to the worker, stores the dense image |
| `libs/go/serving` | `serving` | Go | Runtime: serving bases, engine processes, adapters, the gateway that routes by model, and doctor's device checks |
| `libs/go/service` | `service` | Go | `wpk service`: wpkd as a launchd agent or systemd user unit |
| `apps/wpk-control`, `libs/go/control` | `wpk-control`, `control` | Go | Cluster control plane: OpenAI gateway over the node agents, base-affinity placement, catalogue, metrics |
| `deploy/helm/weightpack` | | Helm | Kubernetes chart: wpkd as a node-agent DaemonSet, wpk-control behind a Service |
| `libs/go/worker`, `libs/go/workerpb` | `worker-client` | Go | gRPC client for the build worker; generated contract code |
| `python/worker` | `worker` | Python | Build worker (squash, delta import, codecs) and `weightpack-engine`, the reference serving engine |
| `python/vllm-plugin` | `vllm-plugin` | Python | vLLM plugin: composition loading, expert pool, routers |
| `proto/` | `proto` | Protobuf | gRPC contract between `wpkd` and the workers (`npx nx run proto:generate`) |
| `docs/` | | | Quickstart, design doc, Weightfile reference |
| `scripts/` | `scripts` | Shell | `install.sh`, tested against a fake release server |
| `docker/` | | Docker | CPU and CUDA images of wpkd with its worker and engines |
| `.goreleaser.yaml` | | | Release archives for macOS (Apple Silicon) and Linux, with checksums |

## Getting started

To use Weightpack, follow the [quickstart](docs/quickstart.md): install, check the machine with `wpk doctor`, build a mix of fine-tunes from the Hugging Face Hub and serve it behind one OpenAI endpoint. In short:

```bash
curl -fsSL https://raw.githubusercontent.com/rizkaitbrik/weightpack/main/scripts/install.sh | sh
wpk service install && wpk doctor
```

Container images (`ghcr.io/rizkaitbrik/weightpack:cpu` and `:cuda`, from `docker/Dockerfile`) bundle the daemon, the worker and an engine.

To work on it:

Requirements: Node.js 22+, Go 1.27+, Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rizkaitbrik/weightpack
cd weightpack
npm ci
npx nx run-many -t sync --projects=tag:lang:python   # create the Python venvs

npx nx run-many -t lint test build                     # everything
npx nx test store                                      # one project
npx nx graph                                           # project graph
```

Try the CLI:

```bash
npx nx run-many -t build --projects=wpk,wpkd
./dist/apps/wpkd --worker "uv run --project python/worker weightpack-worker" &   # store in ~/.weightpack, or set WPK_ROOT
./dist/apps/wpk validate path/to/Weightfile     # or a directory, or a config.json
./dist/apps/wpk build -t assistant:v1 path/to/project
./dist/apps/wpk images
./dist/apps/wpk history assistant:v1
./dist/apps/wpk squash -t assistant-legal --expert legal assistant:v1   # dense weights, one expert baked in
./dist/apps/wpk import --parent ./base -t finance ./finance-finetune         # a full fine-tune, stored as XOR deltas
./dist/apps/wpk rm assistant-legal && ./dist/apps/wpk prune                    # untag, then delete unreachable blobs
./dist/apps/wpk export -o ./assistant-legal assistant-legal               # ./assistant-legal/base loads in transformers or vLLM
./dist/apps/wpk tag assistant:v1 registry.example.com/team/assistant:v1
./dist/apps/wpk push registry.example.com/team/assistant:v1
./dist/apps/wpk pull --exact registry.example.com/team/assistant:v1
```

`push` and `pull` use the credentials `docker login` stores, and talk plain HTTP to registries on localhost (or anywhere with `--plain-http`). `pull` checks the manifest and config, digests and chain IDs included, before fetching any tensor; `--exact` refuses lossy layers.

### Sign and verify images

Signatures are Sigstore bundles stored next to the image, in the format cosign v3 uses, so `cosign verify` and `wpk verify` each check the other's. Sign with a key, or keyless with an OIDC identity (a browser login, or a CI job's token):

```bash
./dist/apps/wpk keygen -o release                                         # release.key (encrypted) and release.pub
./dist/apps/wpk push --sign --key release.key registry.example.com/team/assistant:v1
./dist/apps/wpk sign registry.example.com/team/assistant:v1               # keyless, logged in Sigstore's public Rekor
./dist/apps/wpk verify --key release.pub registry.example.com/team/assistant:v1
```

A trust policy makes `wpkd` refuse images that lack a trusted signature, on `pull` and on builds `FROM` a registry, before any tensor moves:

```bash
./dist/apps/wpk trust set registry.example.com/team --key release.pub
./dist/apps/wpk trust set ghcr.io/acme --identity-regexp 'https://github.com/acme/.*' --issuer https://token.actions.githubusercontent.com
./dist/apps/wpk trust set '*' --reject                                     # everything else
./dist/apps/wpk trust ls
```

`wpk` finds the daemon at `$WPK_SOCKET`, else `$WPK_ROOT/wpkd.sock`, else `~/.weightpack/wpkd.sock`.

### Serve images

`wpk run` serves an image on `wpkd`'s OpenAI-compatible gateway (`127.0.0.1:8090` by default). Images on the same base share one engine process and one copy of the base weights; each image's top LoRA layer, or each expert of a MIX, is an adapter loaded into that engine.

```bash
./dist/apps/wpkd --worker "uv run --project python/worker weightpack-worker" \
  --engine-reference "uv run --project python/worker --extra serve weightpack-engine" &
./dist/apps/wpk run assistant:v1            # model assistant:v1 routes each token across the experts (reference engine);
                                            # assistant:v1/<expert> picks one
./dist/apps/wpk run --name fr french:v1     # answers to model fr, on the same engine if the base is the same
./dist/apps/wpk ps
curl http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "fr", "messages": [{"role": "user", "content": "Bonjour"}]}'
./dist/apps/wpk logs -f fr
./dist/apps/wpk stop fr assistant:v1        # the engine stops with its last container
```

| Hardware | Default engine | Notes |
| --- | --- | --- |
| NVIDIA GPU (`cuda`) | vLLM, when `vllm` is installed; else the reference engine | vLLM loads adapters at runtime and batches them |
| Apple Silicon GPU (`mps`) | Reference engine (transformers + PEFT on Metal) | |
| CPU (`cpu`) | Reference engine | |

`wpkd` detects the device (`nvidia-smi`, then Apple Silicon, then CPU); `--device` and `--engine` override it per container, and `wpkd --device`, `--engine-reference` and `--engine-vllm` set the defaults and commands. vLLM claims 90% of GPU memory by default, so two different bases on one GPU need a smaller share each, such as `--engine-vllm "vllm serve --gpu-memory-utilization 0.45"`. If the gateway listens beyond loopback (`--gateway 0.0.0.0:8090`), set `--gateway-key` and send it as `Authorization: Bearer <key>`.

### Run on Kubernetes

The Helm chart runs `wpkd` on each GPU node and `wpk-control` in front of them: one OpenAI endpoint for the cluster, which keeps a catalogue of images running and places each on a node already serving its base, so a new fine-tune of a running base loads only its adapters. See [docs/kubernetes.md](docs/kubernetes.md).

```bash
helm install weightpack deploy/helm/weightpack -n weightpack --create-namespace \
  --set 'catalogue.models[0].name=qwen-tasks' --set 'catalogue.models[0].image=ghcr.io/acme/qwen-tasks:v1' \
  --set 'catalogue.models[0].replicas=2'
```

## Roadmap

1. **Format and store**: tensor store on OCI, base and LoRA layers, `build` / `push` / `pull` / `squash`, per-request serving on vLLM.
2. **Delta import**: `wpk import`, lossless XOR and lossy 1-bit codecs, FFN experts, fidelity records and eval gates.
3. **Token-level mixing**: Arrow router, a vLLM module for mixtures of LoRA and FFN experts, router training.
4. **Ecosystem**: CNCF ModelPack compatibility, HF and GGUF export, Sigstore signing, `wpk rebase`.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), and open an issue before large changes so we can agree on the design first.

## License

[Apache License 2.0](LICENSE)
