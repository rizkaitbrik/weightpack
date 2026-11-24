# Weightpack

**Docker for neural nets.** Ship a model as a base plus small, content-addressed weight layers. Stack layers, or mix them side by side as experts behind a router, and push the result to any OCI registry.

> **Status: early development.** `wpk build` turns a Weightfile with a local safetensors base and PEFT LoRA adapters into an image in the local store; `images`, `inspect` and `history` read it back; `push` and `pull` move it through any OCI registry, sending only the blobs the other side lacks; `squash` bakes LoRA layers into dense weights that match PEFT's merge bit for bit; `export` writes an image out as Hugging Face and PEFT files that transformers, PEFT and vLLM load. Serving does not exist yet. See the [design doc](docs/design.md) and the roadmap below.

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
| `apps/wpkd` | `wpkd` | Go | Daemon: local store, build cache, push/pull; node agent on GPU hosts |
| `libs/go/spec` | `spec` | Go | Image format: media types, config schema, layer digests, chain IDs, architecture fingerprint |
| `libs/go/store` | `store` | Go | Content-addressed blob store and tags |
| `libs/go/weightfile` | `weightfile` | Go | Weightfile parser ([reference](docs/weightfile.md)) |
| `libs/go/build` | `build` | Go | Builder: Weightfile to image (base split into tensor blobs, LoRA layers checked against the base) |
| `libs/go/safetensors` | `safetensors` | Go | safetensors header reader and canonical writer; never decodes values |
| `libs/go/reference` | `reference` | Go | OCI image reference parsing |
| `libs/go/registry` | `registry` | Go | Push and pull to OCI registries; pulls are checked before any tensor is fetched |
| `libs/go/api` | `api` | Go | `wpk` to `wpkd` wire types and client |
| `libs/go/daemon` | `daemon` | Go | `wpkd`'s HTTP API over the store |
| `libs/go/export` | `export` | Go | Export: Hugging Face base, PEFT adapters and a Weightfile that rebuilds the image |
| `libs/go/squash` | `squash` | Go | Squash: picks the layers, hands blobs to the worker, stores the dense image |
| `libs/go/worker`, `libs/go/workerpb` | `worker-client` | Go | gRPC client for the build worker; generated contract code |
| `python/worker` | `worker` | Python | Build worker: training, delta import, codecs, evals |
| `python/vllm-plugin` | `vllm-plugin` | Python | vLLM plugin: composition loading, expert pool, routers |
| `proto/` | `proto` | Protobuf | gRPC contract between `wpkd` and the workers (`npx nx run proto:generate`) |
| `docs/` | | | Design doc, Weightfile reference |

## Getting started

Requirements: Node.js 22+, Go 1.24+, Python 3.10+ and [uv](https://docs.astral.sh/uv/).

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
./dist/apps/wpk export -o ./assistant-legal assistant-legal               # ./assistant-legal/base loads in transformers or vLLM
./dist/apps/wpk tag assistant:v1 registry.example.com/team/assistant:v1
./dist/apps/wpk push registry.example.com/team/assistant:v1
./dist/apps/wpk pull --exact registry.example.com/team/assistant:v1
```

`push` and `pull` use the credentials `docker login` stores, and talk plain HTTP to registries on localhost (or anywhere with `--plain-http`). `pull` checks the manifest and config, digests and chain IDs included, before fetching any tensor; `--exact` refuses lossy layers.

`wpk` finds the daemon at `$WPK_SOCKET`, else `$WPK_ROOT/wpkd.sock`, else `~/.weightpack/wpkd.sock`.

## Roadmap

1. **Format and store**: tensor store on OCI, base and LoRA layers, `build` / `push` / `pull` / `squash`, per-request serving on vLLM.
2. **Delta import**: `wpk import`, lossless XOR and lossy 1-bit codecs, FFN experts, fidelity records and eval gates.
3. **Token-level mixing**: Arrow router, a vLLM module for mixtures of LoRA and FFN experts, router training.
4. **Ecosystem**: CNCF ModelPack compatibility, HF and GGUF export, Sigstore signing, `wpk rebase`.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), and open an issue before large changes so we can agree on the design first.

## License

[Apache License 2.0](LICENSE)
