# Weightpack

**Docker for neural nets.** Ship a model as a base plus small, content-addressed weight layers. Stack layers, or mix them side by side as experts behind a router, and push the result to any OCI registry.

> **Status: early design and scaffolding.** The image format types, the local blob store and reference codecs exist; `build`, `push`, `pull` and serving do not yet. See the [design doc](docs/design.md) and the roadmap below.

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
| `libs/go/spec` | `spec` | Go | Image format: media types, config schema, chain IDs, architecture fingerprint |
| `libs/go/store` | `store` | Go | Content-addressed blob store |
| `python/worker` | `worker` | Python | Build worker: training, delta import, codecs, evals |
| `python/vllm-plugin` | `vllm-plugin` | Python | vLLM plugin: composition loading, expert pool, routers |
| `proto/` | | Protobuf | gRPC contract between `wpkd` and the workers |
| `docs/` | | | Design doc |

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
npx nx build wpk
./dist/apps/wpk version
./dist/apps/wpk validate path/to/config.json
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
