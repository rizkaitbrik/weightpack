# Contributing to Weightpack

Thanks for helping. This project is early, so design discussion is as valuable as code.

## Before you start

- For anything larger than a bug fix, open an issue first and describe the change. The [design doc](docs/design.md) is the reference; if your change departs from it, say so in the issue.
- Keep pull requests focused: one change per PR, with tests.

## Development setup

You need Node.js 22+, Go 1.24+, Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
npm ci
npx nx run-many -t sync --projects=tag:lang:python
```

Run tasks through Nx so caching and the project graph work:

```bash
npx nx affected -t lint test build   # what CI runs
npx nx test spec                     # a single project
npx nx format worker                 # format a Python project
npx nx run proto:generate            # regenerate Go and Python code after editing proto/
npx nx run worker:parity             # squash vs PEFT's merge (installs torch, transformers, peft)
npx nx run worker:smoke              # the whole flow on Qwen2.5-0.5B-Instruct and public LoRAs (downloads ~1 GB)
```

## Conventions

**Go** (`apps/`, `libs/go/`)

- One module at the repository root: `github.com/rizkaitbrik/weightpack`.
- Format with `gofmt`; `go vet` must pass. Tests run with `-race`.
- The daemon and CLI never do per-weight work. Anything that reads or transforms tensor values belongs in a Python worker.

**Python** (`python/`)

- Each project has its own `pyproject.toml` and `uv.lock`, managed by uv.
- Lint and format with Ruff (`nx lint <project>`, `nx format <project>`).
- Reference implementations (codecs, the mix forward pass) stay simple and readable; optimized kernels are tested against them.

**Format changes**

- Media types, the config schema and chain-ID rules live in `libs/go/spec`. A change there is a format change: update `docs/design.md` in the same PR.

## Commits and pull requests

- Write commit messages in the imperative ("Add XOR codec"), with a short body explaining why.
- CI must be green: lint, tests and builds for affected projects.
- By contributing you agree that your work is licensed under the [Apache License 2.0](LICENSE).
