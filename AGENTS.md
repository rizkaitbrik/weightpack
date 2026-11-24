# Notes for coding agents

Weightpack is an Nx workspace with Go and Python projects. Read `docs/design.md` before changing the image format, the store or the runtime.

- Run tasks through Nx with the workspace's package manager: `npx nx test store`, `npx nx affected -t lint test build`.
- Go: one module at the root (`github.com/rizkaitbrik/weightpack`). Projects are declared in each `project.json` with `@nx-go/nx-go` executors. Keep `gofmt` clean.
- Python: each project under `python/` has its own `pyproject.toml` and `uv.lock`. Use the Nx targets (`sync`, `lint`, `format`, `test`, `build`) from `@nxlv/python`.
- The Go daemon never does per-weight work; tensor math belongs in the Python workers.
- A change to media types, the config schema or chain-ID rules (`libs/go/spec`) must update `docs/design.md` in the same change.
- Check generator and executor options with `--help` instead of guessing flags.
