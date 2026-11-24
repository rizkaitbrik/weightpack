#!/usr/bin/env bash
# Generates the Go and Python code for the wpkd <-> worker contract.
#
#   Go:     libs/go/workerpb
#   Python: python/worker/weightpack/worker/v1
#
# protoc comes from grpcio-tools in the worker's dev dependencies; the Go
# plugins are pinned here and installed into tmp/protoc-bin. Run it through
# Nx from the workspace root: npx nx run proto:generate
set -euo pipefail

PROTOC_GEN_GO=v1.36.12      # matches google.golang.org/protobuf in go.mod
PROTOC_GEN_GO_GRPC=v1.6.1   # supports google.golang.org/grpc in go.mod

root="$(cd "$(dirname "$0")/.." && pwd)"
bin="$root/tmp/protoc-bin"
mkdir -p "$bin" "$root/libs/go/workerpb"
GOBIN="$bin" go install "google.golang.org/protobuf/cmd/protoc-gen-go@$PROTOC_GEN_GO"
GOBIN="$bin" go install "google.golang.org/grpc/cmd/protoc-gen-go-grpc@$PROTOC_GEN_GO_GRPC"

cd "$root"
PATH="$bin:$PATH" uv run --project python/worker python -m grpc_tools.protoc \
  -I proto \
  --go_out=libs/go/workerpb --go_opt=paths=source_relative \
  --go-grpc_out=libs/go/workerpb --go-grpc_opt=paths=source_relative \
  --python_out=python/worker --pyi_out=python/worker --grpc_python_out=python/worker \
  weightpack/worker/v1/worker.proto

# protoc mirrors the proto path under the Go output; flatten it so the
# package lives at libs/go/workerpb as go_package says.
mv libs/go/workerpb/weightpack/worker/v1/*.go libs/go/workerpb/
rm -r libs/go/workerpb/weightpack
