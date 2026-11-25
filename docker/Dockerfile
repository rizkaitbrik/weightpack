# syntax=docker/dockerfile:1
#
# wpkd images: the daemon, its build worker and a serving engine in one
# container. Build from the repository root:
#
#   docker build -f docker/Dockerfile --target cpu  -t weightpack:cpu  .
#   docker build -f docker/Dockerfile --target cuda -t weightpack:cuda .
#
# Run (the store lives in the volume; the gateway answers on 8090):
#
#   docker run -d --name wpkd -p 127.0.0.1:8090:8090 -v weightpack:/var/lib/weightpack weightpack:cpu
#   docker run -d --name wpkd --gpus all -p 127.0.0.1:8090:8090 -v weightpack:/var/lib/weightpack weightpack:cuda
#   docker exec wpkd wpk doctor
#
# The gateway listens on all of the container's interfaces. Publish it on
# 127.0.0.1 as above, or set WPK_GATEWAY_KEY when it is reachable from
# other machines.

ARG GO_VERSION=1.24
ARG PYTHON_VERSION=3.12
# The CUDA image starts from vLLM's, which brings CUDA, torch and vLLM.
ARG VLLM_IMAGE=vllm/vllm-openai:latest

FROM golang:${GO_VERSION} AS binaries
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY apps apps
COPY libs libs
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/ ./apps/wpk ./apps/wpkd

# CPU: the reference engine on CPU-only torch.
FROM python:${PYTHON_VERSION}-slim AS cpu
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu
COPY python/worker /tmp/worker
RUN uv pip install --system --no-cache "/tmp/worker[serve]" && rm -rf /tmp/worker
COPY --from=binaries /out/wpk /out/wpkd /usr/local/bin/
ENV WPK_ROOT=/var/lib/weightpack HF_HOME=/var/lib/weightpack/hf
VOLUME /var/lib/weightpack
EXPOSE 8090
ENTRYPOINT ["wpkd", "--gateway", "0.0.0.0:8090", "--worker", "weightpack-worker", "--device", "cpu"]

# CUDA: vLLM by default, the reference engine as a fallback.
FROM ${VLLM_IMAGE} AS cuda
COPY python/worker /tmp/worker
# torch and transformers come with vLLM; this adds the worker and peft.
RUN python3 -m pip install --no-cache-dir "/tmp/worker[serve]" && rm -rf /tmp/worker
COPY --from=binaries /out/wpk /out/wpkd /usr/local/bin/
ENV WPK_ROOT=/var/lib/weightpack HF_HOME=/var/lib/weightpack/hf
VOLUME /var/lib/weightpack
EXPOSE 8090
ENTRYPOINT ["wpkd", "--gateway", "0.0.0.0:8090", "--worker", "weightpack-worker"]
