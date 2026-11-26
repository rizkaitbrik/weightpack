# Quickstart

From a fresh machine to a mix of three fine-tunes served behind one OpenAI endpoint. It works on Apple Silicon Macs (Metal), Linux machines with NVIDIA GPUs (CUDA) and plain CPUs. The example model, Qwen2.5-0.5B-Instruct, is small enough for any of them.

## 1. Install

Install the latest release, on macOS (Apple Silicon) or Linux. It needs [uv](https://docs.astral.sh/uv/) for the Python parts:

```bash
curl -fsSL https://raw.githubusercontent.com/rizkaitbrik/weightpack/main/scripts/install.sh | sh
```

The script puts `wpk` and `wpkd` in `~/.local/bin` (`WPK_INSTALL_DIR` changes that), and installs `weightpack-worker[serve]`, the build worker and the reference engine, with `uv tool install`. It checks every download against the release's `checksums.txt`. `WPK_VERSION=0.2.0` pins a release, and `WPK_SERVICE=1` also runs step 2's `wpk service install`.

Or build from source, with Go 1.27+ and uv:

```bash
git clone https://github.com/rizkaitbrik/weightpack
cd weightpack
go build -o ~/.local/bin/ ./apps/wpk ./apps/wpkd     # or any directory on your PATH
uv tool install "./python/worker[serve]"             # weightpack-worker and weightpack-engine
```

Or run everything in a container (the store lives in the volume):

```bash
docker run -d --name wpkd -p 127.0.0.1:8090:8090 -v weightpack:/var/lib/weightpack ghcr.io/rizkaitbrik/weightpack:cpu
docker run -d --name wpkd --gpus all -p 127.0.0.1:8090:8090 -v weightpack:/var/lib/weightpack ghcr.io/rizkaitbrik/weightpack:cuda
docker exec wpkd wpk doctor     # then docker exec wpkd wpk … for every command below
```

`docker/Dockerfile` builds both images from a checkout. The CUDA image starts from vLLM's.

On a machine with an NVIDIA GPU, also install vLLM, which serves many adapters at once and batches requests, with the `weightpack-vllm` plugin, which lets it route each token of a mix across its experts:

```bash
uv tool install vllm --with "weightpack-vllm @ git+https://github.com/rizkaitbrik/weightpack#subdirectory=python/vllm-plugin"
```

## 2. Start the daemon

```bash
wpk service install    # a launchd agent on macOS, a systemd user unit on Linux
wpk doctor
```

`wpk service install` starts `wpkd` now and at every login, and records your `PATH` so it finds the worker and the engines. To run it in a terminal instead: `wpkd --worker weightpack-worker`.

`wpk doctor` checks everything `wpk` needs and says how to fix what fails:

```text
ok    wpkd              version dev, store /Users/you/.weightpack
ok    build worker      weightpack-worker 0.1.0 on /Users/you/.weightpack/worker.sock (codecs: lora, xor)
ok    device            mps (detected): Apple M3 GPU (Metal)
ok    reference engine  weightpack-engine 0.1.0: torch 2.15.0, transformers 5.24.1, peft 0.22.0; Metal available
-     vllm              serves on NVIDIA GPUs only; mps uses the reference engine
ok    gateway           http://127.0.0.1:8090/v1
ok    store             0 B of blobs; 134.7 GB free
warn  hugging face      no token: public repositories only
                        fix: for gated or private models (Llama, Gemma), run hf auth login or set HF_TOKEN, then restart wpkd
```

## 3. Build an image

A Weightfile names a base and the layers on top of it. This one mixes three public LoRA fine-tunes of the same base, pinned to commits:

```dockerfile
FROM hf://Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775
MIX tasks top_k=1
  EXPERT python hf://ali4566544/qwen2.5-0.5b-instruct-python-lora@5ecafffc2e23deebae61af586617143e596743a1
  EXPERT sql    hf://vindows/qwen2.5-0.5b-text-to-sql@427e5eda47e6f5752fab024e54da34ab08d27a13
  EXPERT math   hf://tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora@06abad5db457724a01606284f7a21e0b27f07c56
```

```bash
mkdir qwen-tasks && cd qwen-tasks    # save the Weightfile above here
wpk build -t qwen-tasks .
wpk history qwen-tasks
```

`wpkd` downloads only the files the build needs, checks every one against the Hub's hashes, and checks each adapter against the base: a layer built for another model is a build error, not a silent mismatch. The image stores the base once. The three adapters together add about 70 MB, where three full fine-tuned copies would take 3 GB.

## 4. Serve it

```bash
wpk run qwen-tasks
```

```text
Running qwen-tasks on the reference engine (mps)
  endpoint  http://127.0.0.1:8090/v1
  model     qwen-tasks  (routes each token across the experts: arrow)
  model     qwen-tasks/python
  model     qwen-tasks/sql
  model     qwen-tasks/math
```

`qwen-tasks` itself picks experts token by token, with the Arrow router: a prompt that mixes SQL and arithmetic can use both experts in one reply. Arrow needs no training, but it routes coarsely. To route by examples of your own traffic, add `ROUTER train data=./mix.jsonl steps=600` under the `EXPERT` lines and build again: the build trains the router on your data with the experts frozen (see [Trained routers](weightfile.md#trained-routers)). When you know the domain, name the expert. Each expert is its own model name, so a request picks one:

```bash
curl http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen-tasks/sql",
  "messages": [{"role": "user", "content": "Count the orders per customer."}]
}'
```

Any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen-tasks/math",
    messages=[{"role": "user", "content": "Tom has 3 apples and buys 5 more. How many now?"}],
)
print(reply.choices[0].message.content)
```

One engine process holds the base, and every expert is an adapter loaded into it. Another image on the same base (`wpk run` of a single fine-tune, say) joins the same engine instead of loading the base again. `wpk ps` lists what runs, and `wpk logs -f qwen-tasks` follows the engine's output.

## 5. Share it

Images are OCI artifacts, so any registry stores them, and a push or pull moves only the blobs the other side lacks:

```bash
wpk tag qwen-tasks registry.example.com/team/qwen-tasks:v1
wpk push registry.example.com/team/qwen-tasks:v1
# on another machine
wpk pull registry.example.com/team/qwen-tasks:v1 && wpk run registry.example.com/team/qwen-tasks:v1
```

Sign what you push, so machines that pull it can check where it came from. `wpk keygen` makes a key pair in cosign's format (the password comes from `$COSIGN_PASSWORD`, or is asked for):

```bash
wpk keygen -o release
wpk push --sign --key release.key registry.example.com/team/qwen-tasks:v1
```

On the machines that pull, require that signature for your registry. `wpk pull` (and `wpk build` of a `FROM` in that registry) then refuses any image without it, before downloading a tensor:

```bash
wpk trust set registry.example.com/team --key release.pub
wpk pull registry.example.com/team/qwen-tasks:v1    # Verified sha256:…: signed by key SHA256:…
```

In CI, sign keyless instead: `wpk push --sign` without `--key` gets the job's OIDC token (in GitHub Actions, grant `id-token: write`), and the policy trusts the workflow rather than a key: `wpk trust set registry.example.com/team --identity https://github.com/acme/models/.github/workflows/release.yml@refs/heads/main --issuer https://token.actions.githubusercontent.com`. Keyless signatures are logged in Sigstore's public Rekor, with the identity and the image digest. Signatures are cosign's, so `cosign verify` checks them too.

To hand one expert to a tool that knows nothing of Weightpack, bake it in and export plain files: `wpk squash -t sql-dense --expert sql qwen-tasks`, then `wpk export -o ./sql-dense sql-dense`. `./sql-dense/base` loads in transformers, vLLM or any other engine.

## 6. Clean up

```bash
wpk stop qwen-tasks       # the engine stops with its last container
wpk rm qwen-tasks
wpk prune                 # deletes blobs and serving files nothing uses
wpk service uninstall     # stops wpkd; the store stays in ~/.weightpack
```

## Hardware

| Machine | Device | Engine | Notes |
| --- | --- | --- | --- |
| Mac with Apple Silicon | `mps` | Reference engine (transformers + PEFT on Metal) | macOS 14 or later |
| Linux with an NVIDIA GPU | `cuda` | vLLM if installed, else the reference engine | `uv tool install vllm` with the `weightpack-vllm` plugin (step 1) for token routing; the `serve` extra brings CUDA torch |
| Anything else | `cpu` | Reference engine | Fine for small models and tests |

`wpkd` picks the device itself. `wpk run --device cpu` or `--engine reference` overrides it for one container, and `wpkd --device` sets the default. vLLM claims 90% of GPU memory, so to serve two different bases on one GPU, give each a share: `wpkd --engine-vllm "vllm serve --gpu-memory-utilization 0.45"`.
