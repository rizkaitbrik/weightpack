"""Throughput of bit1 fine-tunes served per token on vLLM, beside the base.

    python bench_bit1.py WORK VLLM

It makes a model with random weights in Qwen2.5-7B's shapes (cost depends
on shapes only) with Qwen's tokenizer, and four random bit1 fine-tunes of it
as wpkd exports them. Then it serves the base with vLLM and weightpack-vllm
(delta slots, --max-loras 4), loads the fine-tunes, and measures output
tokens per second under concurrent load for the base alone, one fine-tune,
and all four in shared batches; and, for comparison, the base on plain vLLM
(the plugin off, LoRA on, as wpkd starts it). Needs an NVIDIA GPU with about
24 GB free, vLLM with this plugin, and network access for the tokenizer.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PORT = 18100
REQUESTS, CONCURRENCY, PROMPT, OUTPUT = 128, 32, 256, 128


def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        a = tensors[name]
        dtype = {np.uint8: "U8", np.float32: "F32"}[a.dtype.type]
        header[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "data_offsets": [offset, offset + a.nbytes],
        }
        offset += a.nbytes
        blobs.append(a.tobytes())
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for b in blobs:
            f.write(b)


def make_model(base: Path) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    if (base / "config.json").exists():
        return
    cfg = transformers.Qwen2Config(
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        vocab_size=152064,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = transformers.Qwen2ForCausalLM(cfg)
    model.save_pretrained(base, max_shard_size="5GB")
    del model
    torch.cuda.empty_cache()
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = hf_hub_download("Qwen/Qwen2.5-7B-Instruct", f)
        (base / f).write_bytes(Path(src).read_bytes())


def make_deltas(base: Path, work: Path, n: int) -> list[Path]:
    """n random bit1 fine-tunes of base, as wpkd exports them."""
    index = json.loads((base / "model.safetensors.index.json").read_text())
    shapes = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(base / shard, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            for name, info in json.loads(f.read(size)).items():
                if name != "__metadata__":
                    shapes[name] = info["shape"]
    rng = np.random.default_rng(0)
    out = []
    for k in range(n):
        d = work / f"ft{k}"
        out.append(d)
        if (d / "weightpack_delta.json").exists():
            continue
        d.mkdir(parents=True)
        files = []
        for i, (name, shape) in enumerate(sorted(shapes.items())):
            count = int(np.prod(shape))
            if len(shape) == 2:
                entries = {
                    name + ".signs": rng.integers(0, 256, (count + 7) // 8, np.uint8),
                    name + ".scale": np.array([2e-4], np.float32),
                }
            else:
                entries = {
                    name + ".delta": rng.standard_normal(count, np.float32) * 1e-3
                }
            files.append(f"{i:06d}.safetensors")
            write_safetensors(d / files[-1], entries)
        (d / "weightpack_delta.json").write_text(
            json.dumps({"version": 1, "codec": "bit1", "layer": d.name, "files": files})
        )
        (d / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 1,
                    "lora_alpha": 1,
                    "target_modules": [],
                    "bias": "none",
                    "use_rslora": False,
                    "fan_in_fan_out": False,
                    "base_model_name_or_path": "",
                }
            )
        )
    return out


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip().startswith(b"{") else {}


def serve(vllm: str, base: Path, log: Path, plugin: bool):
    env = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    # Neither run reuses the other's compiled graphs (see weightpack_vllm).
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    if plugin:
        env["WEIGHTPACK_DELTA_SLOTS"] = "1"
    else:
        env["WEIGHTPACK_DISABLE"] = "1"
    args = [
        vllm,
        "serve",
        str(base),
        "--port",
        str(PORT),
        "--served-model-name",
        "base",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--max-loras",
        "4",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.9",
    ]
    out = open(log, "w")  # noqa: SIM115 (vLLM writes to it while it runs)
    proc = subprocess.Popen(args, env=env, stdout=out, stderr=subprocess.STDOUT)
    for _ in range(900):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited: see {log}") from None
            time.sleep(1)
    raise RuntimeError("vLLM did not start")


async def load(models: list[str]) -> tuple[float, int]:
    import aiohttp

    rnd = random.Random(0)
    prompts = [
        [rnd.randrange(1000, 100000) for _ in range(PROMPT)] for _ in range(REQUESTS)
    ]
    sem = asyncio.Semaphore(CONCURRENCY)
    tokens = 0

    async def one(session, i):
        nonlocal tokens
        body = {
            "model": models[i % len(models)],
            "prompt": prompts[i],
            "max_tokens": OUTPUT,
            "temperature": 0,
            "ignore_eos": True,
        }
        async with (
            sem,
            session.post(f"http://127.0.0.1:{PORT}/v1/completions", json=body) as r,
        ):
            out = await r.json()
            if "usage" not in out:
                raise RuntimeError(out)
            tokens += out["usage"]["completion_tokens"]

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        await asyncio.gather(*(one(s, i) for i in range(min(8, REQUESTS))))  # warm up
        tokens = 0
        start = time.perf_counter()
        await asyncio.gather(*(one(s, i) for i in range(REQUESTS)))
    return time.perf_counter() - start, tokens


def main() -> None:
    work, vllm = Path(sys.argv[1]), sys.argv[2]
    base = work / "base"
    work.mkdir(parents=True, exist_ok=True)
    t = time.time()
    make_model(base)
    fts = make_deltas(base, work, 4)
    print(f"model and fine-tunes ready in {time.time() - t:.0f} s", flush=True)
    results = []
    for plugin in (False, True):
        log = work / f"vllm-{'plugin' if plugin else 'plain'}.log"
        proc = serve(vllm, base, log, plugin)
        try:
            runs = [("base", ["base"])]
            if plugin:
                for d in fts:
                    post(
                        f"http://127.0.0.1:{PORT}/v1/load_lora_adapter",
                        {"lora_name": d.name, "lora_path": str(d)},
                    )
                # Each fine-tune answers differently from the base: the deltas
                # are applied, not a graph compiled without them.
                replies = {}
                for m in ["base"] + [d.name for d in fts]:
                    body = {
                        "model": m,
                        "prompt": list(range(1000, 1032)),
                        "max_tokens": 16,
                        "temperature": 0,
                    }
                    out = post(f"http://127.0.0.1:{PORT}/v1/completions", body)
                    replies[m] = out["choices"][0]["text"]
                same = [
                    m for m in replies if m != "base" and replies[m] == replies["base"]
                ]
                print(f"fine-tunes answering as the base: {same or 'none'}", flush=True)
                runs += [
                    ("one fine-tune", ["ft0"]),
                    ("four fine-tunes", [d.name for d in fts]),
                    ("base and four fine-tunes", ["base"] + [d.name for d in fts]),
                ]
            for label, models in runs:
                secs, tokens = asyncio.run(load(models))
                results.append(
                    (("plugin " if plugin else "plain vLLM ") + label, tokens / secs)
                )
                print(
                    f"{results[-1][0]:40s} {results[-1][1]:8.1f} output tokens/s",
                    flush=True,
                )
        finally:
            proc.terminate()
            proc.wait(timeout=120)
        for line in log.read_text().splitlines():
            if (
                "weightpack:" in line
                or "KV cache" in line
                or "Model loading took" in line
            ):
                print("  " + line.split("]", 1)[-1].strip())
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
