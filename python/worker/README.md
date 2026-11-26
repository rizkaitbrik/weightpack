# weightpack-worker

Build worker for [Weightpack](https://github.com/rizkaitbrik/weightpack): squash, delta import and router training, and later eval runs. `wpkd` drives it over the gRPC contract in `proto/weightpack/worker/v1`; the generated code is in `weightpack/worker/v1`.

```bash
weightpack-worker --socket ~/.weightpack/worker.sock   # or let wpkd start it: wpkd --worker weightpack-worker
```

What it does today:

- **Squash** (`weightpack_worker.squash`): bakes LoRA layers into the base tensors they modify, in float32, rounding once to the base dtype. It matches PEFT's `merge_and_unload` bit for bit; `npx nx run worker:parity` checks that against a real PEFT merge, and that what `wpk export` writes loads back into transformers and PEFT unchanged (it installs the `parity` dependency group: torch, transformers and peft).
- **Lossless delta import** (`weightpack_worker.delta`, the `EncodeLayer` RPC): a full fine-tune as XOR deltas over its parent, only for the tensors it changed. Squash decodes XOR layers exactly before any LoRA math, so an imported fine-tune comes back bit for bit.
- **Router training** (`weightpack_worker.router_train`, the `TrainRouter` RPC): `ROUTER train` trains a MIX's router with the base and experts frozen, starting from Arrow, on CUDA, Metal or the CPU, and streams its progress to `wpkd`. It needs the `serve` extra (torch, transformers, peft).
- **safetensors I/O on NumPy** (`weightpack_worker.safetensors_io`), BF16 included, with round-to-nearest-even conversion.
- **Reference codecs** (`weightpack_worker.codecs`): `xor_encode` / `xor_decode`, a lossless XOR delta against the parent tensor, and `bit1_encode` / `bit1_decode`, a lossy 1-bit delta (sign bits plus one scale per matrix).

`npx nx run worker:smoke` runs the whole flow on a real model: Qwen2.5-0.5B-Instruct and three public LoRA adapters from the Hub, built into a mix, pushed to a local registry, pulled into a second store, squashed and exported, then checked against transformers and PEFT (weights and greedy generations). It downloads about 1 GB into `tmp/smoke` and prints sizes and timings.

`npx nx run worker:gate` checks the token-mixing gate of the roadmap on the same model and experts: `wpk build` trains a router on public Python, SQL and math data (about 55 MB, also cached in `tmp/smoke`), and held-out examples of each domain are scored under the base, each expert, the mix with Arrow and the mix with the trained router. It passes when the trained mix beats every single expert.

`npx nx test worker` builds `wpk` and `wpkd` first, because the end-to-end tests drive the real binaries with this worker behind them.
