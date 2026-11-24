# weightpack-worker

Build worker for [Weightpack](https://github.com/rizkaitbrik/weightpack): squash, and later training jobs, delta import and eval runs. `wpkd` drives it over the gRPC contract in `proto/weightpack/worker/v1`; the generated code is in `weightpack/worker/v1`.

```bash
weightpack-worker --socket ~/.weightpack/worker.sock   # or let wpkd start it: wpkd --worker weightpack-worker
```

What it does today:

- **Squash** (`weightpack_worker.squash`): bakes LoRA layers into the base tensors they modify, in float32, rounding once to the base dtype. It matches PEFT's `merge_and_unload` bit for bit; `npx nx run worker:parity` checks that against a real PEFT merge (it installs the `parity` dependency group: torch, transformers and peft).
- **safetensors I/O on NumPy** (`weightpack_worker.safetensors_io`), BF16 included, with round-to-nearest-even conversion.
- **Reference codecs** (`weightpack_worker.codecs`): `xor_encode` / `xor_decode`, a lossless XOR delta against the parent tensor, and `bit1_encode` / `bit1_decode`, a lossy 1-bit delta (sign bits plus one scale per matrix).

`npx nx test worker` builds `wpk` and `wpkd` first, because the end-to-end tests drive the real binaries with this worker behind them.
