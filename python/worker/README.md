# weightpack-worker

Build worker for [Weightpack](https://github.com/rizkaitbrik/weightpack): training jobs, delta import, codecs and eval runs. `wpkd` drives it over the gRPC contract in `proto/weightpack/worker/v1`.

Today it holds the reference codecs:

- `xor_encode` / `xor_decode`: lossless XOR delta against the parent tensor.
- `bit1_encode` / `bit1_decode`: lossy 1-bit delta (sign bits plus one scale per matrix).
