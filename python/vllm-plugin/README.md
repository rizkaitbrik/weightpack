# weightpack-vllm

[Weightpack](https://github.com/rizkaitbrik/weightpack) plugin for vLLM: it routes each token of a MIX across the MIX's LoRA experts, as `weightpack-engine` does, and serves full fine-tunes stored as bit1 layers over their base, per token, so they share one copy of its weights. vLLM loads it through the `vllm.general_plugins` entry point.

Install it in vLLM's environment (vLLM is not a declared dependency, so CI stays light):

```bash
uv tool install vllm --with "weightpack-vllm @ git+https://github.com/rizkaitbrik/weightpack#subdirectory=python/vllm-plugin"
# or, in an existing environment
pip install ./python/vllm-plugin
```

`wpkd` does the rest: for a MIX image on vLLM it writes a mix directory (an adapter config and `weightpack_mix.json`) and loads it with `/v1/load_lora_adapter`. The plugin stacks the experts into one adapter and gates each expert's rank block per token with the Arrow router, or with the trained router whose file `weightpack_mix.json` names; see the design doc's Token routing section.

For a bit1 fine-tune, `wpkd` starts vLLM with `WEIGHTPACK_DELTA_SLOTS=1` and `--max-loras 4`, exports the layer's blobs with a `weightpack_delta.json`, and loads that directory with `/v1/load_lora_adapter`. The plugin gives each adapter slot buffers beside every linear layer, embedding and RMSNorm (1/16 of the base's matrices per slot), writes a fine-tune's deltas into its slot's buffers, and adds each token's delta to the base's output; see the design doc's Full fine-tunes section.

- `weightpack_vllm/routing.py`: the stacking and the gating, in torch, tested without vLLM against `reference.py`, the NumPy ground truth.
- `weightpack_vllm/routing_kernel.py`: the gating as one Triton kernel for CUDA, which stops at once for a token of no mix, tested against `routing.py` on a GPU or under Triton's interpreter (`tests/test_routing_kernel.py`).
- `weightpack_vllm/deltas.py`: reading a bit1 layer, the per-slot buffers and the per-token delta products, in torch, tested without vLLM against the dense fine-tunes.
- `weightpack_vllm/bit1_kernel.py`: the CUDA kernels for the delta products (Triton), tested against `deltas.py` on a GPU or under Triton's interpreter (`tests/test_bit1_kernel.py`), and timed on a 7B model's projections by `tests/bench_bit1_kernels.py`.
- `weightpack_vllm/patches.py`: the patches into vLLM's LoRA path and its linear, embedding and RMSNorm layers, written against vLLM 0.34 and run end to end on 0.29.
- `tests/vllm_e2e.sh`: serves a tiny model with vLLM's CPU image in Docker, loads two experts and two mixes of them (top_k 1 and 2), and compares the replies with weightpack-engine's.
- `tests/vllm_bit1_e2e.sh`: the same for two bit1 fine-tunes and a LoRA adapter over a tiny model with tied embeddings and biased projections.

Limits: tensor parallel size 1, at most 8 experts per mix (`WEIGHTPACK_MAX_EXPERTS`), and as many fine-tunes per batch as adapter slots. With delta slots, a fine-tune runs at about two thirds of the base's throughput, and the base with the plugin at 88% of plain vLLM's, mostly for the token routing's gates, which run with no mix loaded (`tests/bench_bit1.py` on an RTX PRO 4500, a 7B-shaped model and four fine-tunes; the design doc has the numbers); and vLLM compiles without its cache. `WEIGHTPACK_DISABLE=1` leaves vLLM untouched.
