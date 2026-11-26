# weightpack-vllm

[Weightpack](https://github.com/rizkaitbrik/weightpack) plugin for vLLM: it routes each token of a MIX across the MIX's LoRA experts, as `weightpack-engine` does. vLLM loads it through the `vllm.general_plugins` entry point.

Install it in vLLM's environment (vLLM is not a declared dependency, so CI stays light):

```bash
uv tool install vllm --with "weightpack-vllm @ git+https://github.com/rizkaitbrik/weightpack#subdirectory=python/vllm-plugin"
# or, in an existing environment
pip install ./python/vllm-plugin
```

`wpkd` does the rest: for a MIX image on vLLM it writes a mix directory (an adapter config and `weightpack_mix.json`) and loads it with `/v1/load_lora_adapter`. The plugin stacks the experts into one adapter and gates each expert's rank block per token with the Arrow router; see the design doc's Token routing section.

- `weightpack_vllm/routing.py`: the stacking and the gating, in torch, tested without vLLM against `reference.py`, the NumPy ground truth.
- `weightpack_vllm/patches.py`: the four small patches into vLLM's LoRA path, written against vLLM 0.34 and run end to end on 0.29.
- `tests/vllm_e2e.sh`: serves a tiny model with vLLM's CPU image in Docker, loads two experts and two mixes of them (top_k 1 and 2), and compares the replies with weightpack-engine's.

Limits: tensor parallel size 1, and at most 8 experts per mix (`WEIGHTPACK_MAX_EXPERTS`). `WEIGHTPACK_DISABLE=1` leaves vLLM untouched.
