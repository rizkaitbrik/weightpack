# weightpack-vllm

[Weightpack](https://github.com/rizkaitbrik/weightpack) plugin for vLLM: composition loading, expert pool and routers. vLLM loads it through the `vllm.general_plugins` entry point.

Install it in the same environment as vLLM (vLLM is not a declared dependency, so CI stays light):

```bash
pip install vllm
pip install ./python/vllm-plugin
```

`weightpack_vllm.reference` holds the NumPy reference for a mix of LoRA experts, `y = Wx + Σ gᵢ BᵢAᵢx`, that the GPU kernels are tested against.
