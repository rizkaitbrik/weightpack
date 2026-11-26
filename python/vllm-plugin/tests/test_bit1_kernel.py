"""The CUDA kernel for bit1 deltas against deltas.linear_delta. Needs an
NVIDIA GPU and Triton; skipped elsewhere."""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from weightpack_vllm import bit1_kernel  # noqa: E402
from weightpack_vllm.deltas import linear_delta  # noqa: E402


def setup(t, n, k, slots, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, k, device="cuda", generator=g).to(dtype)
    kb = (k + 7) // 8
    signs = torch.randint(0, 256, (slots, n, kb), device="cuda", generator=g)
    signs = signs.to(torch.uint8)
    # Padding bits past k are zero, as ParamSlots packs them.
    if k % 8:
        keep = torch.tensor(
            [0xFF << (8 - k % 8) & 0xFF], dtype=torch.uint8, device="cuda"
        )
        signs[:, :, -1] &= keep
    scale = torch.rand(slots, n, device="cuda", generator=g) * 1e-2
    active = torch.ones(slots, dtype=torch.int32, device="cuda")
    active[-1] = 0
    s = torch.randint(-1, slots, (t,), device="cuda", generator=g)
    return x, s, signs, scale, active


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "t,n,k", [(1, 64, 64), (7, 100, 13), (300, 896, 896), (33, 4864, 896)]
)
def test_kernel_matches_reference(dtype, t, n, k):
    x, s, signs, scale, active = setup(t, n, k, 4, dtype)
    got = bit1_kernel.linear(x, s, signs, scale, active)
    want = linear_delta(x, s, signs, scale, active, graph_safe=False)
    tol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(
        got, want, rtol=tol, atol=tol * float(scale.max()) * k**0.5
    )
    # Tokens of no slot, or of an inactive one, get nothing.
    idle = (s < 0) | (active[s.clamp(min=0)] == 0)
    assert not got[idle].any()


def test_kernel_is_captured_by_cuda_graphs():
    x, s, signs, scale, active = setup(64, 512, 512, 4, torch.bfloat16)
    out = bit1_kernel.linear(x, s, signs, scale, active)  # compiles the kernel
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = bit1_kernel.linear(x, s, signs, scale, active)
    # New slots and new signs, same buffers: the replay sees them.
    s.copy_(torch.randint(-1, 4, s.shape, device="cuda"))
    signs.copy_(torch.randint(0, 256, signs.shape, device="cuda").to(torch.uint8))
    graph.replay()
    want = linear_delta(x, s, signs, scale, active, graph_safe=False)
    torch.testing.assert_close(
        out, want, rtol=2e-2, atol=2e-2 * float(scale.max()) * 512**0.5
    )


def _ms(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1e3


def test_kernel_speed():
    """Prints the kernel's time beside the base matmul's, for a decode batch
    and a prefill of a 7B-sized projection."""
    for t in (16, 2048):
        x, s, signs, scale, active = setup(t, 4096, 4096, 4, torch.bfloat16)
        w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        base = _ms(lambda x=x, w=w: x @ w.T)
        delta = _ms(
            lambda x=x, s=s, signs=signs, scale=scale, active=active: (
                bit1_kernel.linear(x, s, signs, scale, active)
            )
        )
        print(
            f"{t:5d} tokens, 4096x4096: base {base:.3f} ms, bit1 delta {delta:.3f} ms"
        )
        assert np.isfinite(delta)
