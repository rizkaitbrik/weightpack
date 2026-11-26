"""The Triton kernels for bit1 deltas against deltas.py's torch ops. They run
on an NVIDIA GPU, or on the CPU under Triton's interpreter
(TRITON_INTERPRET=1); skipped without either."""

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)

from weightpack_vllm import bit1_kernel  # noqa: E402
from weightpack_vllm.deltas import (  # noqa: E402
    linear_delta,
    rms_normalized,
    rows_delta,
    unpack,
)

SLOTS = 4
# The interpreter does float32 and float16; a GPU bfloat16 too.
DTYPES = [torch.float32, torch.float16] + ([torch.bfloat16] if DEVICE == "cuda" else [])
TOL = {torch.float32: 1e-4, torch.float16: 4e-3, torch.bfloat16: 3e-2}


def gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def packed(n, k, g):
    kb = (k + 7) // 8
    signs = torch.randint(0, 256, (SLOTS, n, kb), device=DEVICE, generator=g)
    signs = signs.to(torch.uint8)
    if k % 8:  # padding bits past k are zero, as ParamSlots packs them
        signs[:, :, -1] &= (0xFF << (8 - k % 8)) & 0xFF
    return signs


def active_but_last():
    a = torch.ones(SLOTS, dtype=torch.int32, device=DEVICE)
    a[-1] = 0
    return a


SIZES = [(1, 64, 64), (7, 100, 13), (33, 200, 96)] + (
    [(150, 96, 40)] if DEVICE == "cpu" else [(300, 896, 896), (40, 4864, 896)]
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("t,n,k", SIZES)
def test_linear_adds_each_tokens_delta(dtype, bias, t, n, k):
    g = gen(0)
    x = torch.randn(t, k, device=DEVICE, generator=g).to(dtype)
    y = torch.randn(t, n, device=DEVICE, generator=g).to(dtype)
    slots = torch.randint(-1, SLOTS, (t,), device=DEVICE, generator=g)
    signs = packed(n, k, g)
    scale = torch.rand(SLOTS, n, device=DEVICE, generator=g) * 1e-2
    active = active_but_last()
    bias_delta = torch.randn(SLOTS, n, device=DEVICE, generator=g) * 1e-2
    bias_active = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device=DEVICE)
    want = y.float() + linear_delta(x, slots, signs, scale, active, graph_safe=False)
    s = slots.clamp(min=0)
    if bias:
        on = (slots >= 0) & (bias_active[s] > 0)
        want = want + torch.where(on.view(-1, 1), bias_delta[s], 0.0)
    got = y.clone()
    bit1_kernel.linear_(
        got,
        x,
        slots,
        signs,
        scale,
        active,
        bias_delta if bias else None,
        bias_active if bias else None,
    )
    tol = TOL[dtype]
    torch.testing.assert_close(got.float(), want.to(dtype).float(), rtol=tol, atol=tol)
    # Tokens of no slot, or of a slot with nothing here, keep the base's.
    idle = (slots < 0) | (active[s] == 0)
    if bias:
        idle &= (slots < 0) | (bias_active[s] == 0)
    assert torch.equal(got[idle], y[idle])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", [1, 3])
def test_norm_adds_the_weights_delta(dtype, heads):
    g = gen(1)
    t, h = 6, 40
    x = torch.randn(t, heads, h, device=DEVICE, generator=g).to(dtype)
    y = torch.randn(t, heads, h, device=DEVICE, generator=g).to(dtype)
    slots = torch.tensor([0, -1, 3, 1, 0, 2], device=DEVICE)
    delta = torch.randn(SLOTS, h, device=DEVICE, generator=g) * 1e-2
    active = active_but_last()
    s = slots.clamp(min=0)
    on = ((slots >= 0) & (active[s] > 0)).view(t, 1, 1)
    want = y.float() + torch.where(
        on, rms_normalized(x, 1e-6) * delta[s].view(t, 1, h), 0.0
    )
    got = y.clone()
    bit1_kernel.norm_(got.view(-1, h), x.view(-1, h), slots, delta, active, 1e-6)
    tol = TOL[dtype]
    torch.testing.assert_close(got.float(), want.to(dtype).float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rows_add_the_embeddings_delta(dtype):
    g = gen(2)
    t, vocab, h = 7, 30, 37
    ids = torch.randint(0, vocab, (t,), device=DEVICE, generator=g)
    y = torch.randn(t, h, device=DEVICE, generator=g).to(dtype)
    slots = torch.randint(-1, SLOTS, (t,), device=DEVICE, generator=g)
    signs = packed(vocab, h, g)
    scale = torch.rand(SLOTS, vocab, device=DEVICE, generator=g) * 1e-2
    active = active_but_last()

    class Slots:
        pass

    p = Slots()
    p.signs, p.row_scale, p.active, p.cols = signs, scale, active, h
    want = y.float() + rows_delta(ids, slots, p, torch.float32)
    got = y.clone()
    bit1_kernel.rows_(got, ids, slots, signs, scale, active)
    tol = TOL[dtype]
    torch.testing.assert_close(got.float(), want.to(dtype).float(), rtol=tol, atol=tol)


def test_unpack_is_msb_first():
    signs = torch.tensor([[0b10000001]], dtype=torch.uint8)
    got = unpack(signs, 8, torch.float32).tolist()
    assert got == [[1, -1, -1, -1, -1, -1, -1, 1]]


@pytest.mark.skipif(DEVICE != "cuda", reason="CUDA graphs")
def test_linear_is_captured_by_cuda_graphs():
    g = gen(3)
    t, n, k = 64, 512, 512
    x = torch.randn(t, k, device="cuda", generator=g).to(torch.bfloat16)
    y0 = torch.randn(t, n, device="cuda", generator=g).to(torch.bfloat16)
    slots = torch.randint(-1, SLOTS, (t,), device="cuda", generator=g)
    signs = packed(n, k, g)
    scale = torch.rand(SLOTS, n, device="cuda", generator=g) * 1e-2
    active = active_but_last()
    y = y0.clone()
    bit1_kernel.linear_(y, x, slots, signs, scale, active)  # compiles
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bit1_kernel.linear_(y, x, slots, signs, scale, active)
    # New slots and signs, same buffers: the replay sees them.
    slots.copy_(torch.randint(-1, SLOTS, slots.shape, device="cuda"))
    signs.copy_(packed(n, k, g))
    y.copy_(y0)
    graph.replay()
    want = y0.float() + linear_delta(x, slots, signs, scale, active, graph_safe=False)
    torch.testing.assert_close(
        y.float(), want.to(torch.bfloat16).float(), rtol=3e-2, atol=3e-2
    )


def _ms(fn, reps=50):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1e3


@pytest.mark.skipif(DEVICE != "cuda", reason="timing")
def test_speed():
    """Prints the fused linear op's time beside the base matmul's, inside a
    CUDA graph, for a 4096x4096 projection and four slots: decode batches of
    no fine-tune (the base alone) and of mixed slots, and a prefill of four
    sequences."""
    g = gen(4)
    signs = packed(4096, 4096, g)
    scale = torch.rand(SLOTS, 4096, device="cuda", generator=g) * 1e-2
    active = torch.ones(SLOTS, dtype=torch.int32, device="cuda")
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for label, t, how in [
        ("decode 32, base alone", 32, "none"),
        ("decode 32, mixed", 32, "mixed"),
        ("decode 128, mixed", 128, "mixed"),
        ("prefill 2048, 4 seqs", 2048, "runs"),
    ]:
        x = torch.randn(t, 4096, device="cuda", dtype=torch.bfloat16)
        y = torch.zeros(t, 4096, device="cuda", dtype=torch.bfloat16)
        slots = {
            "none": torch.full((t,), -1, device="cuda"),
            "mixed": torch.randint(0, SLOTS, (t,), device="cuda"),
            "runs": torch.arange(t, device="cuda") // (t // 4),
        }[how]
        base = _ms(lambda x=x, w=w: x @ w.T)
        delta = _ms(
            lambda x=x, y=y, s=slots: bit1_kernel.linear_(y, x, s, signs, scale, active)
        )
        print(f"{label:24s} base {base:.3f} ms, delta {delta:.3f} ms")
        assert np.isfinite(delta)
