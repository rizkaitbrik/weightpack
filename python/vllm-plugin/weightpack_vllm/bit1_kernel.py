"""The CUDA kernel for bit1 deltas: each token's row_scale * (sign x), read
from packed signs, for the tokens of every adapter slot at once.

Tokens are sorted by slot on the device; each program takes a block of one
slot's tokens and a block of output rows, unpacks that slot's signs tile by
tile into +1 and -1 in the activation dtype, and multiplies on tensor cores.
Nothing is read on the host, so a CUDA graph captures it, and a slot's signs
are read once per block of its tokens, not once per token.
"""

from __future__ import annotations

_triton = None


def available() -> bool:
    global _triton
    if _triton is None:
        try:
            import triton  # noqa: F401

            _triton = True
        except ImportError:
            _triton = False
    return _triton


_kernel = None


def _build():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        x_ptr,  # (T, K)
        order_ptr,  # (T,) token ids sorted by slot
        start_ptr,  # (S,) each slot's first position in order
        count_ptr,  # (S,) each slot's tokens
        signs_ptr,  # (S, N, KB) uint8
        scale_ptr,  # (S, N) float32
        out_ptr,  # (T, N) float32
        N,
        K,
        KB,
        stride_xt,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        IEEE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        s = tl.program_id(2)
        count = tl.load(count_ptr + s)
        if pid_m * BM >= count:
            return
        start = tl.load(start_ptr + s)
        m = pid_m * BM + tl.arange(0, BM)
        m_ok = m < count
        tok = tl.load(order_ptr + start + m, mask=m_ok, other=0)
        n = pid_n * BN + tl.arange(0, BN)
        n_ok = n < N
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        signs_row = signs_ptr + s.to(tl.int64) * N * KB + n.to(tl.int64) * KB
        for k0 in range(0, K, BK):
            k = k0 + kk
            k_ok = k < K
            x = tl.load(
                x_ptr + tok.to(tl.int64)[:, None] * stride_xt + k[None, :],
                mask=m_ok[:, None] & k_ok[None, :],
                other=0.0,
            )
            byte = tl.load(
                signs_row[:, None] + (k // 8)[None, :],
                mask=n_ok[:, None] & k_ok[None, :],
                other=0,
            )
            bit = (byte >> (7 - (k % 8)).to(tl.uint8)[None, :]) & 1
            w = (bit.to(tl.float32) * 2.0 - 1.0).to(x.dtype)
            if IEEE:
                acc += tl.dot(x, tl.trans(w), input_precision="ieee")
            else:
                acc += tl.dot(x, tl.trans(w))
        scale = tl.load(scale_ptr + s.to(tl.int64) * N + n, mask=n_ok, other=0.0)
        acc = acc * scale[None, :]
        tl.store(
            out_ptr + tok.to(tl.int64)[:, None] * N + n[None, :],
            acc,
            mask=m_ok[:, None] & n_ok[None, :],
        )

    _kernel = kernel
    return kernel


def linear(x, slots, signs, row_scale, active):
    """deltas.linear_delta on CUDA: (tokens, rows) float32."""
    import torch
    import triton

    kernel = _build()
    x = x.contiguous()
    t, k = x.shape
    n_slots, n, kb = signs.shape
    out = torch.zeros(t, n, dtype=torch.float32, device=x.device)
    if t == 0:
        return out
    s = slots.to(torch.int64)
    ok = (s >= 0) & (active[s.clamp(min=0)] > 0)
    # Tokens of no active slot go to slot n_slots, which no program reads.
    key = torch.where(ok, s, n_slots)
    order = torch.sort(key, stable=True).indices.to(torch.int32)
    hits = key.view(-1, 1) == torch.arange(n_slots, device=x.device).view(1, -1)
    count = hits.sum(0).to(torch.int32)
    start = (torch.cumsum(count, 0) - count).to(torch.int32)
    bm, bn, bk = 16, 64, 64
    grid = (triton.cdiv(t, bm), triton.cdiv(n, bn), n_slots)
    kernel[grid](
        x,
        order,
        start,
        count,
        signs,
        row_scale,
        out,
        n,
        k,
        kb,
        x.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        IEEE=x.dtype == torch.float32,
    )
    return out
