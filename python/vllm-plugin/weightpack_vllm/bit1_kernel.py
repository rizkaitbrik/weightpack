"""The Triton kernels for bit1 deltas: each adds its tokens' deltas into the
base's output in place, reading each token's adapter slot on the device.

- linear_: y += row_scale * (sign x) (+ the bias's delta) for each token in
  a slot with a delta. Up to SMALL tokens (a decode batch), one program per
  block of output rows and slot covers every token, its slot's tokens kept
  and the others masked to zero, so each slot's signs are read and unpacked
  once; past that (a prefill), one per block of tokens, rows and slot,
  stopping at once if no token of the block is the slot's, so a block of
  one sequence's tokens costs one pass.
- norm_: y += normalized(x) * delta, a fine-tuned RMSNorm's weight.
- rows_: y += row_scale * sign for each token's embedding row.

Signs are packed by row, eight to a byte, most significant bit first; each
byte is loaded once and expanded in registers, and the products run on
tensor cores. Nothing is read on the host and nothing else runs, so a CUDA
graph captures each op as one launch, which exits early for the base's
tokens. TRITON_INTERPRET=1 runs them on the CPU, for tests.
"""

from __future__ import annotations

# Decode-sized batches: every token in one program.
SMALL = 128

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


_kernels = None


def _build():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def linear_kernel(
        y_ptr,  # (T, N) the base's output, updated in place
        x_ptr,  # (T, K)
        slots_ptr,  # (T,) each token's slot, -1 for none
        active_ptr,  # (S,) int32: the slot's delta changes the weight
        signs_ptr,  # (S, N, KB) uint8
        scale_ptr,  # (S, N) float32
        bias_ptr,  # (S, N) float32, the bias's delta
        bias_active_ptr,  # (S,) int32
        T,
        N,  # rows written: y's width
        NR,  # rows in the slot buffers (at least N)
        K,
        KB,
        stride_y,
        stride_x,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        ALL_TOKENS: tl.constexpr,
        IEEE: tl.constexpr,
    ):
        if ALL_TOKENS:
            pid_m = 0
            pid_n = tl.program_id(0)
            s = tl.program_id(1)
        else:
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            s = tl.program_id(2)
        w_on = tl.load(active_ptr + s)
        b_on = tl.load(bias_active_ptr + s)
        if w_on + b_on == 0:
            return
        m = pid_m * BM + tl.arange(0, BM)
        slot = tl.load(slots_ptr + m, mask=m < T, other=-1)
        mine = slot == s
        if tl.sum(mine.to(tl.int32), axis=0) == 0:
            return
        n = pid_n * BN + tl.arange(0, BN)
        n_ok = n < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if w_on != 0:
            kb = tl.arange(0, BK // 8)
            shifts = 7 - tl.arange(0, 8)
            signs_row = signs_ptr + s.to(tl.int64) * NR * KB + n.to(tl.int64) * KB
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                x = tl.load(
                    x_ptr + m.to(tl.int64)[:, None] * stride_x + k[None, :],
                    mask=mine[:, None] & (k < K)[None, :],
                    other=0.0,
                )
                byte_k = k0 // 8 + kb
                byte = tl.load(
                    signs_row[:, None] + byte_k[None, :],
                    mask=n_ok[:, None] & (byte_k < KB)[None, :],
                    other=0,
                ).to(tl.int32)
                bits = (byte[:, :, None] >> shifts[None, None, :]) & 1
                w = tl.reshape(bits, (BN, BK)).to(tl.float32) * 2.0 - 1.0
                w = w.to(x.dtype)
                if IEEE:
                    acc += tl.dot(x, tl.trans(w), input_precision="ieee")
                else:
                    acc += tl.dot(x, tl.trans(w))
            scale = tl.load(scale_ptr + s.to(tl.int64) * NR + n, mask=n_ok, other=0.0)
            acc = acc * scale[None, :]
        if b_on != 0:
            bias = tl.load(bias_ptr + s.to(tl.int64) * NR + n, mask=n_ok, other=0.0)
            acc = acc + bias[None, :]
        out = y_ptr + m.to(tl.int64)[:, None] * stride_y + n[None, :]
        keep = mine[:, None] & n_ok[None, :]
        y = tl.load(out, mask=keep, other=0.0)
        tl.store(out, (y.to(tl.float32) + acc).to(y.dtype), mask=keep)

    @triton.jit
    def norm_kernel(
        y_ptr,  # (R, H) the norm's output, updated in place
        x_ptr,  # (R, H) its input
        slots_ptr,  # (T,) each token's slot
        active_ptr,  # (S,) int32
        delta_ptr,  # (S, H) float32, the weight's delta
        H,
        VAR,  # how many of a row's entries its variance covers
        PER_TOKEN,  # rows per token: heads, for a per-head norm
        stride_y,
        stride_x,
        eps,
        BH: tl.constexpr,
    ):
        r = tl.program_id(0)
        s = tl.load(slots_ptr + r // PER_TOKEN)
        if s < 0:
            return
        if tl.load(active_ptr + s) == 0:
            return
        h = tl.arange(0, BH)
        x = tl.load(x_ptr + r.to(tl.int64) * stride_x + h, mask=h < H, other=0.0)
        x = x.to(tl.float32)
        var = tl.sum(tl.where(h < VAR, x * x, 0.0), axis=0) / VAR
        d = tl.load(delta_ptr + s.to(tl.int64) * H + h, mask=h < H, other=0.0)
        out = y_ptr + r.to(tl.int64) * stride_y + h
        y = tl.load(out, mask=h < H, other=0.0)
        y = y.to(tl.float32) + x * tl.rsqrt(var + eps) * d
        tl.store(out, y.to(out.dtype.element_ty), mask=h < H)

    @triton.jit
    def rows_kernel(
        y_ptr,  # (T, H) the embedding's output, updated in place
        ids_ptr,  # (T,)
        slots_ptr,  # (T,)
        active_ptr,  # (S,) int32
        signs_ptr,  # (S, V, HB) uint8
        scale_ptr,  # (S, V) float32
        V,
        H,
        HB,
        stride_y,
        BH: tl.constexpr,
    ):
        t = tl.program_id(0)
        s = tl.load(slots_ptr + t)
        if s < 0:
            return
        if tl.load(active_ptr + s) == 0:
            return
        row = tl.load(ids_ptr + t).to(tl.int64)
        h = tl.arange(0, BH)
        byte = tl.load(
            signs_ptr + (s.to(tl.int64) * V + row) * HB + h // 8,
            mask=h < H,
            other=0,
        ).to(tl.int32)
        sign = ((byte >> (7 - h % 8)) & 1).to(tl.float32) * 2.0 - 1.0
        scale = tl.load(scale_ptr + s.to(tl.int64) * V + row)
        out = y_ptr + t.to(tl.int64) * stride_y + h
        y = tl.load(out, mask=h < H, other=0.0)
        y = y.to(tl.float32) + scale * sign
        tl.store(out, y.to(out.dtype.element_ty), mask=h < H)

    _kernels = (linear_kernel, norm_kernel, rows_kernel)
    return _kernels


_zeros: dict = {}


def _idle(n: int, device):
    """A slot flag buffer of n zeros, for a layer with no bias."""
    import torch

    key = (n, str(device))
    if key not in _zeros:
        _zeros[key] = torch.zeros(n, dtype=torch.int32, device=device)
    return _zeros[key]


def linear_(y, x, slots, signs, row_scale, active, bias=None, bias_active=None):
    """y (tokens, rows) += each token's delta product (and bias delta)."""
    import torch
    import triton

    linear_kernel, _, _ = _build()
    t, k = x.shape
    n_slots, n_rows, kb = signs.shape
    n = min(n_rows, y.shape[1])
    if t == 0:
        return
    if bias is None:
        bias, bias_active = row_scale, _idle(n_slots, x.device)
    bn, bk = 64, 128
    if t <= SMALL:
        bm = max(16, triton.next_power_of_2(t))
        grid = (triton.cdiv(n, bn), n_slots)
    else:
        bm = 64
        grid = (triton.cdiv(t, bm), triton.cdiv(n, bn), n_slots)
    linear_kernel[grid](
        y,
        x,
        slots,
        active,
        signs,
        row_scale,
        bias,
        bias_active,
        t,
        n,
        n_rows,
        k,
        kb,
        y.stride(0),
        x.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        ALL_TOKENS=t <= SMALL,
        IEEE=x.dtype == torch.float32,
    )


def norm_(y, x, slots, delta, active, eps, var_size=-1):
    """y (rows, H) += normalized(x) * delta for each token's row(s)."""
    import triton

    _, norm_kernel, _ = _build()
    rows, h = x.shape
    t = slots.shape[0]
    if rows == 0:
        return
    norm_kernel[(rows,)](
        y,
        x,
        slots,
        active,
        delta,
        h,
        h if var_size < 0 else var_size,
        max(1, rows // max(1, t)),
        y.stride(0),
        x.stride(0),
        eps,
        BH=triton.next_power_of_2(h),
    )


def rows_(y, ids, slots, signs, row_scale, active):
    """y (tokens, H) += each token's embedding row delta."""
    import triton

    _, _, rows_kernel = _build()
    t, h = y.shape
    if t == 0:
        return
    n_slots, v, hb = signs.shape
    rows_kernel[(t,)](
        y,
        ids,
        slots,
        active,
        signs,
        row_scale,
        v,
        h,
        hb,
        y.stride(0),
        BH=triton.next_power_of_2(h),
    )
