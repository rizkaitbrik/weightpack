"""The Triton kernel for token routing's gates: it scales the shrink buffer
of a LoRA layer in place, each mix token's rank positions by its routing
coefficient for the expert that owns them, as routing.gates() computes them.

One program per token and slice. A token whose slot holds no mix (or no
adapter) keeps a gate of 1, so its program stops at once: a batch without a
mix costs one launch per layer. A mix token's program computes its own
slot's logits only, |x . v_e| + b_e for each expert e that touches the
slice, picks exactly top_k of them (ties to the lower index, as
weightpack-engine does), takes their softmax and scales the token's rank
positions. Everything is read on the device, so a CUDA graph captures it.
TRITON_INTERPRET=1 runs it on the CPU, for tests.
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
    def gate_kernel(
        buf_ptr,  # (SL, T, R) float32, the shrink buffer, scaled in place
        x_ptr,  # (T, IN)
        slots_ptr,  # (T,) each token's adapter slot, -1 for none
        is_mix_ptr,  # (S,) bool
        top_k_ptr,  # (S,) int64
        protos_ptr,  # (SL, S, E, IN) float32
        bias_ptr,  # (SL, S, E) float32
        touch_ptr,  # (SL, S, E) bool
        rank_expert_ptr,  # (S, R) int64: the expert owning each rank position
        T,
        IN,
        R,
        S,
        E,
        stride_x,
        BE: tl.constexpr,
        BK: tl.constexpr,
        BR: tl.constexpr,
    ):
        t = tl.program_id(0)
        sl = tl.program_id(1)
        s = tl.load(slots_ptr + t)
        if s < 0:
            return
        if tl.load(is_mix_ptr + s) == 0:
            return
        e = tl.arange(0, BE)
        e_ok = e < E
        row = (sl * S + s).to(tl.int64) * E
        acc = tl.zeros((BE,), dtype=tl.float32)
        for k0 in range(0, IN, BK):
            k = k0 + tl.arange(0, BK)
            k_ok = k < IN
            x = tl.load(x_ptr + t.to(tl.int64) * stride_x + k, mask=k_ok, other=0.0)
            p = tl.load(
                protos_ptr + (row + e)[:, None] * IN + k[None, :],
                mask=e_ok[:, None] & k_ok[None, :],
                other=0.0,
            )
            acc += tl.sum(p * x.to(tl.float32)[None, :], axis=1)
        bias = tl.load(bias_ptr + row + e, mask=e_ok, other=0.0)
        touch = tl.load(touch_ptr + row + e, mask=e_ok, other=0) != 0
        logit = tl.where(touch & e_ok, tl.abs(acc) + bias, float("-inf"))
        # Expert i's place: the experts ahead of it, by logit, then index.
        li = logit[:, None]
        lj = logit[None, :]
        ahead = (lj > li) | ((lj == li) & (e[None, :] < e[:, None]))
        place = tl.sum(ahead.to(tl.int32), axis=1)
        keep = (place < tl.load(top_k_ptr + s)) & (logit > float("-inf"))
        top = tl.max(tl.where(keep, logit, float("-inf")), axis=0)
        w = tl.where(keep, tl.exp(logit - top), 0.0)
        total = tl.sum(w, axis=0)
        coeff = tl.where(total > 0, w / total, 0.0)
        for r0 in range(0, R, BR):
            r = r0 + tl.arange(0, BR)
            r_ok = r < R
            owner = tl.load(
                rank_expert_ptr + s.to(tl.int64) * R + r, mask=r_ok, other=0
            )
            g = tl.sum(
                tl.where(owner[:, None] == e[None, :], coeff[None, :], 0.0), axis=1
            )
            dst = buf_ptr + (sl * T + t).to(tl.int64) * R + r
            b = tl.load(dst, mask=r_ok, other=0.0)
            tl.store(dst, b * g, mask=r_ok)

    _kernel = gate_kernel
    return _kernel


def gate_(buffer, x, token_slots, is_mix, top_k, protos, bias, touch, rank_expert):
    """buffer (slices, tokens, R) *= each token's gates, in place."""
    import triton

    gate_kernel = _build()
    n_slices, t, rank = buffer.shape
    if t == 0:
        return
    _, slots, n_exp, in_f = protos.shape
    gate_kernel[(t, n_slices)](
        buffer,
        x,
        token_slots,
        is_mix,
        top_k,
        protos,
        bias,
        touch,
        rank_expert,
        t,
        in_f,
        rank,
        slots,
        n_exp,
        x.stride(0),
        BE=triton.next_power_of_2(max(n_exp, 2)),
        BK=min(1024, triton.next_power_of_2(in_f)),
        BR=min(64, triton.next_power_of_2(rank)),
    )
