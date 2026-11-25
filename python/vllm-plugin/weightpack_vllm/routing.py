"""Token routing for vLLM: a MIX served as one LoRA adapter whose rank blocks
are its experts, gated per token by Arrow.

vLLM applies a LoRA in two steps through a float32 buffer: ``z = x A^T``
(shrink), then ``y += z B^T`` (expand). Stack the experts of a MIX into one
adapter, ``A = [A_1; A_2; ...]`` and ``B = [s_1 B_1, s_2 B_2, ...]`` with
each expert's scaling folded in, and ``z B^T`` is the sum of the experts'
updates. Scaling each expert's block of ``z`` by the token's routing
coefficient between the two steps gives the mix::

    y += sum_i g_i(x) * s_i * B_i A_i x

The router is Arrow, as in weightpack-engine (weightpack_worker.mixing) and
PEFT: an expert's prototype in a module is the top right singular vector of
its B_i A_i; a token's logit is |x . v_i|; the top k logits, among the
experts that touch the module, go through a softmax.

This module needs only torch, so it is tested without vLLM; patches.py
connects it to vLLM.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

MIX_FILE = "weightpack_mix.json"

# Module names as Weightfiles name targets, as libs/go/build does.
TARGET_NAMES = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}
BLOCK = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")


def in_domain(module: str, targets: list[str], blocks: str) -> bool:
    """Whether a module is among a MIX's targets, in its blocks ("" for
    all, or ranges such as "0-3,8-31")."""
    last = module.rsplit(".", 1)[-1]
    if TARGET_NAMES.get(last, last) not in targets:
        return False
    if not blocks:
        return True
    m = BLOCK.search(module)
    if m is None:
        return False
    n = int(m.group(1))
    for part in blocks.split(","):
        lo, _, hi = part.partition("-")
        if int(lo) <= n <= int(hi or lo):
            return True
    return False


def load_mix(lora_dir: str) -> dict | None:
    """The mix description wpkd writes next to the adapter config, or None
    for an ordinary adapter."""
    path = os.path.join(lora_dir, MIX_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        mix = json.load(f)
    if mix.get("version") != 1 or mix.get("router", "arrow") != "arrow":
        raise ValueError(f"{path}: want version 1 and the arrow router")
    if not mix.get("experts") or not 1 <= int(mix["top_k"]) <= len(mix["experts"]):
        raise ValueError(f"{path}: needs experts and top_k in 1..len(experts)")
    return mix


def arrow_prototype(a, b):
    """The unit top right singular vector of B @ A, in float32 on the CPU,
    without forming B @ A: with A^T = QR, B @ A = (B R^T) Q^T."""
    import torch

    a32 = a.detach().to("cpu", torch.float32)  # (r, in)
    b32 = b.detach().to("cpu", torch.float32)  # (out, r)
    q, r = torch.linalg.qr(a32.T)
    _, _, vh = torch.linalg.svd(b32 @ r.T, full_matrices=False)
    v = q @ vh[0]
    return v / v.norm()


def stack_experts(
    experts: list[dict[str, tuple]],
    ranks: list[int],
    keep: Callable[[str], bool],
    dtype,
):
    """Stacks experts' LoRA weights into one adapter's.

    experts maps, per expert, module name -> (A (r, in), B (out, r), scaling).
    Returns module name -> (A (R, in), B (out, R)) for every kept module any
    expert touches, R = sum(ranks), with zero blocks for experts that do not
    touch it and each B block scaled.
    """
    import torch

    names = sorted({n for e in experts for n in e if keep(n)})
    out = {}
    for name in names:
        a0, b0, _ = next(e[name] for e in experts if name in e)
        in_f, out_f = a0.shape[1], b0.shape[0]
        a_parts, b_parts = [], []
        for e, r in zip(experts, ranks, strict=True):
            if name in e:
                a, b, scaling = e[name]
                if tuple(a.shape) != (r, in_f) or tuple(b.shape) != (out_f, r):
                    raise ValueError(
                        f"{name}: expert of rank {r} has A {tuple(a.shape)}, "
                        f"B {tuple(b.shape)}"
                    )
                a_parts.append(a.to(torch.float32))
                b_parts.append(b.to(torch.float32) * float(scaling))
            else:
                a_parts.append(torch.zeros(r, in_f))
                b_parts.append(torch.zeros(out_f, r))
        out[name] = (
            torch.cat(a_parts, dim=0).to(dtype),
            torch.cat(b_parts, dim=1).to(dtype),
        )
    return out


class Routing:
    """The routing state of one LoRA layer: which adapter slots hold a mix,
    and each mix's prototypes, rank blocks and top_k.

    Its buffers are allocated once, before vLLM captures CUDA graphs, and
    gates() always runs, so captured graphs route tokens of mixes loaded
    later.
    """

    def __init__(
        self, n_slices, slots, max_rank, in_features, max_experts, device, dtype
    ):
        import torch

        self.max_experts = max_experts
        self.protos = torch.zeros(
            n_slices, slots, max_experts, in_features, device=device, dtype=dtype
        )
        self.touch = torch.zeros(
            n_slices, slots, max_experts, device=device, dtype=torch.bool
        )
        self.rank_expert = torch.zeros(slots, max_rank, device=device, dtype=torch.long)
        self.top_k = torch.ones(slots, device=device, dtype=torch.long)
        self.is_mix = torch.zeros(slots, device=device, dtype=torch.bool)

    def clear(self, slot: int) -> None:
        self.is_mix[slot] = False

    def set_slot(
        self, slot: int, ranks: list[int], top_k: int, a_slices, b_slices
    ) -> None:
        """Routes the tokens of an adapter slot that holds a mix. a_slices and
        b_slices are the slot's stacked weights, (R, in) and (out, R) per
        slice; an expert touches a slice when its blocks are not zero."""
        import torch

        if len(ranks) > self.max_experts:
            raise ValueError(
                f"a mix of {len(ranks)} experts; this vLLM routes at most "
                f"{self.max_experts} (WEIGHTPACK_MAX_EXPERTS)"
            )
        offsets = [sum(ranks[:i]) for i in range(len(ranks))]
        for s, (a, b) in enumerate(zip(a_slices, b_slices, strict=True)):
            self.protos[s, slot].zero_()
            self.touch[s, slot].zero_()
            for e, (o, r) in enumerate(zip(offsets, ranks, strict=True)):
                ae, be = a[o : o + r], b[:, o : o + r]
                if bool(ae.abs().sum() > 0) and bool(be.abs().sum() > 0):
                    self.touch[s, slot, e] = True
                    self.protos[s, slot, e] = arrow_prototype(ae, be).to(
                        self.protos.device, self.protos.dtype
                    )
        experts = torch.zeros(self.rank_expert.shape[1], dtype=torch.long)
        for e, (o, r) in enumerate(zip(offsets, ranks, strict=True)):
            experts[o : o + r] = e
        self.rank_expert[slot] = experts.to(self.rank_expert.device)
        self.top_k[slot] = top_k
        self.is_mix[slot] = True

    def gates(self, x, token_slots):
        """Per-token gates on the shrink buffer, (slices, tokens, R) in
        float32: each mix token's routing coefficient for the expert that
        owns each rank position, and 1 for every other token. token_slots
        is each token's adapter slot, -1 for none."""
        import torch

        n_slices, slots, n_exp, in_f = self.protos.shape
        t = x.shape[0]
        slot = token_slots.clamp(min=0)
        mix = self.is_mix[slot] & (token_slots >= 0)
        k = (self.top_k[slot] - 1).clamp(0, n_exp - 1).view(t, 1)
        experts = self.rank_expert[slot]  # (t, R)
        xf = x.to(torch.float32)
        out = []
        for s in range(n_slices):
            protos = self.protos[s].reshape(slots * n_exp, in_f).to(torch.float32)
            logits = (xf @ protos.T).abs().view(t, slots, n_exp)
            logits = logits.gather(1, slot.view(t, 1, 1).expand(t, 1, n_exp)).squeeze(1)
            logits = logits.masked_fill(~self.touch[s][slot], float("-inf"))
            kth = logits.sort(dim=1, descending=True).values.gather(1, k)
            logits = logits.masked_fill(logits < kth, float("-inf"))
            coeff = torch.softmax(logits, dim=1).nan_to_num(0.0)
            g = coeff.gather(1, experts)
            out.append(torch.where(mix.view(t, 1), g, torch.ones_like(g)))
        return torch.stack(out)


def gated_lora_linear(
    wrapper,
    routing: Routing,
    y,
    x,
    lora_a_stacked,
    lora_b_stacked,
    scale,
    output_slices,
    **kwargs,
) -> None:
    """vLLM's add_lora_linear with the shrink buffer gated per token: shrink,
    scale each rank position by its gate, expand. Returns y, as vLLM's does
    for platforms that do not update it in place."""
    import torch

    add_inputs = kwargs.pop("add_inputs", True)
    x = x.view(-1, x.shape[-1])
    t, rank = x.shape[0], lora_b_stacked[0].shape[-1]
    buffer = torch.zeros(
        (len(output_slices), t, rank), dtype=torch.float32, device=x.device
    )
    wrapper.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
    buffer.mul_(routing.gates(x, wrapper._token_lora_indices[:t]))
    wrapper.add_expand(
        y, buffer, lora_b_stacked, output_slices, add_inputs=add_inputs, **kwargs
    )
    return y
