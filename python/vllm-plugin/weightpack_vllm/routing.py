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
experts that touch the module, go through a softmax. Equal logits go to the
expert listed first, so exactly k experts are picked. A trained router
(ROUTER train) has the same form with a bias, |x . w_i| + b_i, its w and b
read from the router file wpkd names.

This module needs only torch, so it is tested without vLLM; patches.py
connects it to vLLM.
"""

from __future__ import annotations

import json
import logging
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
    router = mix.get("router", "arrow")
    if mix.get("version") != 1 or router not in ("arrow", "trained"):
        raise ValueError(f"{path}: want version 1 and the arrow or trained router")
    if router == "trained" and not mix.get("router_file"):
        raise ValueError(f"{path}: a trained router needs router_file")
    if not mix.get("experts") or not 1 <= int(mix["top_k"]) <= len(mix["experts"]):
        raise ValueError(f"{path}: needs experts and top_k in 1..len(experts)")
    return mix


# A router file holds these per routed module, one row per expert.
ROUTER_WEIGHT = ".router.weight"  # (experts, in_features), float32
ROUTER_BIAS = ".router.bias"  # (experts,), float32


def load_router(path: str) -> dict:
    """A router file's rows by module: name -> (weight (experts, in), bias
    (experts)), float32 on the CPU, one row per expert in mix order."""
    from safetensors.torch import load_file

    tensors = load_file(path)
    out = {}
    for name, w in tensors.items():
        if name.endswith(ROUTER_WEIGHT):
            module = name.removesuffix(ROUTER_WEIGHT)
            bias = tensors.get(module + ROUTER_BIAS)
            if bias is None:
                raise ValueError(f"{path}: {module} has a router weight, no bias")
            out[module] = (w.float(), bias.float())
    return out


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
    the gate op runs in every batch with an adapter token, so captured
    graphs route tokens of mixes loaded later.
    """

    def __init__(self, n_slices, slots, max_rank, in_features, max_experts, device):
        import torch

        # The op is registered now, before vLLM compiles the model: torch
        # .compile cannot trace a registration.
        gate_op()
        self.max_experts = max_experts
        # Float32 whatever the weights' dtype, so routing reads the same
        # values weightpack-engine does.
        self.protos = torch.zeros(
            n_slices,
            slots,
            max_experts,
            in_features,
            device=device,
            dtype=torch.float32,
        )
        self.touch = torch.zeros(
            n_slices, slots, max_experts, device=device, dtype=torch.bool
        )
        self.bias = torch.zeros(
            n_slices, slots, max_experts, device=device, dtype=torch.float32
        )
        self.rank_expert = torch.zeros(slots, max_rank, device=device, dtype=torch.long)
        self.top_k = torch.ones(slots, device=device, dtype=torch.long)
        self.is_mix = torch.zeros(slots, device=device, dtype=torch.bool)

    def clear(self, slot: int) -> None:
        self.is_mix[slot] = False

    def set_slot(
        self, slot: int, ranks: list[int], top_k: int, a_slices, b_slices, rows=None
    ) -> None:
        """Routes the tokens of an adapter slot that holds a mix. a_slices and
        b_slices are the slot's stacked weights, (R, in) and (out, R) per
        slice; an expert touches a slice when its blocks are not zero. rows,
        for a trained router, gives each slice's (weight, bias) rows, one
        per expert; without it the router is Arrow's."""
        import torch

        if len(ranks) > self.max_experts:
            raise ValueError(
                f"a mix of {len(ranks)} experts; this vLLM routes at most "
                f"{self.max_experts} (WEIGHTPACK_MAX_EXPERTS)"
            )
        offsets = [sum(ranks[:i]) for i in range(len(ranks))]
        for s, (a, b) in enumerate(zip(a_slices, b_slices, strict=True)):
            self.protos[s, slot].zero_()
            self.bias[s, slot].zero_()
            self.touch[s, slot].zero_()
            trained = rows[s] if rows is not None else None
            for e, (o, r) in enumerate(zip(offsets, ranks, strict=True)):
                ae, be = a[o : o + r], b[:, o : o + r]
                if not (bool(ae.abs().sum() > 0) and bool(be.abs().sum() > 0)):
                    continue
                if rows is not None and trained is None:
                    raise ValueError(
                        "the trained router has no rows for a module the mix routes in"
                    )
                self.touch[s, slot, e] = True
                if trained is None:
                    v, bias = arrow_prototype(ae, be), 0.0
                else:
                    v, bias = trained[0][e], float(trained[1][e])
                self.protos[s, slot, e] = v.to(self.protos.device, self.protos.dtype)
                self.bias[s, slot, e] = bias
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
        return gates(x, token_slots, *self.tensors())

    def tensors(self):
        """The routing state as the gate op takes it."""
        return (
            self.is_mix,
            self.top_k,
            self.protos,
            self.bias,
            self.touch,
            self.rank_expert,
        )


def gates(x, token_slots, is_mix, top_k, protos, bias, touch, rank_expert):
    """Routing.gates() over a routing state's tensors."""
    import torch

    n_slices, slots, n_exp, in_f = protos.shape
    t = x.shape[0]
    slot = token_slots.clamp(min=0)
    mix = is_mix[slot] & (token_slots >= 0)
    k = top_k[slot].view(t, 1)
    experts = rank_expert[slot]  # (t, R)
    positions = torch.arange(n_exp, device=x.device).expand(t, n_exp)
    xf = x.to(torch.float32)
    out = []
    for s in range(n_slices):
        p = protos[s].reshape(slots * n_exp, in_f).to(torch.float32)
        logits = (xf @ p.T).abs().view(t, slots, n_exp)
        logits = logits.gather(1, slot.view(t, 1, 1).expand(t, 1, n_exp)).squeeze(1)
        logits = logits + bias[s][slot]
        logits = logits.masked_fill(~touch[s][slot], float("-inf"))
        # Exactly top_k experts, ties to the lower index, as the engine.
        order = logits.argsort(dim=1, descending=True, stable=True)
        place = torch.empty_like(order).scatter_(1, order, positions)
        logits = logits.masked_fill(place >= k, float("-inf"))
        coeff = torch.softmax(logits, dim=1).nan_to_num(0.0)
        g = coeff.gather(1, experts)
        out.append(torch.where(mix.view(t, 1), g, torch.ones_like(g)))
    return torch.stack(out)


# Each LoRA layer's routing, by the address of its first lora_a buffer,
# which vLLM allocates once. The gate op looks it up when it runs, so code
# torch.compile traces holds no reference to it: vLLM's compile cache
# would save such a reference by the object's id in this process, which
# the next process does not have.
_attached: dict[int, Routing] = {}


def attach(lora_a, routing: Routing) -> None:
    """Routes the LoRA layer whose first lora_a buffer is lora_a."""
    _attached[lora_a.data_ptr()] = routing


_gate_op = None


def gate_op():
    """weightpack::gate_(buffer, x, token_slots, lora_a, no_lora): buffer *=
    the gates of the layer whose first lora_a buffer is lora_a, in place,
    as one op opaque to torch.compile. On CUDA it runs routing_kernel's
    Triton kernel, which stops at once for a token of no mix; elsewhere
    gates() in torch. no_lora is vLLM's host flag for a batch with no
    adapter token (its own LoRA kernels skip on it, and it picks which CUDA
    graph replays), and the op then does nothing; so it does for a layer
    with no routing. WEIGHTPACK_KERNELS=triton runs the kernel on the CPU
    too, under TRITON_INTERPRET=1, to test it inside vLLM without a GPU."""
    global _gate_op
    if _gate_op is not None:
        return _gate_op
    import torch

    from . import routing_kernel

    forced = os.environ.get("WEIGHTPACK_KERNELS") == "triton"
    chosen = []

    @torch.library.custom_op(
        "weightpack::gate_",
        mutates_args=("buffer",),
        schema="(Tensor(a!) buffer, Tensor x, Tensor token_slots, Tensor lora_a, "
        "Tensor? no_lora) -> ()",
    )
    def gate_(buffer, x, token_slots, lora_a, no_lora):
        if no_lora is not None and bool(no_lora[0]):
            return
        routing = _attached.get(lora_a.data_ptr())
        if routing is None:
            return
        state = routing.tensors()
        kernel = (x.is_cuda or forced) and routing_kernel.available()
        if not chosen:
            chosen.append(kernel)
            logging.getLogger("vllm.weightpack").info(
                "weightpack: routing gates run on %s",
                "a Triton kernel" if kernel else "torch ops",
            )
        if kernel:
            routing_kernel.gate_(buffer, x, token_slots, *state)
        else:
            buffer.mul_(gates(x, token_slots, *state))

    gate_.register_fake(lambda *args: None)
    _gate_op = torch.ops.weightpack.gate_
    return _gate_op


def gated_lora_linear(
    wrapper,
    y,
    x,
    lora_a_stacked,
    lora_b_stacked,
    scale,
    output_slices,
    **kwargs,
) -> None:
    """vLLM's add_lora_linear with the shrink buffer gated per token: shrink,
    scale each rank position by its gate, expand. The layer's routing is
    the one attached to lora_a_stacked[0]. Returns y, as vLLM's does for
    platforms that do not update it in place."""
    import torch

    add_inputs = kwargs.pop("add_inputs", True)
    x = x.view(-1, x.shape[-1])
    t, rank = x.shape[0], lora_b_stacked[0].shape[-1]
    buffer = torch.zeros(
        (len(output_slices), t, rank), dtype=torch.float32, device=x.device
    )
    wrapper.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
    # vLLM's flag for a batch with no adapter token, where its LoRA kernels
    # skip; versions or platforms without it gate every batch.
    no_lora = getattr(
        getattr(wrapper, "token_mapping_meta", None), "no_lora_flag_cpu", None
    )
    gate_op()(buffer, x, wrapper._token_lora_indices[:t], lora_a_stacked[0], no_lora)
    wrapper.add_expand(
        y, buffer, lora_b_stacked, output_slices, add_inputs=add_inputs, **kwargs
    )
    return y
