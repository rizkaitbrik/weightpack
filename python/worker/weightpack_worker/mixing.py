"""Token-level mixing of LoRA experts with Arrow routing.

A MIX serves its experts side by side on one base. In each LoRA module the
MIX covers (its targets, in its blocks), every token picks its own experts:

    y = W x + sum over i in TopK(g(x)) of g_i(x) * scaling_i * B_i A_i x

The router g is Arrow (Ostapenko et al., 2024, "Towards Modular LLMs by
Building and Reusing a Library of LoRAs"), which needs no training: each
expert's prototype in a module is the top right singular vector of its
B_i A_i, the input direction it changes most. A token's logit for an expert
is |<prototype_i, x>| (a singular vector's sign is arbitrary), the top k
logits go through a softmax and the rest get 0. This is what PEFT's Arrow
does (peft.tuners.lora.arrow), with two differences: experts may differ in
rank and targets, a module routing only among the experts that touch it;
and the prototypes are exact (an SVD of r x r factors), not power-iterated
from a random start, so they do not depend on a seed.

Modules an expert touches outside the MIX's targets and blocks are not
applied when it is mixed; the build warns about them. An expert that
replaces whole modules (PEFT's modules_to_save) cannot be mixed: there is
no per-token sum of replacements, and dropping them would serve the
expert without weights it was trained with.
"""

from __future__ import annotations

import re

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

# Arrow's softmax temperature, PEFT's default.
TEMPERATURE = 1.0


def target_name(module: str) -> str:
    last = module.rsplit(".", 1)[-1]
    return TARGET_NAMES.get(last, last)


def in_blocks(module: str, blocks: str) -> bool:
    """Whether a module's block is in a selection such as "0-3,8-31"; ""
    selects every block. Modules outside any block (lm_head) are never in
    a range."""
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


def in_domain(module: str, targets: list[str], blocks: str) -> bool:
    return target_name(module) in targets and in_blocks(module, blocks)


def arrow_prototype(a, b):
    """The unit top right singular vector of B @ A, in float32 on the CPU,
    without forming B @ A: with A^T = QR, B @ A = (B R^T) Q^T, so it is Q
    times the top right singular vector of the small B R^T."""
    import torch

    a32 = a.detach().to("cpu", torch.float32)  # (r, in)
    b32 = b.detach().to("cpu", torch.float32)  # (out, r)
    q, r = torch.linalg.qr(a32.T)  # (in, r), (r, r)
    _, _, vh = torch.linalg.svd(b32 @ r.T, full_matrices=False)
    v = q @ vh[0]
    return v / v.norm()


def arrow_coefficients(x, prototypes, top_k: int):
    """Per-token expert weights, (tokens, experts) in float32: a softmax over
    each token's top_k logits |x . prototype|, zero elsewhere."""
    import torch

    sim = (x.to(torch.float32) @ prototypes.T).abs()
    k = min(top_k, prototypes.shape[0])
    top, idx = torch.topk(sim, k, dim=1)
    full = torch.full_like(sim, float("-inf")).scatter(1, idx, top)
    return torch.softmax(full / TEMPERATURE, dim=1)


class Route:
    """One mixed module: the experts that touch it and their prototypes."""

    def __init__(self, name: str, module, experts: list[str]):
        import torch

        self.name, self.module, self.experts = name, module, experts
        protos = [
            arrow_prototype(module.lora_A[e].weight, module.lora_B[e].weight)
            for e in experts
        ]
        weight = module.lora_A[experts[0]].weight
        self.prototypes = torch.stack(protos).to(weight.device)

    def delta(self, x, top_k: int):
        """The routed sum of the experts' updates for input x."""
        m = self.module
        dtype = m.lora_A[self.experts[0]].weight.dtype
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        coeff = arrow_coefficients(tokens, self.prototypes, top_k)
        tokens = tokens.to(dtype)
        out = None
        for j, e in enumerate(self.experts):
            y = m.lora_B[e](m.lora_A[e](m.lora_dropout[e](tokens))) * m.scaling[e]
            term = coeff[:, j : j + 1].to(y.dtype) * y
            out = term if out is None else out + term
        return out.reshape(*shape[:-1], out.shape[-1])


def refuse_saved_modules(peft_model, experts: list[str]) -> None:
    """Raises if an expert replaces whole modules (modules_to_save)."""
    import torch

    saved: dict[str, list[str]] = {}
    for name, module in peft_model.named_modules():
        copies = getattr(module, "modules_to_save", None)
        if not isinstance(copies, torch.nn.ModuleDict):
            continue
        for e in experts:
            if e in copies:
                saved.setdefault(e, []).append(name.removeprefix("base_model.model."))
    for e, modules in saved.items():
        raise ValueError(
            f"expert {e!r} replaces whole modules ({', '.join(sorted(modules))}); "
            "token routing mixes LoRA updates only, so serve the expert as its own "
            "model"
        )


class Mix:
    """A MIX of loaded adapters, routed per token in the modules it covers."""

    def __init__(
        self,
        peft_model,
        experts: list[str],
        top_k: int,
        targets: list[str],
        blocks: str = "",
        router: str = "arrow",
    ):
        if router not in ("", "arrow"):
            raise ValueError(f"router {router!r}: only arrow is supported")
        if not experts:
            raise ValueError("a mix needs experts")
        if not 1 <= top_k <= len(experts):
            raise ValueError(f"top_k {top_k} out of range 1..{len(experts)}")
        for part in filter(None, blocks.split(",")):
            lo, _, hi = part.partition("-")
            if not lo.isdigit() or (hi and not hi.isdigit()):
                raise ValueError(f"bad blocks range {part!r}")
        self.experts, self.top_k = list(experts), top_k
        self.targets, self.blocks = list(targets), blocks
        refuse_saved_modules(peft_model, self.experts)
        self.routes: list[Route] = []
        for name, module in peft_model.named_modules():
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None or not hasattr(module, "base_layer"):
                continue
            touching = [e for e in self.experts if e in lora_a]
            if touching and in_domain(name, self.targets, self.blocks):
                self.routes.append(Route(name, module, touching))
        if not self.routes:
            raise ValueError(
                "no expert has a LoRA module in the mix's targets "
                f"({','.join(targets)}) and blocks ({blocks or 'all'})"
            )

    def hooks(self):
        """Forward hooks that add each mixed module's routed update to its
        output. Run them with the adapters themselves disabled."""

        def hook_for(route: Route):
            def hook(_module, args, output):
                return output + route.delta(args[0], self.top_k).to(output.dtype)

            return hook

        return [(r.module, hook_for(r)) for r in self.routes]
