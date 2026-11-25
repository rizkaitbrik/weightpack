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

A trained router (ROUTER train, router_train.py) keeps Arrow's form and
starts from it: a token's logit for an expert is |<w_i, x>| + b_i, with a
weight vector and a bias per expert and module, stored in a router file
(see load_router). Arrow is the case w_i = prototype_i, b_i = 0.

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


def router_logits(x, weight, bias=None):
    """Per-token logits, (tokens, experts) in float32: |x . w_i| + b_i."""
    import torch

    sim = (x.to(torch.float32) @ weight.T).abs()
    return sim if bias is None else sim + bias


def top_k_coefficients(logits, top_k: int):
    """A softmax over each token's top_k logits, zero elsewhere. Equal
    logits go to the expert listed first (torch.topk leaves that order
    unspecified), so weightpack-vllm picks the same experts."""
    import torch

    k = min(top_k, logits.shape[1])
    idx = logits.argsort(dim=1, descending=True, stable=True)[:, :k]
    top = logits.gather(1, idx)
    full = torch.full_like(logits, float("-inf")).scatter(1, idx, top)
    return torch.softmax(full / TEMPERATURE, dim=1)


def arrow_coefficients(x, prototypes, top_k: int):
    """Per-token expert weights, (tokens, experts) in float32: Arrow's
    softmax over each token's top_k logits |x . prototype|, zero elsewhere."""
    return top_k_coefficients(router_logits(x, prototypes), top_k)


# A router file holds, per routed module (in the base's naming, such as
# model.layers.0.mlp.up_proj), these two float32 tensors, one row per expert
# of the mix in mix order. A row of an expert that does not touch the module
# is zero and unused.
ROUTER_WEIGHT = ".router.weight"  # (experts, in_features)
ROUTER_BIAS = ".router.bias"  # (experts,)
PEFT_PREFIX = "base_model.model."


def load_router(path: str) -> dict:
    """A router file's tensors, by name, on the CPU."""
    from safetensors.torch import load_file

    return load_file(path)


class Route:
    """One mixed module: the experts that touch it and their router rows,
    Arrow's prototypes unless rows (weight (n, in), bias (n)) are given."""

    def __init__(self, name: str, module, experts: list[str], rows=None):
        import torch

        self.name, self.module, self.experts = name, module, experts
        device = module.lora_A[experts[0]].weight.device
        if rows is None:
            protos = [
                arrow_prototype(module.lora_A[e].weight, module.lora_B[e].weight)
                for e in experts
            ]
            rows = (torch.stack(protos), torch.zeros(len(experts)))
        self.weight = rows[0].to(device, torch.float32)
        self.bias = rows[1].to(device, torch.float32)
        # Set while a router trains: see coefficients.
        self.training = False

    def coefficients(self, tokens, top_k: int):
        """Each token's expert weights. While training, the values are the
        served ones and the gradient is the full softmax's (a straight-
        through estimator), so experts the top k leave out still learn,
        including with top_k 1, whose served weight is a constant 1."""
        import torch

        logits = router_logits(tokens, self.weight, self.bias)
        served = top_k_coefficients(logits, top_k)
        if not self.training:
            return served
        soft = torch.softmax(logits, dim=1)
        return served.detach() + soft - soft.detach()

    def delta(self, x, top_k: int):
        """The routed sum of the experts' updates for input x."""
        m = self.module
        dtype = m.lora_A[self.experts[0]].weight.dtype
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        coeff = self.coefficients(tokens, top_k)
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
        router_file: str = "",
    ):
        if router not in ("", "arrow", "trained"):
            raise ValueError(f"router {router!r}: want arrow or trained")
        if router == "trained" and not router_file:
            raise ValueError("a trained router needs its router file")
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
        trained = load_router(router_file) if router == "trained" else None
        self.routes: list[Route] = []
        for name, module in peft_model.named_modules():
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None or not hasattr(module, "base_layer"):
                continue
            touching = [e for e in self.experts if e in lora_a]
            if touching and in_domain(name, self.targets, self.blocks):
                rows = None
                if trained is not None:
                    rows = self._rows(trained, name, module, touching)
                self.routes.append(Route(name, module, touching, rows))
        if not self.routes:
            raise ValueError(
                "no expert has a LoRA module in the mix's targets "
                f"({','.join(targets)}) and blocks ({blocks or 'all'})"
            )

    def _rows(self, trained: dict, name: str, module, touching: list[str]):
        """A route's rows of a trained router: its touching experts'."""
        key = name.removeprefix(PEFT_PREFIX)
        weight, bias = trained.get(key + ROUTER_WEIGHT), trained.get(key + ROUTER_BIAS)
        if weight is None or bias is None:
            raise ValueError(f"the router file has no weights for {key}")
        n, in_features = len(self.experts), module.lora_A[touching[0]].weight.shape[1]
        if tuple(weight.shape) != (n, in_features) or tuple(bias.shape) != (n,):
            raise ValueError(
                f"{key}: router weight {tuple(weight.shape)} and bias "
                f"{tuple(bias.shape)}, want ({n}, {in_features}) and ({n},)"
            )
        rows = [self.experts.index(e) for e in touching]
        return weight[rows], bias[rows]

    def router_tensors(self) -> dict:
        """This mix's router as a router file's tensors, on the CPU."""
        import torch

        out = {}
        for r in self.routes:
            key = r.name.removeprefix(PEFT_PREFIX)
            weight = torch.zeros(len(self.experts), r.weight.shape[1])
            bias = torch.zeros(len(self.experts))
            for j, e in enumerate(r.experts):
                i = self.experts.index(e)
                weight[i] = r.weight[j].detach().float().cpu()
                bias[i] = r.bias[j].detach().float().cpu()
            out[key + ROUTER_WEIGHT] = weight.contiguous()
            out[key + ROUTER_BIAS] = bias.contiguous()
        return out

    def hooks(self):
        """Forward hooks that add each mixed module's routed update to its
        output. Run them with the adapters themselves disabled."""

        def hook_for(route: Route):
            def hook(_module, args, output):
                return output + route.delta(args[0], self.top_k).to(output.dtype)

            return hook

        return [(r.module, hook_for(r)) for r in self.routes]
