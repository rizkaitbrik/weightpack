"""Connects routing.py and deltas.py to vLLM's LoRA path. Written against
vLLM 0.34; tests/vllm_e2e.sh runs it in vLLM 0.29's CPU image.

For token routing, four patches, all small:

- LoRAModel.from_local_checkpoint: an adapter directory with a
  weightpack_mix.json is a MIX. Each expert is loaded with vLLM's own loader
  from the directory the file names, and the experts are stacked into one
  adapter (routing.stack_experts) over the modules the MIX covers.
- LoRAModelManager.__init__: every linear LoRA layer gets a Routing, before
  vLLM captures CUDA graphs.
- LoRAModelManager.activate_adapter: an adapter put in a slot sets that
  slot's routing, from the slot's own weights, or clears it.
- add_lora_linear of the punica wrapper class the manager uses: for a layer
  with a Routing, the shrink buffer is gated per token before the expand
  (routing.gated_lora_linear).

For bit1 fine-tunes (deltas.py), with WEIGHTPACK_DELTA_SLOTS=1, which wpkd
sets when it starts vLLM to serve them:

- LoRAModel.from_local_checkpoint: a directory with a weightpack_delta.json
  is a fine-tune; it loads as an adapter with no LoRA weights, carrying its
  deltas, placed in the model's parameters.
- LoRAModelManager.__init__: every linear layer's weight and bias, every
  embedding's and every RMSNorm's weight gets delta buffers for each adapter
  slot, before vLLM captures CUDA graphs.
- LoRAModelManager.activate_adapter: an adapter put in a slot writes its
  deltas into that slot's buffers, or clears them; _create_merged_loras_inplace
  skips a fine-tune, which has no LoRA weights to pack.
- UnquantizedLinearMethod.apply, UnquantizedEmbeddingMethod.embedding and
  .apply (the output head) and RMSNorm.forward: each token's delta, by its
  adapter slot, is added to the base's output.
"""

from __future__ import annotations

import logging
import os

from . import deltas, routing

# A child of vLLM's logger, so vLLM's logging prints it.
logger = logging.getLogger("vllm.weightpack")

# The most experts a mix may have; prototype buffers grow with it.
MAX_EXPERTS = int(os.environ.get("WEIGHTPACK_MAX_EXPERTS", "8"))

# Delta buffers cost each adapter slot 1/16 of the base's matrices, so they
# are made only when asked for.
DELTA_SLOTS = os.environ.get("WEIGHTPACK_DELTA_SLOTS") == "1"

_routings: dict[int, routing.Routing] = {}  # id(layer.lora_a_stacked[0]) -> routing


def apply() -> None:
    import torch
    from vllm.lora import lora_model, model_manager
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.lora.lora_weights import LoRALayerWeights
    from vllm.lora.peft_helper import PEFTHelper

    LoRAModel = lora_model.LoRAModel
    Manager = model_manager.LoRAModelManager
    if DELTA_SLOTS:
        # First: should it fail, no fine-tune loads, rather than one that
        # loads and is served as the base.
        _patch_forward()

    load = LoRAModel.from_local_checkpoint.__func__

    def from_local_checkpoint(cls, lora_dir, expected_lora_modules, peft_helper, **kw):
        if deltas.is_delta_dir(lora_dir):
            return _delta_model(cls, lora_dir, kw["lora_model_id"])
        mix = routing.load_mix(lora_dir)
        if mix is None:
            return load(cls, lora_dir, expected_lora_modules, peft_helper, **kw)
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() > 1:
            raise ValueError("token routing needs --tensor-parallel-size 1 for now")
        dtype = kw.get("dtype") or torch.float32
        device = kw.pop("device", "cpu")
        experts, ranks = [], []
        for spec in mix["experts"]:
            helper = PEFTHelper.from_local_dir(
                spec["path"], peft_helper.vllm_max_position_embeddings
            )
            model = load(
                cls, spec["path"], expected_lora_modules, helper, device="cpu", **kw
            )
            # vLLM 0.30 and later load modules_to_save; 0.29 ignores them.
            if getattr(model, "modules_to_save", None):
                saved = ", ".join(model.modules_to_save)
                raise ValueError(
                    f"expert {spec['id']!r} replaces whole modules ({saved}); "
                    "token routing mixes LoRA updates only"
                )
            experts.append(
                {n: (w.lora_a, w.lora_b, w.scaling) for n, w in model.loras.items()}
            )
            ranks.append(model.rank)

        def keep(name):
            return routing.in_domain(name, mix["targets"], mix.get("blocks") or "")

        total = sum(ranks)
        stacked = routing.stack_experts(experts, ranks, keep, dtype)
        if not stacked:
            raise ValueError(
                "no expert has a LoRA module in the mix's targets and blocks"
            )
        loras = {
            name: LoRALayerWeights(
                name, total, total, a.to(device), b.to(device), scaling=1.0
            )
            for name, (a, b) in stacked.items()
        }
        out = cls(kw["lora_model_id"], total, loras)
        out.weightpack_mix = {"ranks": ranks, "top_k": int(mix["top_k"])}
        if mix.get("router") == "trained":
            out.weightpack_mix["router"] = routing.load_router(mix["router_file"])
        logger.info(
            "weightpack: mix of %d experts (%s), rank %d over %d modules, top_k %d, "
            "%s router",
            len(ranks),
            ", ".join(e["id"] for e in mix["experts"]),
            total,
            len(loras),
            mix["top_k"],
            mix.get("router", "arrow"),
        )
        return out

    LoRAModel.from_local_checkpoint = classmethod(from_local_checkpoint)

    init = Manager.__init__

    def manager_init(self, *args, **kw):
        init(self, *args, **kw)
        if DELTA_SLOTS:
            _setup_deltas(self)
        made = 0
        for module in self.modules.values():
            if not isinstance(module, BaseLinearLayerWithLoRA) or not hasattr(
                module, "lora_a_stacked"
            ):
                continue
            # The platform's punica wrapper class (GPU, CPU, ...) is only
            # known now: the manager gives each layer one (several, one per
            # tower, in multimodal models).
            _wrap_lora_linear(type(module.punica_wrapper))
            if getattr(module, "tp_size", 1) > 1:
                logger.warning(
                    "weightpack: token routing needs tensor parallel size 1; "
                    "mixes are off"
                )
                return
            a = module.lora_a_stacked[0]
            r = routing.Routing(
                n_slices=len(module.lora_a_stacked),
                slots=a.shape[0],
                max_rank=module.lora_b_stacked[0].shape[-1],
                in_features=a.shape[-1],
                max_experts=MAX_EXPERTS,
                device=a.device,
            )
            module._weightpack_routing = r
            _routings[id(a)] = r
            made += 1
        logger.info("weightpack: token routing ready in %d LoRA layers", made)

    Manager.__init__ = manager_init

    activate = Manager.activate_adapter

    def activate_adapter(self, lora_id):
        result = activate(self, lora_id)
        if not result:
            return result
        model = self._registered_adapters[lora_id]
        index = self.lora_index_to_id.index(model.id)
        _activate_delta(index, getattr(model, "weightpack_delta", None))
        mix = getattr(model, "weightpack_mix", None)
        # A packed layer (qkv_proj, gate_up_proj) lists the modules its
        # slices hold, in slice order; a trained router has rows per module.
        packed = getattr(self, "packed_modules", {})
        for name, module in self.modules.items():
            r = getattr(module, "_weightpack_routing", None)
            if r is None:
                continue
            if mix is None:
                r.clear(index)
                continue
            rows = None
            if "router" in mix:
                parts = packed.get(name, [name])
                if len(parts) != len(module.lora_a_stacked):
                    raise ValueError(
                        f"{name}: {len(module.lora_a_stacked)} LoRA slices, "
                        f"{len(parts)} modules"
                    )
                rows = [mix["router"].get(p) for p in parts]
            r.set_slot(
                index,
                mix["ranks"],
                mix["top_k"],
                [a[index, 0] for a in module.lora_a_stacked],
                [b[index, 0] for b in module.lora_b_stacked],
                rows,
            )
        return result

    Manager.activate_adapter = activate_adapter

    merge = Manager._create_merged_loras_inplace

    def create_merged_loras_inplace(self, lora_model):
        # A bit1 fine-tune has no LoRA weights to pack.
        if getattr(lora_model, "weightpack_delta", None) is not None:
            return
        return merge(self, lora_model)

    Manager._create_merged_loras_inplace = create_merged_loras_inplace


def _wrap_lora_linear(cls) -> None:
    """Gates add_lora_linear of the punica wrapper class in use, once."""
    if getattr(cls, "_weightpack_gated", False):
        return
    original = cls.add_lora_linear

    def add_lora_linear(
        self,
        y,
        x,
        lora_a_stacked,
        lora_b_stacked,
        scale,
        output_slices,
        *,
        buffer=None,
        **kwargs,
    ):
        r = _routings.get(id(lora_a_stacked[0]))
        if r is None:
            return original(
                self,
                y,
                x,
                lora_a_stacked,
                lora_b_stacked,
                scale,
                output_slices,
                buffer=buffer,
                **kwargs,
            )
        return routing.gated_lora_linear(
            self,
            r,
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            scale,
            output_slices,
            **kwargs,
        )

    cls.add_lora_linear = add_lora_linear
    cls._weightpack_gated = True
    logger.info("weightpack: gating LoRA in %s", cls.__name__)


# bit1 fine-tunes ------------------------------------------------------------

# The language model's punica wrapper: each token's adapter slot
# (_token_lora_indices) and each sampled row's (_sampler_indices).
_punica = None
# vLLM parameter name -> its delta buffers and the module that owns it.
_targets: dict[str, tuple[deltas.ParamSlots, object]] = {}
# A packed module's parts (q_proj of qkv_proj, ...) -> (packed, slice).
_parts: dict[str, tuple[str, int]] = {}


def _clean(name: str) -> str:
    """A parameter's name as the checkpoint has it, without the LoRA
    wrappers' base_layer."""
    return name.replace(".base_layer", "")


def _setup_deltas(manager) -> None:
    """Delta buffers, one per adapter slot, beside every parameter a bit1
    fine-tune may change and the plugin can apply per token."""
    global _punica
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    model = manager.model
    wrappers = [m.punica_wrapper for m in manager.modules.values()]
    if not wrappers:
        logger.warning("weightpack: no LoRA layers; bit1 deltas are off")
        return
    slots = manager.lora_slots
    made: dict[int, deltas.ParamSlots] = {}
    for mname, module in model.named_modules(remove_duplicate=False):
        if getattr(module, "tp_size", 1) > 1:
            logger.warning("weightpack: bit1 deltas need tensor parallel size 1")
            _targets.clear()
            return
        kinds = []
        if isinstance(module, LinearBase):
            # By the layer's sizes: on the CPU, the weight is repacked.
            out = getattr(module, "output_size_per_partition", module.output_size)
            inp = getattr(module, "input_size_per_partition", module.input_size)
            kinds = [("weight", "_wp_weight", (out, inp)), ("bias", "_wp_bias", (out,))]
        elif isinstance(module, VocabParallelEmbedding) or (
            isinstance(module, RMSNorm) and module.has_weight
        ):
            kinds = [("weight", "_wp_weight", None)]
        for pname, attr, shape in kinds:
            p = module._parameters.get(pname)
            if p is None or not p.dtype.is_floating_point:
                continue
            if id(p) not in made:
                shape = tuple(p.shape) if shape is None else shape
                made[id(p)] = deltas.ParamSlots(shape, slots, p.device)
            setattr(module, attr, made[id(p)])
            _targets[_clean(f"{mname}.{pname}" if mname else pname)] = (
                made[id(p)],
                module,
            )
    for packed, parts in getattr(manager, "packed_modules_mapping", {}).items():
        for i, part in enumerate(parts):
            _parts[part] = (packed, i)
    _punica = wrappers[0]
    size = 0
    for ps in made.values():
        bufs = (ps.signs, ps.row_scale) if ps.matrix else (ps.vector,)
        size += sum(b.numel() * b.element_size() for b in bufs)
    logger.info(
        "weightpack: bit1 deltas ready in %d parameters, %.2f GB for %d slots",
        len(made),
        size / 1e9,
        slots,
    )


def _place(name: str):
    """Where a checkpoint tensor's delta goes: the vLLM parameter and the
    rows (or elements) it covers there."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    if name in _targets:
        ps, module = _targets[name]
        if isinstance(module, VocabParallelEmbedding) and ps.matrix:
            return name, deltas.Placed(name, 0, module.org_vocab_size)
        return name, deltas.Placed(name, 0, ps.shape[0])
    module, _, leaf = name.rpartition(".")
    parent, _, part = module.rpartition(".")
    if part in _parts:
        packed, i = _parts[part]
        vname = f"{parent}.{packed}.{leaf}" if parent else f"{packed}.{leaf}"
        if vname in _targets:
            sizes = list(_targets[vname][1].output_sizes)
            return vname, deltas.Placed(vname, sum(sizes[:i]), sizes[i])
    raise ValueError(
        f"the fine-tune changes {name}, which this vLLM cannot change per "
        "token; wpk squash the image to serve it dense"
    )


def _delta_model(cls, lora_dir: str, lora_id: int):
    """A bit1 fine-tune as an adapter with no LoRA weights, carrying its
    deltas by vLLM parameter."""
    if not _targets:
        raise ValueError(
            "this vLLM has no delta slots for bit1 fine-tunes; start it with "
            "WEIGHTPACK_DELTA_SLOTS=1 (wpkd does), at tensor parallel size 1"
        )
    placed: dict[str, list] = {}
    seen: dict[tuple, object] = {}
    for name, delta in sorted(deltas.load_delta(lora_dir).items()):
        vname, where = _place(name)
        key = (vname, where.offset)
        if key in seen:
            # A tied tensor (embeddings and output head) given twice.
            if not _same(seen[key], delta):
                raise ValueError(f"{name} changes a tied tensor differently")
            continue
        seen[key] = delta
        ps = _targets[vname][0]
        if ps.matrix:
            delta.rows(where.rows, ps.cols)  # checks the sign count
        placed.setdefault(vname, []).append((where, delta))
    out = cls(lora_id, 1, {})
    out.weightpack_delta = placed
    logger.info(
        "weightpack: bit1 fine-tune %s over %d parameters", lora_dir, len(placed)
    )
    return out


def _same(a, b) -> bool:
    import numpy as np

    if isinstance(a, deltas.Matrix) and isinstance(b, deltas.Matrix):
        return a.scale == b.scale and np.array_equal(a.signs, b.signs)
    if isinstance(a, deltas.Vector) and isinstance(b, deltas.Vector):
        return np.array_equal(a.delta, b.delta)
    return False


def _activate_delta(index: int, placed) -> None:
    """Writes the deltas of the adapter put in slot index into the slot's
    buffers, or clears them for any other adapter."""
    if not _targets:
        return
    for ps in {id(ps): ps for ps, _ in _targets.values()}.values():
        ps.clear(index)
    for vname, items in (placed or {}).items():
        ps = _targets[vname][0]
        for where, delta in items:
            ps.put(index, where, delta)


def _patch_forward() -> None:
    """Adds each token's delta to the outputs of the layers that hold delta
    buffers. A layer without them is left as it is."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    _register_ops()
    linear_apply = UnquantizedLinearMethod.apply

    def apply_linear(self, layer, x, bias=None):
        y = linear_apply(self, layer, x, bias)
        ps = getattr(layer, "_wp_weight", None)
        if ps is None or _punica is None:
            return y
        xs = x.reshape(-1, x.shape[-1])
        slots = _punica._token_lora_indices[: xs.shape[0]]
        d = _product(xs, slots, ps)
        pb = getattr(layer, "_wp_bias", None)
        if pb is not None and bias is not None:
            d = d + deltas.vector_delta(slots, pb)
        return y + d.to(y.dtype).view(y.shape)

    UnquantizedLinearMethod.apply = apply_linear

    embed = UnquantizedEmbeddingMethod.embedding

    def embedding(self, layer, input_):
        e = embed(self, layer, input_)
        ps = getattr(layer, "_wp_weight", None)
        if ps is None or _punica is None:
            return e
        ids = input_.reshape(-1)
        slots = _punica._token_lora_indices[: ids.shape[0]]
        d = deltas.rows_delta(ids, slots, ps, e.dtype)
        return e + d.to(e.dtype).view(e.shape)

    UnquantizedEmbeddingMethod.embedding = embedding

    head_apply = UnquantizedEmbeddingMethod.apply

    def apply_head(self, layer, x, bias=None):
        y = head_apply(self, layer, x, bias)
        ps = getattr(layer, "_wp_weight", None)
        if ps is None or _punica is None:
            return y
        xs = x.reshape(-1, x.shape[-1])
        # The output head sees the rows to sample, one per sequence.
        slots = _punica._sampler_indices[: xs.shape[0]]
        d = _product(xs, slots, ps)
        return y + d[:, : y.shape[-1]].to(y.dtype).view(y.shape)

    UnquantizedEmbeddingMethod.apply = apply_head

    norm_forward = RMSNorm.forward

    def rms_forward(self, x, residual=None):
        out = norm_forward(self, x, residual)
        ps = getattr(self, "_wp_weight", None)
        if ps is None or _punica is None:
            return out
        y, res = (out, None) if residual is None else out
        base = x if residual is None else res
        slots = _punica._token_lora_indices[: base.shape[0]]
        d = deltas.vector_delta(slots, ps)
        shape = (base.shape[0],) + (1,) * (base.dim() - 2) + (ps.shape[0],)
        n = deltas.rms_normalized(
            base, self.variance_epsilon, self.variance_size_override
        )
        y = y + (n * d.view(shape)).to(y.dtype)
        return y if residual is None else (y, res)

    RMSNorm.forward = rms_forward
    logger.info("weightpack: bit1 deltas apply per token")


def _product(x, slots, ps: deltas.ParamSlots):
    import torch

    return torch.ops.weightpack.bit1_linear(x, slots, ps.signs, ps.row_scale, ps.active)


_ops_registered = False


def _register_ops() -> None:
    """weightpack::bit1_linear, opaque to torch.compile: the Triton kernel
    on CUDA, deltas.linear_delta elsewhere."""
    global _ops_registered
    if _ops_registered:
        return
    import torch

    @torch.library.custom_op(
        "weightpack::bit1_linear",
        mutates_args=(),
        schema="(Tensor x, Tensor slots, Tensor signs, Tensor row_scale, "
        "Tensor active) -> Tensor",
    )
    def bit1_linear(x, slots, signs, row_scale, active):
        if x.is_cuda:
            from . import bit1_kernel

            if bit1_kernel.available():
                return bit1_kernel.linear(x, slots, signs, row_scale, active)
        return deltas.linear_delta(
            x, slots, signs, row_scale, active, graph_safe=x.is_cuda
        )

    @bit1_linear.register_fake
    def _(x, slots, signs, row_scale, active):
        return x.new_empty((x.shape[0], signs.shape[1]), dtype=torch.float32)

    _ops_registered = True
