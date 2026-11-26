"""Connects routing.py to vLLM's LoRA path. Written against vLLM 0.34;
tests/vllm_e2e.sh runs it in vLLM 0.29's CPU image.

Four patches, all small:

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
"""

from __future__ import annotations

import logging
import os

from . import routing

# A child of vLLM's logger, so vLLM's logging prints it.
logger = logging.getLogger("vllm.weightpack")

# The most experts a mix may have; prototype buffers grow with it.
MAX_EXPERTS = int(os.environ.get("WEIGHTPACK_MAX_EXPERTS", "8"))


def apply() -> None:
    _key_compile_cache()

    import torch
    from vllm.lora import lora_model, model_manager
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.lora.lora_weights import LoRALayerWeights
    from vllm.lora.peft_helper import PEFTHelper

    LoRAModel = lora_model.LoRAModel
    Manager = model_manager.LoRAModelManager

    load = LoRAModel.from_local_checkpoint.__func__

    def from_local_checkpoint(cls, lora_dir, expected_lora_modules, peft_helper, **kw):
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
        made = 0
        for module in self.modules.values():
            if not isinstance(module, BaseLinearLayerWithLoRA) or not hasattr(
                module, "lora_a_stacked"
            ):
                continue
            if getattr(module, "tp_size", 1) > 1:
                logger.warning(
                    "weightpack: token routing needs tensor parallel size 1; "
                    "mixes are off"
                )
                return
            # The platform's punica wrapper class (GPU, CPU, ...) is only
            # known now: the manager gives each layer one (several, one per
            # tower, in multimodal models).
            _wrap_lora_linear(type(module.punica_wrapper))
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
            routing.attach(a, r)
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


def _wrap_lora_linear(cls) -> None:
    """Gates add_lora_linear of the punica wrapper class in use, once."""
    if getattr(cls, "_weightpack_gated", False):
        return

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
        # Every linear LoRA layer has a routing (manager_init), which the
        # gate op finds by lora_a_stacked[0] when it runs.
        return routing.gated_lora_linear(
            self,
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


def _key_compile_cache() -> None:
    """vLLM's AOT compile cache keys an entry by vLLM's config and
    environment, not by the code that was traced: an entry written without
    this plugin would load with it, the patches missing from the compiled
    model. So the plugin's version and settings join the key. (vLLM checks
    the traced files for changes when it loads an entry, which covers this
    plugin's own code once it is traced.)"""
    try:
        from vllm.compilation import caching
    except ImportError:
        return
    factors = getattr(caching, "aot_compile_hash_factors", None)
    if factors is None or getattr(factors, "_weightpack", False):
        return
    from . import __version__

    plugin = f"weightpack-vllm {__version__} max_experts={MAX_EXPERTS}"

    def with_plugin(vllm_config):
        return [*factors(vllm_config), plugin]

    with_plugin._weightpack = True
    caching.aot_compile_hash_factors = with_plugin
