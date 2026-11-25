"""Bake LoRA layers into their base tensors.

For each base tensor ``W`` that some layer modifies, with layers applied in
chain order (bottom first)::

    W' = round(W + sum_i scaling_i * (B_i @ A_i))

Everything is widened to float32, each delta is computed as
``(B @ A) * scaling`` and added in order, and the sum is rounded once, to
nearest even, to ``W``'s dtype. That is what PEFT's ``merge_and_unload``
does for a float32 adapter (its default, even over a BF16 model), so the
results agree but for the rare last-bit flip from float32 summation order.
For a BF16 or F16 adapter PEFT rounds the delta to the adapter's dtype
before adding it; this rounds once, which is at least as accurate.

Tensors no layer touches are not written at all: the daemon keeps their
blobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from weightpack_worker.delta import Parent
from weightpack_worker.safetensors_io import (
    FLOAT_DTYPES,
    TensorInfo,
    from_float32,
    load_raw,
    read_header,
    storage_dtype,
    to_float32,
    write_tensor,
)

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


@dataclass
class LoRALayer:
    """A LoRA layer: its scaling and the paths of its lora.v1 blobs."""

    id: str
    scaling: float
    blobs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Delta:
    """A LoRA module: add scaling * B @ A to its base tensor."""

    layer: str
    scaling: float
    path: str
    a: TensorInfo
    b: TensorInfo


@dataclass(frozen=True)
class _Replace:
    """A modules_to_save copy: the base tensor becomes this tensor."""

    layer: str
    path: str
    info: TensorInfo


def _ops(
    layers: list[LoRALayer], base: dict[str, tuple[str, TensorInfo]]
) -> dict[str, list[_Delta | _Replace]]:
    """Collect each layer's work by the base tensor it touches, in chain
    order, checking every shape against the base before any math runs.

    A blob holding one module's lora_A and lora_B is a LoRA delta; a blob
    holding one tensor named after a base tensor is a modules_to_save copy
    that replaces it.
    """
    out: dict[str, list[_Delta | _Replace]] = {}
    for layer in layers:
        for path in layer.blobs:
            header = read_header(path)
            single = len(header) == 1 and not next(iter(header)).endswith(
                (_A_SUFFIX, _B_SUFFIX)
            )
            if single:
                ((name, info),) = header.items()
                want = base.get(name)
                if want is None:
                    raise ValueError(
                        f"layer {layer.id}: {path}: {name} replaces no base tensor"
                    )
                if (info.dtype, info.shape) != (want[1].dtype, want[1].shape):
                    raise ValueError(
                        f"layer {layer.id}: saved {name} is "
                        f"{info.dtype}{list(info.shape)}, the base has "
                        f"{want[1].dtype}{list(want[1].shape)}"
                    )
                out.setdefault(name, []).append(_Replace(layer.id, path, info))
                continue
            modules = {n[: -len(_A_SUFFIX)] for n in header if n.endswith(_A_SUFFIX)}
            if len(header) != 2 or len(modules) != 1:
                raise ValueError(
                    f"layer {layer.id}: {path} must hold one module's lora_A and "
                    f"lora_B, found {sorted(header)}"
                )
            module = modules.pop()
            a = header.get(module + _A_SUFFIX)
            b = header.get(module + _B_SUFFIX)
            if a is None or b is None:
                raise ValueError(f"layer {layer.id}: {path} lacks lora_B for {module}")
            target = module + ".weight"
            if target not in base:
                raise ValueError(f"layer {layer.id}: the base has no tensor {target}")
            w = base[target][1]
            if (
                w.dtype not in FLOAT_DTYPES
                or a.dtype not in FLOAT_DTYPES
                or b.dtype not in FLOAT_DTYPES
            ):
                raise ValueError(f"layer {layer.id}: {module}: tensors must be float")
            if len(w.shape) != 2 or len(a.shape) != 2 or len(b.shape) != 2:
                raise ValueError(
                    f"layer {layer.id}: {module}: all of W, A and B must be matrices"
                )
            out_f, in_f = w.shape
            if a.shape[1] != in_f or b.shape[0] != out_f or a.shape[0] != b.shape[1]:
                raise ValueError(
                    f"layer {layer.id}: {module}: lora_A {list(a.shape)} and lora_B "
                    f"{list(b.shape)} do not fit {target} {list(w.shape)}"
                )
            out.setdefault(target, []).append(
                _Delta(layer.id, layer.scaling, path, a, b)
            )
    return out


def squash(
    base_blobs: list[str],
    layers: list[LoRALayer],
    out_dir: str | Path,
    xor_layers: list[list[str]] = (),
) -> list[tuple[str, str]]:
    """Merge XOR layers, then ``layers``, into the base tensors they modify.

    XOR layers apply first, exactly, bit for bit; LoRA deltas then add in
    float32. Writes one safetensors file per merged tensor into ``out_dir``
    and returns ``(tensor name, file name)`` pairs, sorted by tensor name.
    """
    out_dir = Path(out_dir)
    parent = Parent(base_blobs, xor_layers)
    ops = _ops(layers, parent.base)
    merged: list[tuple[str, str]] = []
    for i, target in enumerate(sorted(set(ops) | parent.touched())):
        info = parent.base[target][1]
        raw = parent.bits(target).view(storage_dtype(info.dtype))
        w = None  # float32 working copy, made when the first delta needs it
        for op in ops.get(target, []):
            if isinstance(op, _Replace):
                raw, w = np.asarray(load_raw(op.path, op.info)), None
                continue
            if w is None:
                w = to_float32(info.dtype, raw)
            a = to_float32(op.a.dtype, load_raw(op.path, op.a))
            b = to_float32(op.b.dtype, load_raw(op.path, op.b))
            delta = b @ a
            delta *= np.float32(op.scaling)
            w += delta
        if w is not None:
            raw = from_float32(info.dtype, w)
        file = f"{i:06d}.safetensors"
        write_tensor(out_dir / file, target, info.dtype, raw)
        merged.append((target, file))
    return merged
