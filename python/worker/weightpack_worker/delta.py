"""Lossless XOR deltas between a full fine-tune and its parent.

A ``delta.xor`` layer stores, for each tensor the fine-tune changed, the
fine-tune's bits XORed with the parent's. XOR layers sit only on the base or
on other XOR layers, so the parent's bits are exact (no float math) and the
fine-tune comes back bit for bit: ``target = parent ^ delta``.

Tensors equal to the parent's are not stored at all. Fine-tuning moves
weights by little, so most high bits of a delta are zero, which is what the
daemon's entropy coder (zstd) then squeezes out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from weightpack_worker.safetensors_io import (
    TensorInfo,
    index_tensors,
    load_raw,
    read_header,
    storage_dtype,
    write_tensor,
)

_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def bits(path: str, info: TensorInfo) -> np.ndarray:
    """A tensor's raw bits as unsigned integers of its item size."""
    raw = np.asarray(load_raw(path, info))
    return raw.view(_UINT[raw.dtype.itemsize])


class Parent:
    """The exact bits of a chain's tensors: the base with XOR layers applied
    in order, bottom first."""

    def __init__(self, base_blobs: list[str], xor_layers: list[list[str]] = ()):
        self.base = index_tensors(base_blobs)
        self.deltas: dict[str, list[tuple[str, TensorInfo]]] = {}
        for layer in xor_layers:
            for path in layer:
                header = read_header(path)
                if len(header) != 1:
                    raise ValueError(f"XOR blob {path} must hold one tensor")
                ((name, info),) = header.items()
                base = self.base.get(name)
                if base is None:
                    raise ValueError(f"XOR blob {path}: the base has no tensor {name}")
                if (info.dtype, info.shape) != (base[1].dtype, base[1].shape):
                    raise ValueError(
                        f"XOR blob {path}: {name} is {info.dtype}{list(info.shape)}, "
                        f"the base has {base[1].dtype}{list(base[1].shape)}"
                    )
                self.deltas.setdefault(name, []).append((path, info))

    def touched(self) -> set[str]:
        """Tensors some XOR layer changes."""
        return set(self.deltas)

    def bits(self, name: str) -> np.ndarray:
        """A new array holding the tensor's bits in the chain."""
        out = np.array(bits(*self.base[name]))
        for path, info in self.deltas.get(name, []):
            np.bitwise_xor(out, bits(path, info), out=out)
        return out


def fine_tune(
    parent: Parent, target_files: list[str]
) -> dict[str, tuple[str, TensorInfo]]:
    """A fine-tune's tensors, checked to be exactly the base's names, dtypes
    and shapes."""
    target = index_tensors(target_files)
    missing = sorted(set(parent.base) - set(target))
    extra = sorted(set(target) - set(parent.base))
    if missing or extra:
        raise ValueError(
            "the fine-tune's tensors differ from the base's: "
            f"missing {missing[:3]}{'…' if len(missing) > 3 else ''}, "
            f"extra {extra[:3]}{'…' if len(extra) > 3 else ''}"
        )
    for name, (_, info) in target.items():
        want = parent.base[name][1]
        if (info.dtype, info.shape) != (want.dtype, want.shape):
            raise ValueError(
                f"{name} is {info.dtype}{list(info.shape)} in the fine-tune, "
                f"{want.dtype}{list(want.shape)} in the base"
            )
    return target


def encode_xor(
    base_blobs: list[str],
    parents: list[list[str]],
    target_files: list[str],
    out_dir: str | Path,
) -> tuple[list[tuple[str, str]], int]:
    """Encode a fine-tune as XOR deltas over its parent.

    Writes one safetensors file per changed tensor into ``out_dir`` and
    returns ``(tensor name, file name)`` pairs, sorted by name, and the
    number of tensors equal to the parent's. The fine-tune must have exactly
    the base's tensor names, dtypes and shapes.
    """
    out_dir = Path(out_dir)
    parent = Parent(base_blobs, parents)
    target = fine_tune(parent, target_files)

    encoded: list[tuple[str, str]] = []
    unchanged = 0
    for i, name in enumerate(sorted(target)):
        path, info = target[name]
        new = bits(path, info)
        old = parent.bits(name)
        if np.array_equal(new, old):
            unchanged += 1
            continue
        np.bitwise_xor(old, new, out=old)
        file = f"{i:06d}.safetensors"
        write_tensor(
            out_dir / file, name, info.dtype, old.view(storage_dtype(info.dtype))
        )
        encoded.append((name, file))
    return encoded, unchanged
