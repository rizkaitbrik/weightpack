"""Reference delta codecs.

These are the readable reference implementations the optimized codecs are
tested against. They work on raw tensor bytes, so BF16 tensors are handled
as uint16 arrays (NumPy has no bfloat16 type).

- ``xor``: lossless. Bitwise XOR of a tensor against the same tensor in the
  parent. Small fine-tuning changes leave most high bits equal, so the
  result is mostly zeros and entropy-codes well.
- ``bit1``: lossy, BitDelta-style. One sign bit per weight plus one scale
  per matrix, initialised to the mean absolute delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_UINT_FOR_ITEMSIZE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _as_uint(a: np.ndarray) -> np.ndarray:
    try:
        dtype = _UINT_FOR_ITEMSIZE[a.dtype.itemsize]
    except KeyError as exc:
        raise TypeError(f"unsupported itemsize {a.dtype.itemsize}") from exc
    return np.ascontiguousarray(a).view(dtype)


def _check_pair(parent: np.ndarray, target: np.ndarray) -> None:
    if parent.shape != target.shape:
        raise ValueError(f"shape mismatch: {parent.shape} vs {target.shape}")
    if parent.dtype.itemsize != target.dtype.itemsize:
        raise ValueError(f"dtype size mismatch: {parent.dtype} vs {target.dtype}")


def xor_encode(parent: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the bitwise XOR of ``target`` against ``parent`` as unsigned ints."""
    _check_pair(parent, target)
    return np.bitwise_xor(_as_uint(parent), _as_uint(target))


def xor_decode(parent: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Restore the target exactly from ``parent`` and an ``xor_encode`` delta."""
    _check_pair(parent, delta)
    return np.bitwise_xor(_as_uint(parent), _as_uint(delta)).view(parent.dtype)


@dataclass(frozen=True)
class Bit1Delta:
    """A 1-bit delta: packed sign bits plus one scale for the whole matrix."""

    shape: tuple[int, ...]
    signs: np.ndarray  # packed with np.packbits, 1 = positive
    scale: float

    @property
    def nbytes(self) -> int:
        return int(self.signs.nbytes) + 4


def bit1_encode(parent: np.ndarray, target: np.ndarray) -> Bit1Delta:
    """Encode ``target - parent`` as sign bits and a mean-absolute scale."""
    _check_pair(parent, target)
    delta = target.astype(np.float32) - parent.astype(np.float32)
    scale = float(np.mean(np.abs(delta))) if delta.size else 0.0
    signs = np.packbits((delta >= 0).ravel())
    return Bit1Delta(shape=tuple(parent.shape), signs=signs, scale=scale)


def bit1_decode(parent: np.ndarray, delta: Bit1Delta) -> np.ndarray:
    """Approximate the target as ``parent + scale * sign``."""
    if tuple(parent.shape) != delta.shape:
        raise ValueError(f"shape mismatch: {parent.shape} vs {delta.shape}")
    n = int(np.prod(delta.shape))
    bits = np.unpackbits(delta.signs, count=n).astype(np.float32)
    sign = (bits * 2.0 - 1.0).reshape(delta.shape)
    out = parent.astype(np.float32) + delta.scale * sign
    return out.astype(parent.dtype)
