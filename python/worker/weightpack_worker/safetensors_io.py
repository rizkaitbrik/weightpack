"""Read and write safetensors files with NumPy, BF16 included.

NumPy has no bfloat16 type, so BF16 tensors are read as raw ``uint16``
arrays and converted explicitly with :func:`to_float32` and
:func:`from_float32`, which rounds to nearest even as PyTorch does.

:func:`write_tensor` writes the same canonical layout as the Go side
(``libs/go/safetensors``): one tensor, a compact JSON header with sorted
keys, padded with spaces to a multiple of 8 bytes. The daemon re-encodes
whatever a worker writes anyway, so this is a convenience, not a contract.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_HEADER_SIZE = 100_000_000

# Raw storage dtype of each safetensors dtype, little-endian. BF16 is kept as
# its 16-bit pattern.
_STORAGE = {
    "BOOL": np.dtype("?"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "F64": np.dtype("<f8"),
}

FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})


@dataclass(frozen=True)
class TensorInfo:
    """One header entry; ``offset`` is absolute within the file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int


def read_header(path: str | Path) -> dict[str, TensorInfo]:
    """Parse a safetensors header and check each entry's byte length."""
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: shorter than its 8-byte header length")
        (n,) = struct.unpack("<Q", prefix)
        if n > MAX_HEADER_SIZE:
            raise ValueError(f"{path}: header of {n} bytes is too large")
        header = json.loads(f.read(n))
    data_start = 8 + n
    out: dict[str, TensorInfo] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = entry["dtype"]
        if dtype not in _STORAGE:
            raise ValueError(f"{path}: tensor {name}: unsupported dtype {dtype}")
        shape = tuple(int(d) for d in entry["shape"])
        begin, end = (int(x) for x in entry["data_offsets"])
        want = _STORAGE[dtype].itemsize * int(np.prod(shape, dtype=np.int64))
        if end - begin != want:
            raise ValueError(
                f"{path}: tensor {name}: {end - begin} bytes, want {want} "
                f"for {dtype}{list(shape)}"
            )
        out[name] = TensorInfo(name, dtype, shape, data_start + begin, end - begin)
    return out


def load_raw(path: str | Path, info: TensorInfo) -> np.ndarray:
    """Return a tensor's raw values, memory-mapped and read-only."""
    storage = _STORAGE[info.dtype]
    if info.size == 0:
        return np.zeros(info.shape, dtype=storage)
    return np.memmap(
        path, dtype=storage, mode="r", offset=info.offset, shape=info.shape
    )


def to_float32(dtype: str, raw: np.ndarray) -> np.ndarray:
    """Widen a float tensor's raw values to a new float32 array."""
    if dtype == "BF16":
        return (np.asarray(raw, dtype=np.uint32) << 16).view(np.float32)
    if dtype in ("F16", "F32", "F64"):
        return np.array(raw, dtype=np.float32)
    raise ValueError(f"dtype {dtype} is not a float type")


def from_float32(dtype: str, x: np.ndarray) -> np.ndarray:
    """Round float32 values to a float dtype's raw storage, to nearest even."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "BF16":
        bits = x.view(np.uint32)
        rounded = (
            (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16
        ).astype(np.uint16)
        nan = np.isnan(x)
        if nan.any():
            # Keep NaNs NaN (and quiet) instead of letting rounding carry
            # them into infinity.
            rounded[nan] = ((bits[nan] >> 16) | np.uint32(0x40)).astype(np.uint16)
        return rounded
    if dtype in ("F16", "F32", "F64"):
        return x.astype(_STORAGE[dtype])
    raise ValueError(f"dtype {dtype} is not a float type")


def write_tensor(path: str | Path, name: str, dtype: str, raw: np.ndarray) -> None:
    """Write one tensor as a canonical single-tensor safetensors file."""
    storage = _STORAGE[dtype]
    data = np.ascontiguousarray(raw, dtype=storage).tobytes()
    # Field order as the Go writer emits it: dtype, shape, data_offsets.
    entry = {"dtype": dtype, "shape": list(raw.shape), "data_offsets": [0, len(data)]}
    header = json.dumps(
        {name: entry}, separators=(",", ":"), ensure_ascii=False
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(data)
