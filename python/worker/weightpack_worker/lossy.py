"""Lossy deltas of a full fine-tune over its parent: bit1 and sparse.

For each tensor the fine-tune changed, a lossy layer stores an
approximation of the float32 difference ``target - parent``:

- ``bit1``, as BitDelta (Liu et al., 2024) does: a matrix's delta as one
  sign bit per weight and one scale, the mean absolute delta, which is the
  scale that minimises the squared error of ``scale * sign``.
- ``sparse``: a matrix's delta keeps the ``density`` fraction of its entries
  with the largest magnitudes, as flat indices and float32 values.

A tensor that is not a matrix (a norm's weight, a bias) is small, so its
delta is kept whole, in float32. Decoding adds a delta to the parent in
float32 and rounds once to the tensor's dtype, to nearest even.

Each tensor's delta is a safetensors file of its own:

- bit1: ``<name>.signs`` (U8, the sign bits packed eight to a byte, most
  significant first, in row-major order, 1 for a delta >= 0) and
  ``<name>.scale`` (F32, one value);
- sparse: ``<name>.indices`` (U32, or I64 past 2^32 entries; flat row-major
  indices, ascending) and ``<name>.values`` (F32);
- a tensor kept whole: ``<name>.delta`` (F32, the tensor's shape).

Lossy layers sit on the base or on XOR layers, as XOR layers do: the
parent's bits are exact, so a delta is taken against exactly the weights it
is later added to.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weightpack_worker.delta import Parent, bits, fine_tune
from weightpack_worker.safetensors_io import (
    FLOAT_DTYPES,
    from_float32,
    load_raw,
    read_header,
    storage_dtype,
    to_float32,
    write_tensor,
    write_tensors,
)

BIT1, SPARSE = "bit1", "sparse"
SIGNS, SCALE = ".signs", ".scale"
INDICES, VALUES = ".indices", ".values"
DELTA = ".delta"
SUFFIXES = (SIGNS, SCALE, INDICES, VALUES, DELTA)


def encode(codec: str, delta: np.ndarray, density: float = 0.1) -> list:
    """A tensor's delta as (suffix, dtype, values) entries."""
    if delta.ndim < 2:
        return [(DELTA, "F32", delta.astype(np.float32))]
    flat = delta.reshape(-1)
    if codec == BIT1:
        scale = np.float32(np.mean(np.abs(flat))) if flat.size else np.float32(0)
        signs = np.packbits(flat >= 0)
        return [(SIGNS, "U8", signs), (SCALE, "F32", np.array([scale], np.float32))]
    if codec == SPARSE:
        if not 0 < density <= 1:
            raise ValueError(f"density {density} is not in (0, 1]")
        k = max(1, int(np.ceil(density * flat.size)))
        keep = np.argpartition(np.abs(flat), flat.size - k)[flat.size - k :]
        keep.sort()
        idx_dtype = "U32" if flat.size <= np.iinfo(np.uint32).max else "I64"
        indices = keep.astype(storage_dtype(idx_dtype))
        return [(INDICES, idx_dtype, indices), (VALUES, "F32", flat[keep])]
    raise ValueError(f"unknown lossy codec {codec!r}")


def decode(tensors: dict[str, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """The float32 delta of shape that encode's entries (by suffix) give."""
    n = int(np.prod(shape, dtype=np.int64))
    if DELTA in tensors:
        return np.asarray(tensors[DELTA], np.float32).reshape(shape)
    if SIGNS in tensors:
        signs = np.unpackbits(np.asarray(tensors[SIGNS]), count=n)
        scale = np.float32(np.asarray(tensors[SCALE]).reshape(-1)[0])
        out = signs.astype(np.float32) * (2 * scale) - scale
        return out.reshape(shape)
    if INDICES in tensors:
        out = np.zeros(n, np.float32)
        out[np.asarray(tensors[INDICES], np.int64)] = tensors[VALUES]
        return out.reshape(shape)
    raise ValueError(f"no delta among {sorted(tensors)}")


def read_delta(path: str) -> tuple[str, dict[str, np.ndarray]]:
    """A delta file's tensor name and its entries by suffix."""
    header = read_header(path)
    names = set()
    entries = {}
    for key, info in header.items():
        suffix = next((s for s in SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            raise ValueError(f"{path}: {key} is no lossy delta entry")
        names.add(key[: -len(suffix)])
        entries[suffix] = load_raw(path, info)
    if len(names) != 1:
        raise ValueError(f"{path} must hold one tensor's delta, found {sorted(names)}")
    return names.pop(), entries


@dataclass
class Encoded:
    tensors: list[tuple[str, str]]  # (tensor name, file name), by name
    unchanged: int
    relative_error: float  # ||delta - decoded|| / ||delta|| over all deltas


def encode_lossy(
    codec: str,
    base_blobs: list[str],
    parents: list[list[str]],
    target_files: list[str],
    out_dir: str | Path,
    density: float = 0.1,
    reconstruct_dir: str | Path | None = None,
) -> Encoded:
    """Encode a fine-tune as a lossy delta over its parent.

    Writes one delta file per changed tensor into out_dir. With
    reconstruct_dir, also writes there the model the layer gives back: the
    parent with each delta decoded and added, as safetensors with an index,
    beside copies of the fine-tune's other files (config, tokenizer), so it
    can be evaluated against the fine-tune.
    """
    out_dir = Path(out_dir)
    parent = Parent(base_blobs, parents)
    target = fine_tune(parent, target_files)
    recon = Path(reconstruct_dir) if reconstruct_dir else None
    weight_map: dict[str, str] = {}
    total = 0
    encoded: list[tuple[str, str]] = []
    unchanged = 0
    err2 = norm2 = 0.0
    for i, name in enumerate(sorted(target)):
        path, info = target[name]
        new = bits(path, info)
        old = parent.bits(name)
        raw = old.view(storage_dtype(info.dtype))
        if not np.array_equal(new, old):
            if info.dtype not in FLOAT_DTYPES:
                raise ValueError(
                    f"{name} is {info.dtype}; lossy codecs take floats only"
                )
            w = to_float32(info.dtype, raw)
            delta = to_float32(info.dtype, new.view(storage_dtype(info.dtype))) - w
            entries = encode(codec, delta, density)
            approx = decode({s: v for s, _, v in entries}, info.shape)
            err2 += float(np.sum((delta - approx) ** 2, dtype=np.float64))
            norm2 += float(np.sum(delta**2, dtype=np.float64))
            file = f"{i:06d}.safetensors"
            write_tensors(out_dir / file, [(name + s, d, v) for s, d, v in entries])
            encoded.append((name, file))
            raw = from_float32(info.dtype, w + approx)
        else:
            unchanged += 1
        if recon is not None:
            file = f"model-{i:06d}.safetensors"
            write_tensor(recon / file, name, info.dtype, raw)
            weight_map[name] = file
            total += raw.nbytes
    if recon is not None:
        _write_index(recon, weight_map, total)
        _copy_meta(Path(target_files[0]).parent, recon)
    rel = float(np.sqrt(err2 / norm2)) if norm2 > 0 else 0.0
    return Encoded(encoded, unchanged, rel)


def _write_index(dir: Path, weight_map: dict[str, str], total: int) -> None:
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))


def _copy_meta(src: Path, dst: Path) -> None:
    """Copies a model directory's files but its weights: config, tokenizer,
    chat template, generation config."""
    for f in src.iterdir():
        if (
            f.is_file()
            and f.suffix != ".safetensors"
            and not f.name.endswith(".safetensors.index.json")
        ):
            shutil.copyfile(f, dst / f.name)
