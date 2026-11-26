"""Lossy deltas of a full fine-tune over its parent: bit1 and sparse.

For each tensor the fine-tune changed, a lossy layer stores an
approximation of the float32 difference ``target - parent``:

- ``bit1``, as BitDelta (Liu et al., 2024) does: a matrix's delta as one
  sign bit per weight and one scale, the mean absolute delta, which is the
  scale that minimises the squared error of ``scale * sign``.
- ``sparse``: a matrix's delta keeps the ``density`` fraction of its entries
  with the largest magnitudes: a bitmap of which entries it keeps, and their
  values rounded to the tensor's own dtype. The daemon stores these blobs
  zstd-compressed, which takes the bitmap to about half a bit per weight at
  the default density.

A tensor that is not a matrix (a norm's weight, a bias) is small, so its
delta is kept whole, in float32. Decoding adds a delta to the parent in
float32 and rounds once to the tensor's dtype, to nearest even.

Each tensor's delta is a safetensors file of its own:

- bit1: ``<name>.signs`` (U8, the sign bits packed eight to a byte, most
  significant first, in row-major order, 1 for a delta >= 0) and
  ``<name>.scale`` (F32, one value);
- sparse: ``<name>.mask`` (U8, one bit per weight packed as the signs are,
  1 for a kept entry) and ``<name>.values`` (the tensor's dtype, the kept
  entries in row-major order);
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
MASK, VALUES = ".mask", ".values"
DELTA = ".delta"
SUFFIXES = (SIGNS, SCALE, MASK, VALUES, DELTA)


def encode(codec: str, delta: np.ndarray, dtype: str = "F32", density: float = 0.1):
    """A tensor's delta as (suffix, dtype, raw values) entries; dtype is the
    tensor's own, which sparse values are rounded to."""
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
        mask = np.zeros(flat.size, bool)
        mask[np.argpartition(np.abs(flat), flat.size - k)[flat.size - k :]] = True
        values = from_float32(dtype, flat[mask])
        return [(MASK, "U8", np.packbits(mask)), (VALUES, dtype, values)]
    raise ValueError(f"unknown lossy codec {codec!r}")


def decode(entries: dict[str, tuple[str, np.ndarray]], shape) -> np.ndarray:
    """The float32 delta of shape that encode's entries give, by suffix, each
    as (dtype, raw values)."""
    n = int(np.prod(shape, dtype=np.int64))
    if DELTA in entries:
        return to_float32(*entries[DELTA]).reshape(shape)
    if SIGNS in entries:
        signs = np.unpackbits(np.asarray(entries[SIGNS][1]), count=n)
        scale = np.float32(to_float32(*entries[SCALE]).reshape(-1)[0])
        out = signs.astype(np.float32) * (2 * scale) - scale
        return out.reshape(shape)
    if MASK in entries:
        mask = np.unpackbits(np.asarray(entries[MASK][1]), count=n).astype(bool)
        values = to_float32(*entries[VALUES]).reshape(-1)
        if int(mask.sum()) != values.size:
            raise ValueError(
                f"sparse delta keeps {int(mask.sum())} entries but holds "
                f"{values.size} values"
            )
        out = np.zeros(n, np.float32)
        out[mask] = values
        return out.reshape(shape)
    raise ValueError(f"no delta among {sorted(entries)}")


def read_delta(path: str) -> tuple[str, dict[str, tuple[str, np.ndarray]]]:
    """A delta file's tensor name and its entries by suffix, each as (dtype,
    raw values)."""
    header = read_header(path)
    names = set()
    entries = {}
    for key, info in header.items():
        suffix = next((s for s in SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            raise ValueError(f"{path}: {key} is no lossy delta entry")
        names.add(key[: -len(suffix)])
        entries[suffix] = (info.dtype, load_raw(path, info))
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
            entries = encode(codec, delta, info.dtype, density)
            approx = decode({s: (d, v) for s, d, v in entries}, info.shape)
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
