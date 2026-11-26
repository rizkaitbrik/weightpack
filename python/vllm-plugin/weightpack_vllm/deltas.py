"""Full fine-tunes served as bit1 deltas over vLLM's base, per token.

A bit1 layer (weightpack_worker.lossy) holds, for each matrix a fine-tune
changed, the signs of its delta packed eight to a byte, most significant bit
first, over the flattened matrix, and one float32 scale; every other tensor
it changed (norms, biases) it holds whole, as a float32 delta. Decoding gives
a matrix W + scale * sign and any other tensor W + delta.

vLLM batches requests for different models together, so the weights cannot
change per request. Each adapter slot instead gets buffers beside every
parameter: a matrix's signs, packed by row, and one scale per row; another
tensor's delta. A token whose slot holds a fine-tune gets, besides the base's
own output, its delta's:

    linear     y += scale * (sign(delta) x)         per output row
    embedding  e += scale * sign(delta)[id]
    vector     y += delta (a bias), or norm(x) * delta (an RMSNorm's weight)

which is the decoded model's output up to the rounding of one sum, not bit
for bit as weightpack-engine's. The buffers are allocated once, before vLLM
captures CUDA graphs, and every op reads the token's slot on the device, so
captured graphs serve fine-tunes loaded later.

This module needs only torch and NumPy, so it is tested without vLLM;
patches.py connects it to vLLM, and bit1_kernel.py is the CUDA kernel.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

DELTA_FILE = "weightpack_delta.json"
DELTA_VERSION = 1
SIGNS, SCALE, WHOLE = ".signs", ".scale", ".delta"

# MSB first: bit j of a byte is (byte >> (7 - j)) & 1.
_SHIFTS = list(range(7, -1, -1))


@dataclass
class Matrix:
    """A matrix's delta: its signs packed eight to a byte over the flattened
    matrix, and one scale."""

    signs: np.ndarray  # uint8
    scale: float

    def rows(self, rows: int, cols: int) -> np.ndarray:
        """The signs packed by row: (rows, ceil(cols / 8)), each row's
        padding bits zero."""
        if self.signs.size != (rows * cols + 7) // 8:
            raise ValueError(f"{self.signs.size} sign bytes for a {rows}x{cols} matrix")
        if cols % 8 == 0:
            return self.signs.reshape(rows, cols // 8)
        bits = np.unpackbits(self.signs, count=rows * cols).reshape(rows, cols)
        return np.packbits(bits, axis=1)


@dataclass
class Vector:
    """Another tensor's whole delta, flattened."""

    delta: np.ndarray  # float32


def is_delta_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, DELTA_FILE))


def _read_safetensors(path: str) -> dict[str, np.ndarray]:
    """A small safetensors file's U8 and F32 tensors, flattened."""
    dtypes = {"U8": np.dtype("u1"), "F32": np.dtype("<f4")}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] not in dtypes:
            raise ValueError(f"{path}: {name} is {info['dtype']}, not a bit1 entry")
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(data[start:end], dtypes[info["dtype"]]).copy()
    return out


def load_delta(path: str) -> dict[str, Matrix | Vector]:
    """A bit1 layer's deltas by checkpoint tensor name, from the directory
    wpkd exports (weightpack_delta.json and the layer's blobs)."""
    with open(os.path.join(path, DELTA_FILE)) as f:
        meta = json.load(f)
    if meta.get("version") != DELTA_VERSION or meta.get("codec") != "bit1":
        raise ValueError(
            f"{path}: {DELTA_FILE} is {meta.get('codec')} version "
            f"{meta.get('version')}; this plugin serves bit1 version {DELTA_VERSION}"
        )
    out: dict[str, Matrix | Vector] = {}
    for rel in meta.get("files") or []:
        file = os.path.join(path, rel)
        entries = _read_safetensors(file)
        names = {k.rsplit(".", 1)[0] for k in entries}
        if len(names) != 1:
            raise ValueError(f"{file} must hold one tensor's delta")
        name = names.pop()
        if set(entries) == {name + SIGNS, name + SCALE}:
            scale = entries[name + SCALE]
            if scale.size != 1 or not np.isfinite(scale[0]):
                raise ValueError(f"{file}: a bit1 scale is one finite value")
            out[name] = Matrix(entries[name + SIGNS], float(scale[0]))
        elif set(entries) == {name + WHOLE}:
            out[name] = Vector(entries[name + WHOLE])
        else:
            raise ValueError(f"{file}: entries {sorted(entries)} are no bit1 delta")
    # No deltas: a fine-tune that changed nothing, served as the base.
    return out


@dataclass
class Placed:
    """Where a checkpoint tensor's delta goes in a parameter: its first row
    (or element, for a vector) and how many."""

    param: str
    offset: int
    rows: int


class ParamSlots:
    """The delta buffers of one parameter of shape, one per adapter slot: a
    matrix's signs packed by row (slots, rows, ceil(cols / 8)) and per-row
    scales (slots, rows), or another tensor's delta (slots, numel). A slot
    whose fine-tune leaves the parameter alone is inactive. The shape is the
    checkpoint's: vLLM's CPU backend repacks a linear layer's weight and
    keeps an empty parameter in its place."""

    def __init__(self, shape, slots: int, device="cpu"):
        import torch

        self.shape = tuple(int(d) for d in shape)
        self.matrix = len(self.shape) == 2
        if self.matrix:
            rows, cols = self.shape
            self.signs = torch.zeros(
                slots, rows, (cols + 7) // 8, dtype=torch.uint8, device=device
            )
            self.row_scale = torch.zeros(
                slots, rows, dtype=torch.float32, device=device
            )
        else:
            numel = int(np.prod(self.shape, dtype=np.int64))
            self.vector = torch.zeros(slots, numel, dtype=torch.float32, device=device)
        self.active = torch.zeros(slots, dtype=torch.int32, device=device)

    @property
    def cols(self) -> int:
        return self.shape[1]

    def clear(self, slot: int) -> None:
        self.active[slot] = 0
        if self.matrix:
            self.row_scale[slot].zero_()
        else:
            self.vector[slot].zero_()

    def put(self, slot: int, placed: Placed, delta: Matrix | Vector) -> None:
        import torch

        o, n = placed.offset, placed.rows
        if self.matrix:
            if not isinstance(delta, Matrix):
                raise ValueError(f"{placed.param}: a matrix with a whole delta")
            signs = torch.from_numpy(delta.rows(n, self.cols))
            self.signs[slot, o : o + n].copy_(signs)
            self.row_scale[slot, o : o + n] = delta.scale
        else:
            if not isinstance(delta, Vector) or delta.delta.size != n:
                raise ValueError(f"{placed.param}: a delta of the wrong size")
            self.vector[slot, o : o + n].copy_(torch.from_numpy(delta.delta))
        self.active[slot] = 1


def unpack(signs, cols: int, dtype):
    """Rows of packed signs, (..., ceil(cols / 8)) uint8, as (..., cols)
    of +1 and -1 in dtype."""
    import torch

    shifts = torch.tensor(_SHIFTS, dtype=torch.uint8, device=signs.device)
    bits = (signs.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(*signs.shape[:-1], -1)[..., :cols]
    return bits.to(dtype) * 2 - 1


def linear_delta(x, slots, signs, row_scale, active, graph_safe: bool = True):
    """Each token's delta product for a linear layer of in-features
    x.shape[1], (tokens, rows) in float32: row_scale * (sign x) for a token
    in an active slot, else 0. signs, row_scale and active are a
    ParamSlots' buffers.

    graph_safe computes every slot's product for every token and keeps each
    token's own, so it reads nothing on the host and a CUDA graph can
    capture it; otherwise only the slots the tokens use are computed. The
    CUDA kernel (bit1_kernel) gives the same, faster."""
    import torch

    t, n_slots, cols = x.shape[0], signs.shape[0], x.shape[1]
    out = torch.zeros(t, signs.shape[1], dtype=torch.float32, device=x.device)
    if graph_safe:
        used = range(n_slots)
    else:
        used = [s for s in slots.unique().tolist() if s >= 0 and int(active[s])]
    for s in used:
        w = unpack(signs[s], cols, x.dtype)
        mine = (slots == s) & (active[s] > 0)
        product = (x @ w.T).float() * row_scale[s]
        out = torch.where(mine.view(t, 1), product, out)
    return out


def rows_delta(ids, slots, p: ParamSlots, dtype):
    """Each token's embedding delta, (tokens, cols) in float32."""
    import torch

    s = slots.clamp(min=0)
    signs = unpack(p.signs[s, ids], p.cols, torch.float32)
    keep = (slots >= 0) & (p.active[s] > 0)
    return torch.where(keep.view(-1, 1), signs * p.row_scale[s, ids].view(-1, 1), 0.0)


def vector_delta(slots, p: ParamSlots):
    """Each token's delta of a vector, (tokens, numel) in float32."""
    import torch

    s = slots.clamp(min=0)
    keep = (slots >= 0) & (p.active[s] > 0)
    return torch.where(keep.view(-1, 1), p.vector[s], 0.0)


def rms_normalized(x, eps: float, var_size: int | None = None):
    """x / sqrt(mean(x^2) + eps) in float32, over the last dimension (its
    first var_size entries, as vLLM's variance_size_override)."""
    import torch

    xf = x.float()
    v = xf if var_size is None else xf[..., :var_size]
    return xf * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
