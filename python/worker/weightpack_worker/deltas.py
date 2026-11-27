"""Full fine-tunes served as bit1 deltas over the reference engine's base.

A bit1 layer (lossy.py) holds, for each matrix a fine-tune changed, the
signs of its delta packed eight to a byte, most significant bit first, and
one float32 scale; every other tensor it changed (norms, biases) it holds
whole, as a float32 delta. Decoding gives a matrix round(W + scale * sign)
and any other tensor round(W + delta), added in float32 and rounded once to
the tensor's dtype.

The reference engine runs one request at a time, so it serves a bit1
fine-tune by switching its weights in place. Before a request on the
fine-tune, each tensor the fine-tune changed is rewritten from the base's
own file as decoding gives it; before a request that needs other weights
(the base, its LoRA adapters, another fine-tune), those tensors are read
back from the same file. A request then runs at the dense model's speed and
on exactly the model decoding gives, which the import's eval gate scored.
However many fine-tunes are loaded, the engine holds one copy of the
weights: a fine-tune's signs stay in its files, read at each switch.

wpkd exports a bit1 layer for the engine as a directory holding
weightpack_delta.json (version 1: the codec, the layer's id, and its files)
beside the layer's own blobs, one tensor's delta each, and loads it with
/v1/load_lora_adapter like any adapter.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .safetensors_io import read_header

DELTA_FILE = "weightpack_delta.json"
DELTA_VERSION = 1

SIGNS, SCALE, WHOLE = ".signs", ".scale", ".delta"

# Weights rewritten per step of a switch: bounds the float32 scratch a
# switch needs to 64 MB, whatever the size of a tensor.
CHUNK = 1 << 24


def is_delta_dir(path: str) -> bool:
    return (Path(path) / DELTA_FILE).is_file()


@dataclass
class _Entry:
    """One tensor a fine-tune changes, as its file stores it: the signs (U8)
    and scale of a matrix, or the whole float32 delta of another tensor, at
    offset, size bytes long."""

    name: str
    file: str
    signs: bool
    offset: int
    size: int
    scale: float = 0.0

    def read(self) -> np.ndarray:
        dtype = np.uint8 if self.signs else np.dtype("<f4")
        count = self.size // np.dtype(dtype).itemsize
        return np.fromfile(self.file, dtype, count, offset=self.offset)


@dataclass
class Delta:
    """A bit1 fine-tune of the engine's base."""

    name: str
    path: str
    # By the model tensor's key (Switcher.key): a tensor tied under several
    # names, such as embeddings and the output head, is one entry.
    entries: dict[str, _Entry]


class BaseFiles:
    """The tensors of the engine's base model directory, as its files store
    them. A switch reads hundreds, so each file stays open (mapped) rather
    than its header being parsed again for every tensor."""

    def __init__(self, model_dir: str):
        root = Path(model_dir)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.files = {n: str(root / f) for n, f in weight_map.items()}
        else:
            self.files = {}
            for f in sorted(root.glob("*.safetensors")):
                for n in read_header(f):
                    self.files[n] = str(f)
        if not self.files:
            raise ValueError(f"{model_dir} holds no safetensors weights")
        self._open: dict = {}  # file -> its safetensors handle

    def _handle(self, name: str):
        from safetensors import safe_open

        file = self.files[name]
        if file not in self._open:
            self._open[file] = safe_open(file, framework="pt")
        return self._open[file]

    def read(self, name: str):
        return self._handle(name).get_tensor(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._handle(name).get_slice(name).get_shape())


class Switcher:
    """Puts a model's weights in the state a request needs: its base's, or
    a loaded bit1 fine-tune's.

    tensors are the model's parameters and buffers by checkpoint name, as
    the engine loaded them, before any adapter wrapped a module: their
    objects stay the model's own after PEFT wraps them.
    """

    def __init__(self, tensors: dict, model_dir: str, device: str):
        import torch

        self.torch = torch
        self.tensors = tensors
        self.model_dir = model_dir
        self.device = device
        self.base: BaseFiles | None = None  # read once the first delta loads
        self.active: Delta | None = None
        self._shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
        # Each tensor's names, the first being its key.
        self.names: dict[int, list[str]] = {}
        for n, t in tensors.items():
            self.names.setdefault(id(t), []).append(n)
        # By key, the name the base's files hold each tensor by.
        self.base_name: dict[str, str] = {}

    def key(self, name: str) -> str:
        return self.names[id(self.tensors[name])][0]

    def load(self, name: str, path: str) -> Delta:
        """Reads a delta directory's description and checks it against the
        model; the signs stay in the files until a switch reads them."""
        meta = json.loads((Path(path) / DELTA_FILE).read_text())
        if meta.get("version") != DELTA_VERSION:
            raise ValueError(
                f"{path}: {DELTA_FILE} version {meta.get('version')}; "
                f"this engine reads version {DELTA_VERSION}"
            )
        if meta.get("codec") != "bit1":
            raise ValueError(
                f"{path}: a {meta.get('codec')} delta; the engine serves bit1 "
                "deltas over its base, so wpk squash the image to serve it dense"
            )
        if self.base is None:
            self.base = BaseFiles(self.model_dir)
        entries: dict[str, _Entry] = {}
        for rel in meta.get("files") or []:
            entry, size = _entry(str(Path(path) / rel))
            t = self.tensors.get(entry.name)
            stored = [n for n in self.names.get(id(t), []) if n in self.base.files]
            if t is None or not stored:
                raise ValueError(
                    f"the fine-tune changes {entry.name}, which the base model has "
                    "no tensor for; wpk squash the image to serve it dense"
                )
            shape = self.base.shape(stored[0])
            if tuple(t.shape) != shape or not t.dtype.is_floating_point:
                raise ValueError(f"{entry.name}: the model's tensor is not the base's")
            n = t.numel()
            if entry.signs and size != (n + 7) // 8:
                raise ValueError(f"{entry.name}: {size} sign bytes for {n} weights")
            if not entry.signs and size != n:
                raise ValueError(f"{entry.name}: a delta of {size} values for {n}")
            key = self.key(entry.name)
            if key in entries:
                # A tied tensor stored under two names: one delta, given twice.
                if not _same(entries[key], entry):
                    raise ValueError(
                        f"{entries[key].name} and {entry.name} are one tensor in "
                        "the model but the fine-tune changes them differently"
                    )
                continue
            self.base_name[key] = stored[0]
            entries[key] = entry
        return Delta(name, path, entries)

    def activate(self, delta: Delta | None) -> None:
        """Switches the weights to delta's fine-tune, or to the base."""
        if delta is self.active:
            return
        start = time.monotonic()
        old = self.active.entries if self.active else {}
        new = delta.entries if delta else {}
        keys = sorted(set(old) | set(new))
        # The state is unknown until every tensor is rewritten: a switch
        # that fails part way puts the base back.
        self.active = None
        try:
            for key in keys:
                self._write(key, new.get(key))
        except BaseException:
            for key in keys:
                self._write(key, None)
            raise
        self.active = delta
        print(
            f"engine: weights switched to {delta.name if delta else 'the base'} "
            f"({len(keys)} tensors) in {time.monotonic() - start:.2f} s",
            file=sys.stderr,
            flush=True,
        )

    def _write(self, key: str, entry: _Entry | None) -> None:
        """Writes one tensor as the base holds it, or as entry changes it:
        chunk by chunk, each rounded once to the checkpoint's dtype, as
        decoding rounds, and then held in the model's."""
        torch = self.torch
        t = self.tensors[key]
        w = self.base.read(self.base_name[key])  # the checkpoint's own dtype
        with torch.no_grad():
            if entry is None:
                t.copy_(w.reshape(t.shape))
                return
            flat, out = w.reshape(-1), t.view(-1)
            data = torch.from_numpy(entry.read()).to(self.device)
            scale = torch.tensor(entry.scale, device=self.device)  # float32
            for i in range(0, flat.numel(), CHUNK):
                n = min(CHUNK, flat.numel() - i)
                if entry.signs:
                    byte = data[i // 8 : (i + n + 7) // 8].unsqueeze(-1)
                    bits = ((byte >> self._shifts) & 1).reshape(-1)[:n]
                    d = torch.where(bits.bool(), scale, -scale)
                else:
                    d = data[i : i + n]
                base = flat[i : i + n].to(self.device).float()
                out[i : i + n] = (base + d).to(w.dtype)


def _entry(file: str) -> tuple[_Entry, int]:
    """A delta file's entry, and how many sign bytes or values it holds."""
    header = read_header(file)
    names = {k.rsplit(".", 1)[0] for k in header}
    if len(names) != 1:
        raise ValueError(f"{file} must hold one tensor's delta")
    name = names.pop()
    keys = set(header)
    if keys == {name + SIGNS, name + SCALE}:
        signs, scale = header[name + SIGNS], header[name + SCALE]
        if signs.dtype != "U8" or scale.dtype != "F32" or scale.size != 4:
            raise ValueError(f"{file}: bit1 holds U8 signs and one F32 scale")
        value = np.fromfile(file, np.dtype("<f4"), 1, offset=scale.offset)[0]
        if not np.isfinite(value):
            raise ValueError(f"{file}: the scale is {value}")
        return _Entry(name, file, True, signs.offset, signs.size, float(value)), (
            signs.size
        )
    if keys == {name + WHOLE} and header[name + WHOLE].dtype == "F32":
        whole = header[name + WHOLE]
        return _Entry(name, file, False, whole.offset, whole.size), whole.size // 4
    raise ValueError(
        f"{file}: entries {sorted(keys)} are no bit1 delta; "
        "wpk squash the image to serve it dense"
    )


def _same(a: _Entry, b: _Entry) -> bool:
    return (
        a.signs == b.signs and a.scale == b.scale and np.array_equal(a.read(), b.read())
    )
