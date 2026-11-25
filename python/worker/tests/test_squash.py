"""Tests for baking LoRA layers into base tensors."""

import json
import struct

import numpy as np
import pytest

from weightpack_worker.safetensors_io import (
    from_float32,
    load_raw,
    read_header,
    to_float32,
    write_tensor,
)
from weightpack_worker.squash import LoRALayer, squash


def write_many(path, tensors):
    """Write several tensors to one file, as a packed base blob holds them."""
    header, chunks, offset = {}, [], 0
    for name, (dtype, raw) in sorted(tensors.items()):
        data = np.ascontiguousarray(raw).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(raw.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    body = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + b"".join(chunks))


def lora_blob(path, module, a, b, dtype="F32"):
    write_many(
        path,
        {
            module + ".lora_A.weight": (dtype, from_float32(dtype, a)),
            module + ".lora_B.weight": (dtype, from_float32(dtype, b)),
        },
    )
    return str(path)


@pytest.fixture
def model(tmp_path):
    """A BF16 base: two matrices in their own blobs, a norm in a packed one."""
    rng = np.random.default_rng(7)
    q = rng.standard_normal((8, 6)).astype(np.float32)
    up = rng.standard_normal((10, 6)).astype(np.float32)
    norm = rng.standard_normal(6).astype(np.float32)
    blobs = []
    for name, w in [("m.q_proj.weight", q), ("m.up_proj.weight", up)]:
        p = tmp_path / f"{name}.safetensors"
        write_tensor(p, name, "BF16", from_float32("BF16", w))
        blobs.append(str(p))
    packed = tmp_path / "packed.safetensors"
    write_many(packed, {"m.norm.weight": ("BF16", from_float32("BF16", norm))})
    blobs.append(str(packed))
    out = tmp_path / "out"
    out.mkdir()
    return tmp_path, blobs, out, rng


def read_merged(out, merged):
    result = {}
    for name, file in merged:
        info = read_header(out / file)
        assert list(info) == [name]
        result[name] = (
            info[name].dtype,
            to_float32(info[name].dtype, load_raw(out / file, info[name])),
        )
    return result


def test_stacked_layers_merge_in_float32_and_round_once(model):
    tmp, blobs, out, rng = model
    base_q = to_float32(
        "BF16", load_raw(blobs[0], read_header(blobs[0])["m.q_proj.weight"])
    )
    a1, b1 = rng.standard_normal((2, 6)), rng.standard_normal((8, 2))
    a2, b2 = rng.standard_normal((3, 6)), rng.standard_normal((8, 3))
    layers = [
        LoRALayer(
            "french", 0.5, [lora_blob(tmp / "f.safetensors", "m.q_proj", a1, b1)]
        ),
        LoRALayer("legal", 2.0, [lora_blob(tmp / "l.safetensors", "m.q_proj", a2, b2)]),
    ]
    merged = squash(blobs, layers, out)

    assert [n for n, _ in merged] == [
        "m.q_proj.weight"
    ]  # untouched tensors are not written
    dtype, got = read_merged(out, merged)["m.q_proj.weight"]
    assert dtype == "BF16"
    # Independent reference: float64 math on the float32 inputs, rounded once.
    a1f, b1f, a2f, b2f = (
        x.astype(np.float32).astype(np.float64) for x in (a1, b1, a2, b2)
    )
    want64 = base_q + 0.5 * (b1f @ a1f) + 2.0 * (b2f @ a2f)
    want = to_float32("BF16", from_float32("BF16", want64.astype(np.float32)))
    np.testing.assert_allclose(got, want, rtol=2**-7, atol=0)
    assert (
        np.mean(got == want) > 0.95
    )  # float32 summation order flips only the odd last bit


def test_f32_base_is_exact_and_packed_tensors_merge(tmp_path):
    rng = np.random.default_rng(3)
    w = rng.standard_normal((4, 5)).astype(np.float32)
    packed = tmp_path / "packed.safetensors"
    write_many(
        packed,
        {"head.weight": ("F32", w), "norm.weight": ("F32", np.ones(5, np.float32))},
    )
    a, b = (
        rng.standard_normal((1, 5)).astype(np.float32),
        rng.standard_normal((4, 1)).astype(np.float32),
    )
    out = tmp_path / "out"
    out.mkdir()
    merged = squash(
        [str(packed)],
        [LoRALayer("x", 3.0, [lora_blob(tmp_path / "x.safetensors", "head", a, b)])],
        out,
    )
    _, got = read_merged(out, merged)["head.weight"]
    want = w + np.float32(3.0) * (b @ a)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    ("module", "a_shape", "b_shape", "match"),
    [
        ("m.k_proj", (2, 6), (8, 2), "no tensor m.k_proj.weight"),
        ("m.q_proj", (2, 7), (8, 2), "do not fit"),
        ("m.q_proj", (2, 6), (8, 3), "do not fit"),
        ("m.norm", (2, 6), (6, 2), "must be matrices"),
    ],
)
def test_rejects_layers_that_do_not_fit(model, module, a_shape, b_shape, match):
    tmp, blobs, out, rng = model
    blob = lora_blob(
        tmp / "bad.safetensors",
        module,
        rng.standard_normal(a_shape),
        rng.standard_normal(b_shape),
    )
    with pytest.raises(ValueError, match=match):
        squash(blobs, [LoRALayer("bad", 1.0, [blob])], out)
    assert list(out.iterdir()) == []  # nothing written before the checks pass


def test_rejects_blobs_that_are_not_one_module(model):
    tmp, blobs, out, rng = model
    p = tmp / "odd.safetensors"
    write_many(p, {"m.q_proj.lora_A.weight": ("F32", np.zeros((2, 6), np.float32))})
    with pytest.raises(ValueError, match="one module"):
        squash(blobs, [LoRALayer("odd", 1.0, [str(p)])], out)


def test_saved_modules_replace_their_base_tensor_in_chain_order(tmp_path):
    rng = np.random.default_rng(9)
    w = rng.standard_normal((4, 5)).astype(np.float32)
    base = tmp_path / "base.safetensors"
    write_many(base, {"head.weight": ("F32", w)})
    out = tmp_path / "out"
    out.mkdir()

    # Layer 1 adds a LoRA to head; layer 2 replaces head with a saved copy;
    # layer 3 adds a LoRA on top of the copy.
    a1, b1 = (
        rng.standard_normal((1, 5)).astype(np.float32),
        rng.standard_normal((4, 1)).astype(np.float32),
    )
    saved = rng.standard_normal((4, 5)).astype(np.float32)
    saved_blob = tmp_path / "saved.safetensors"
    write_tensor(saved_blob, "head.weight", "F32", saved)
    a3, b3 = (
        rng.standard_normal((1, 5)).astype(np.float32),
        rng.standard_normal((4, 1)).astype(np.float32),
    )
    layers = [
        LoRALayer("one", 1.0, [lora_blob(tmp_path / "1.safetensors", "head", a1, b1)]),
        LoRALayer("two", 1.0, [str(saved_blob)]),
        LoRALayer(
            "three", 2.0, [lora_blob(tmp_path / "3.safetensors", "head", a3, b3)]
        ),
    ]
    _, got = read_merged(out, squash([str(base)], layers, out))["head.weight"]
    np.testing.assert_array_equal(got, saved + np.float32(2.0) * (b3 @ a3))

    # A saved copy that does not fit the base is refused.
    bad = tmp_path / "bad.safetensors"
    write_tensor(bad, "head.weight", "F32", np.zeros((4, 4), np.float32))
    with pytest.raises(ValueError, match="the base has"):
        squash([str(base)], [LoRALayer("bad", 1.0, [str(bad)])], tmp_path / "out")
