"""Tests for lossless XOR deltas and decoding them in squash."""

import numpy as np
import pytest

from weightpack_worker.delta import encode_xor
from weightpack_worker.safetensors_io import (
    from_float32,
    load_raw,
    read_header,
    to_float32,
    write_tensor,
)
from weightpack_worker.squash import LoRALayer, squash

from .test_squash import lora_blob, write_many


def base_and_finetune(tmp_path, rng):
    """A base with BF16, F32 and I64 tensors in separate blobs, and a
    fine-tune that changes the matrices but not the norm or the counter."""
    base = {
        "m.q_proj.weight": (
            "BF16",
            from_float32("BF16", rng.standard_normal((8, 6)).astype(np.float32)),
        ),
        "m.head.weight": ("F32", rng.standard_normal((4, 6)).astype(np.float32)),
        "m.norm.weight": ("BF16", from_float32("BF16", np.ones(6, np.float32))),
        "m.steps": ("I64", np.arange(3, dtype=np.int64)),
    }
    tuned = dict(base)
    q = to_float32("BF16", base["m.q_proj.weight"][1])
    tuned["m.q_proj.weight"] = (
        "BF16",
        from_float32(
            "BF16", q + 0.01 * rng.standard_normal(q.shape).astype(np.float32)
        ),
    )
    head = base["m.head.weight"][1].copy()
    head[0, 0] = np.float32("nan")  # NaN bit patterns must survive too
    tuned["m.head.weight"] = ("F32", head + 0)
    blobs = []
    for name, (dtype, raw) in base.items():
        p = tmp_path / f"base-{name}.safetensors"
        write_tensor(p, name, dtype, raw)
        blobs.append(str(p))
    target = tmp_path / "model.safetensors"
    write_many(target, tuned)
    out = tmp_path / "out"
    out.mkdir()
    return blobs, [str(target)], out, tuned


def read_files(out, pairs):
    result = {}
    for name, file in pairs:
        h = read_header(out / file)
        assert list(h) == [name]
        result[name] = np.array(load_raw(out / file, h[name]))
    return result


def bits_equal(a, b):
    return a.tobytes() == b.tobytes()


def test_xor_round_trip_is_bit_exact(tmp_path):
    rng = np.random.default_rng(0)
    blobs, target, out, tuned = base_and_finetune(tmp_path, rng)
    encoded, unchanged = encode_xor(blobs, [], target, out)
    assert [n for n, _ in encoded] == ["m.head.weight", "m.q_proj.weight"]
    assert unchanged == 2  # the norm and the counter cost nothing

    # Decoding through squash (no LoRA) gives the fine-tune back exactly.
    xor_blobs = [str(out / f) for _, f in encoded]
    dec = tmp_path / "dec"
    dec.mkdir()
    merged = read_files(dec, squash(blobs, [], dec, [xor_blobs]))
    assert set(merged) == {"m.head.weight", "m.q_proj.weight"}
    for name, raw in merged.items():
        assert bits_equal(raw, np.asarray(tuned[name][1])), name


def test_stacked_xor_layers_and_lora_on_top(tmp_path):
    rng = np.random.default_rng(1)
    blobs, target, out, tuned = base_and_finetune(tmp_path, rng)
    first, _ = encode_xor(blobs, [], target, out)
    first = [str(out / f) for _, f in first]

    # A second fine-tune of the first one, encoded over base ^ first.
    q = to_float32("BF16", tuned["m.q_proj.weight"][1])
    again = dict(tuned)
    again["m.q_proj.weight"] = ("BF16", from_float32("BF16", q * np.float32(1.5)))
    target2 = tmp_path / "model2.safetensors"
    write_many(target2, again)
    out2 = tmp_path / "out2"
    out2.mkdir()
    second, unchanged = encode_xor(blobs, [first], [str(target2)], out2)
    assert [n for n, _ in second] == ["m.q_proj.weight"] and unchanged == 3
    second = [str(out2 / f) for _, f in second]

    # A LoRA on top merges against the decoded fine-tune, in float32.
    a, b = rng.standard_normal((2, 6)), rng.standard_normal((8, 2))
    lora = LoRALayer(
        "x", 0.5, [lora_blob(tmp_path / "l.safetensors", "m.q_proj", a, b)]
    )
    dec = tmp_path / "dec"
    dec.mkdir()
    merged = read_files(dec, squash(blobs, [lora], dec, [first, second]))
    want = from_float32(
        "BF16",
        to_float32("BF16", again["m.q_proj.weight"][1])
        + np.float32(0.5) * (b.astype(np.float32) @ a.astype(np.float32)),
    )
    assert bits_equal(merged["m.q_proj.weight"], want)
    assert bits_equal(merged["m.head.weight"], tuned["m.head.weight"][1])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda t: t.pop("m.norm.weight"), "missing"),
        (lambda t: t.update({"m.extra": ("F32", np.zeros(2, np.float32))}), "extra"),
        (
            lambda t: t.update(
                {"m.head.weight": ("F32", np.zeros((4, 5), np.float32))}
            ),
            "in the base",
        ),
        (
            lambda t: t.update({"m.steps": ("I32", np.arange(3, dtype=np.int32))}),
            "in the base",
        ),
    ],
)
def test_rejects_a_finetune_of_another_model(tmp_path, change, match):
    rng = np.random.default_rng(2)
    blobs, target, out, tuned = base_and_finetune(tmp_path, rng)
    change(tuned)
    write_many(tmp_path / "model.safetensors", tuned)
    with pytest.raises(ValueError, match=match):
        encode_xor(blobs, [], target, out)
    assert list(out.iterdir()) == []
