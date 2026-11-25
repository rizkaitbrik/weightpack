"""Tests for the lossy delta codecs, bit1 and sparse."""

import json

import numpy as np
import pytest

from weightpack_worker import lossy
from weightpack_worker.safetensors_io import (
    from_float32,
    index_tensors,
    load_raw,
    read_header,
    to_float32,
    write_tensor,
)

from .test_squash import write_many


def test_bit1_is_signs_and_the_mean_absolute_delta():
    delta = np.array([[0.5, -1.0, 2.0], [-0.25, 0.0, -3.0]], np.float32)
    entries = {s: v for s, _, v in lossy.encode(lossy.BIT1, delta)}
    assert sorted(entries) == [lossy.SCALE, lossy.SIGNS]
    scale = np.mean(np.abs(delta))
    assert entries[lossy.SCALE][0] == pytest.approx(scale)
    got = lossy.decode(entries, delta.shape)
    np.testing.assert_allclose(got, np.where(delta >= 0, scale, -scale))
    # The scale minimises the squared error of scale * sign.
    for s in (scale * 0.9, scale * 1.1):
        assert np.sum((delta - got) ** 2) < np.sum((delta - s * np.sign(got)) ** 2)


def test_sparse_keeps_the_largest_entries():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal((10, 20)).astype(np.float32)
    entries = {s: v for s, _, v in lossy.encode(lossy.SPARSE, delta, density=0.1)}
    assert (
        entries[lossy.INDICES].dtype == np.uint32 and len(entries[lossy.VALUES]) == 20
    )
    got = lossy.decode(entries, delta.shape)
    kept = np.abs(delta).reshape(-1) >= np.sort(np.abs(delta).reshape(-1))[-20]
    np.testing.assert_array_equal(got.reshape(-1)[kept], delta.reshape(-1)[kept])
    assert np.count_nonzero(got) == 20
    with pytest.raises(ValueError, match="density"):
        lossy.encode(lossy.SPARSE, delta, density=0)


def test_vectors_are_kept_whole():
    delta = np.array([0.1, -0.2, 0.3], np.float32)
    for codec in (lossy.BIT1, lossy.SPARSE):
        entries = {s: v for s, _, v in lossy.encode(codec, delta)}
        assert list(entries) == [lossy.DELTA]
        np.testing.assert_array_equal(lossy.decode(entries, delta.shape), delta)


@pytest.mark.parametrize("codec", [lossy.BIT1, lossy.SPARSE])
def test_encode_lossy(tmp_path, codec):
    rng = np.random.default_rng(1)
    base = {
        "m.q_proj.weight": (
            "BF16",
            from_float32("BF16", rng.standard_normal((16, 8)).astype(np.float32)),
        ),
        "m.norm.weight": ("F32", np.ones(8, np.float32)),
        "m.head.weight": ("F32", rng.standard_normal((4, 8)).astype(np.float32)),
        "m.steps": ("I64", np.arange(3, dtype=np.int64)),
    }
    tuned = dict(base)
    q = to_float32("BF16", base["m.q_proj.weight"][1])
    tuned["m.q_proj.weight"] = (
        "BF16",
        from_float32(
            "BF16", q + 0.05 * rng.standard_normal(q.shape).astype(np.float32)
        ),
    )
    tuned["m.norm.weight"] = ("F32", np.full(8, 1.5, np.float32))
    blobs = []
    for name, (dtype, raw) in base.items():
        p = tmp_path / f"base-{name}.safetensors"
        write_tensor(p, name, dtype, raw)
        blobs.append(str(p))
    src = tmp_path / "src"
    src.mkdir()
    write_many(src / "model.safetensors", tuned)
    (src / "config.json").write_text('{"model_type": "llama"}')
    out, recon = tmp_path / "out", tmp_path / "recon"
    out.mkdir()
    recon.mkdir()

    res = lossy.encode_lossy(
        codec, blobs, [], [str(src / "model.safetensors")], out, 0.25, recon
    )
    assert [n for n, _ in res.tensors] == ["m.norm.weight", "m.q_proj.weight"]
    assert res.unchanged == 2 and 0 < res.relative_error < 1
    deltas = {}
    for name, file in res.tensors:
        got_name, entries = lossy.read_delta(str(out / file))
        assert got_name == name
        deltas[name] = lossy.decode(entries, base[name][1].shape)
    np.testing.assert_array_equal(deltas["m.norm.weight"], np.full(8, 0.5, np.float32))

    # The reconstruction: the parent plus each decoded delta, rounded once,
    # and the fine-tune's other files.
    index = json.loads((recon / "model.safetensors.index.json").read_text())
    assert sorted(index["weight_map"]) == sorted(base)
    got = index_tensors([str(recon / f) for f in set(index["weight_map"].values())])
    for name, (dtype, raw) in base.items():
        path, info = got[name]
        value = np.array(load_raw(path, info))
        if name in deltas:
            want = from_float32(dtype, to_float32(dtype, raw) + deltas[name])
            np.testing.assert_array_equal(value, want)
        else:
            np.testing.assert_array_equal(value, raw)
    assert (recon / "config.json").read_text() == '{"model_type": "llama"}'
    assert not (recon / "model.safetensors").exists()
    # bit1 stores a byte per 8 weights; sparse a quarter of the entries.
    q_file = dict(res.tensors)["m.q_proj.weight"]
    header = read_header(out / q_file)
    if codec == lossy.BIT1:
        assert header["m.q_proj.weight.signs"].shape == (16,)
    else:
        assert header["m.q_proj.weight.values"].shape == (32,)


def test_lossy_refuses_non_float_changes(tmp_path):
    write_tensor(
        tmp_path / "b.safetensors", "steps", "I64", np.arange(3, dtype=np.int64)
    )
    write_tensor(
        tmp_path / "t.safetensors", "steps", "I64", np.arange(1, 4, dtype=np.int64)
    )
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="floats only"):
        lossy.encode_lossy(
            lossy.BIT1,
            [str(tmp_path / "b.safetensors")],
            [],
            [str(tmp_path / "t.safetensors")],
            out,
        )


def test_squash_gives_what_was_evaluated(tmp_path):
    """Squashing a lossy layer gives exactly the model encode_lossy wrote
    for the eval, so the measured drop is the served model's."""
    from weightpack_worker.squash import squash

    rng = np.random.default_rng(2)
    base = {
        "m.q_proj.weight": (
            "BF16",
            from_float32("BF16", rng.standard_normal((16, 8)).astype(np.float32)),
        ),
        "m.norm.weight": ("BF16", from_float32("BF16", np.ones(8, np.float32))),
    }
    tuned = {
        name: (dtype, from_float32(dtype, to_float32(dtype, raw) + 0.03))
        for name, (dtype, raw) in base.items()
    }
    blobs = []
    for name, (dtype, raw) in base.items():
        p = tmp_path / f"base-{name}.safetensors"
        write_tensor(p, name, dtype, raw)
        blobs.append(str(p))
    src = tmp_path / "src"
    src.mkdir()
    write_many(src / "model.safetensors", tuned)
    out, recon, merged = tmp_path / "out", tmp_path / "recon", tmp_path / "merged"
    for d in (out, recon, merged):
        d.mkdir()
    res = lossy.encode_lossy(
        lossy.BIT1, blobs, [], [str(src / "model.safetensors")], out, 0.1, recon
    )
    files = [str(out / f) for _, f in res.tensors]
    pairs = squash(blobs, [], merged, [], [files])
    assert [n for n, _ in pairs] == sorted(base)
    want = index_tensors([str(p) for p in recon.glob("model-*.safetensors")])
    for name, file in pairs:
        h = read_header(merged / file)
        got = np.array(load_raw(merged / file, h[name]))
        np.testing.assert_array_equal(got, np.array(load_raw(*want[name])))
