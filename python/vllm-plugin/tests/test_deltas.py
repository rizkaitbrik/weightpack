"""Per-token bit1 deltas as the plugin applies them inside vLLM, checked
against the dense fine-tunes they stand for."""

import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from weightpack_vllm.deltas import (  # noqa: E402
    DELTA_FILE,
    Matrix,
    ParamSlots,
    Placed,
    Vector,
    linear_delta,
    load_delta,
    rms_normalized,
    rows_delta,
    unpack,
    vector_delta,
)

SLOTS = 3


def encode(delta: np.ndarray) -> Matrix:
    """bit1 as weightpack_worker.lossy encodes it."""
    flat = delta.reshape(-1)
    return Matrix(np.packbits(flat >= 0), float(np.float32(np.abs(flat).mean())))


def decoded(m: Matrix, shape) -> np.ndarray:
    bits = np.unpackbits(m.signs, count=int(np.prod(shape))).astype(np.float32)
    return (bits * 2 - 1).reshape(shape) * np.float32(m.scale)


def write_safetensors(path, tensors):
    """Canonical safetensors of U8 and F32 arrays, as the Go side writes."""
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        a = tensors[name]
        dtype = {np.uint8: "U8", np.float32: "F32"}[a.dtype.type]
        header[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "data_offsets": [offset, offset + a.nbytes],
        }
        offset += a.nbytes
        blobs.append(a.tobytes())
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


@pytest.mark.parametrize("cols", [16, 13])
def test_rows_repack(cols):
    rng = np.random.default_rng(0)
    rows = 5
    m = encode(rng.standard_normal((rows, cols)).astype(np.float32))
    packed = m.rows(rows, cols)
    assert packed.shape == (rows, (cols + 7) // 8)
    got = unpack(torch.from_numpy(packed), cols, torch.float32).numpy()
    np.testing.assert_array_equal(got * np.float32(m.scale), decoded(m, (rows, cols)))
    with pytest.raises(ValueError, match="sign bytes"):
        m.rows(rows + 1, cols)


def test_load_delta(tmp_path):
    rng = np.random.default_rng(1)
    m = encode(rng.standard_normal((4, 8)).astype(np.float32))
    write_safetensors(
        tmp_path / "0.safetensors",
        {"w.signs": m.signs, "w.scale": np.array([m.scale], np.float32)},
    )
    write_safetensors(
        tmp_path / "1.safetensors", {"n.delta": np.arange(4, dtype=np.float32)}
    )
    files = ["0.safetensors", "1.safetensors"]
    meta = {"version": 1, "codec": "bit1", "layer": "ft", "files": files}
    (tmp_path / DELTA_FILE).write_text(json.dumps(meta))
    got = load_delta(str(tmp_path))
    assert np.array_equal(got["w"].signs, m.signs) and got["w"].scale == m.scale
    assert np.array_equal(got["n"].delta, np.arange(4, dtype=np.float32))

    (tmp_path / DELTA_FILE).write_text(json.dumps({**meta, "codec": "sparse"}))
    with pytest.raises(ValueError, match="sparse version 1"):
        load_delta(str(tmp_path))


def packed_layer(rng, out_sizes=(6, 4, 4), cols=24):
    """A packed linear layer, as vLLM's qkv_proj: three checkpoint matrices'
    rows one after another, each with its own delta and scale."""
    w = torch.from_numpy(rng.standard_normal((sum(out_sizes), cols)).astype(np.float32))
    p = ParamSlots(w.shape, SLOTS)
    dense = {}
    for slot in (0, 2):
        full = np.zeros((sum(out_sizes), cols), np.float32)
        offset = 0
        for i, n in enumerate(out_sizes):
            m = encode(rng.standard_normal((n, cols)).astype(np.float32))
            p.put(slot, Placed(f"qkv{i}", offset, n), m)
            full[offset : offset + n] = decoded(m, (n, cols))
            offset += n
        dense[slot] = torch.from_numpy(full)
    return w, p, dense


@pytest.mark.parametrize("graph_safe", [True, False])
def test_linear_delta_is_the_fine_tune(graph_safe):
    """Each token gets its own slot's fine-tune; tokens of no slot, or of a
    slot with no delta here, get the base."""
    rng = np.random.default_rng(2)
    w, p, dense = packed_layer(rng)
    x = torch.from_numpy(rng.standard_normal((6, w.shape[1])).astype(np.float32))
    slots = torch.tensor([0, -1, 2, 1, 0, 2])
    y = x @ w.T + linear_delta(x, slots, p.signs, p.row_scale, p.active, graph_safe)
    for t, s in enumerate(slots.tolist()):
        want = x[t] @ (w + dense[s]).T if s in dense else x[t] @ w.T
        torch.testing.assert_close(y[t], want, rtol=1e-5, atol=1e-5)

    p.clear(0)
    y = x @ w.T + linear_delta(x, slots, p.signs, p.row_scale, p.active, graph_safe)
    torch.testing.assert_close(y[0], x[0] @ w.T)
    torch.testing.assert_close(y[2], x[2] @ (w + dense[2]).T, rtol=1e-5, atol=1e-5)


def test_rows_delta_is_the_fine_tunes_embedding():
    rng = np.random.default_rng(3)
    vocab, dim = 10, 12
    e = torch.from_numpy(rng.standard_normal((vocab + 6, dim)).astype(np.float32))
    p = ParamSlots(e.shape, SLOTS)  # padded past the vocabulary, as vLLM pads it
    m = encode(rng.standard_normal((vocab, dim)).astype(np.float32))
    p.put(1, Placed("embed", 0, vocab), m)
    dense = torch.from_numpy(decoded(m, (vocab, dim)))
    ids, slots = torch.tensor([3, 3, 9, 0]), torch.tensor([1, -1, 1, 0])
    got = e[ids] + rows_delta(ids, slots, p, torch.float32)
    torch.testing.assert_close(got[0], e[3] + dense[3])
    torch.testing.assert_close(got[1], e[3])
    torch.testing.assert_close(got[2], e[9] + dense[9])
    torch.testing.assert_close(got[3], e[0])


def test_vector_delta_and_norm():
    rng = np.random.default_rng(4)
    w = torch.ones(8)
    p = ParamSlots(w.shape, SLOTS)
    d = rng.standard_normal(8).astype(np.float32)
    p.put(2, Placed("norm", 0, 8), Vector(d))
    slots = torch.tensor([2, -1, 0])
    got = vector_delta(slots, p)
    torch.testing.assert_close(got[0], torch.from_numpy(d))
    assert not got[1:].any()

    x = torch.from_numpy(rng.standard_normal((3, 8)).astype(np.float32))
    n = rms_normalized(x, 1e-6)
    want = x / torch.sqrt((x**2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(n, want)
    # A norm with the fine-tune's weight is the base's plus normalized x * delta.
    torch.testing.assert_close(n[0] * (w + got[0]), n[0] * w + n[0] * got[0])


def test_put_checks_kinds():
    p = ParamSlots((4, 8), SLOTS)
    with pytest.raises(ValueError, match="whole delta"):
        p.put(0, Placed("w", 0, 4), Vector(np.zeros(32, np.float32)))
    v = ParamSlots((8,), SLOTS)
    with pytest.raises(ValueError, match="wrong size"):
        v.put(0, Placed("b", 0, 8), Vector(np.zeros(4, np.float32)))
