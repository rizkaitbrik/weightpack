"""Tests for the NumPy safetensors reader and writer."""

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


@pytest.mark.parametrize(
    ("value", "bits"),
    [
        (1.0, 0x3F80),
        (-2.0, 0xC000),
        (1.0 + 2**-8, 0x3F80),  # halfway between 0x3F80 and 0x3F81: even wins
        (1.0 + 3 * 2**-8, 0x3F82),  # halfway between 0x3F81 and 0x3F82: even wins
        (1.0 + 2**-8 + 2**-20, 0x3F81),  # just past halfway rounds up
        (float("inf"), 0x7F80),
        (3.4e38, 0x7F80),  # rounds past the largest BF16 to infinity
    ],
)
def test_bf16_rounds_to_nearest_even(value, bits):
    assert int(from_float32("BF16", np.array([value], dtype=np.float32))[0]) == bits


def test_bf16_keeps_nan_and_round_trips():
    x = np.array([np.nan, 0.0, -0.0, 1.5, -1e-30], dtype=np.float32)
    raw = from_float32("BF16", x)
    back = to_float32("BF16", raw)
    assert np.isnan(back[0])
    assert np.array_equal(back[1:4], x[1:4])
    # Every BF16 value is exactly representable in float32.
    all_bits = np.arange(0, 1 << 16, dtype=np.uint16)
    finite = ~np.isnan(to_float32("BF16", all_bits))
    assert np.array_equal(
        from_float32("BF16", to_float32("BF16", all_bits))[finite], all_bits[finite]
    )


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16", "I64"])
def test_write_then_read(tmp_path, dtype):
    rng = np.random.default_rng(1)
    values = rng.standard_normal((3, 5)).astype(np.float32)
    raw = values.astype(np.int64) if dtype == "I64" else from_float32(dtype, values)
    path = tmp_path / "t.safetensors"
    write_tensor(path, "model.w", dtype, raw)

    info = read_header(path)["model.w"]
    assert (info.dtype, info.shape) == (dtype, (3, 5))
    assert np.array_equal(load_raw(path, info), raw)
    header_len = struct.unpack("<Q", path.read_bytes()[:8])[0]
    assert (8 + header_len) % 8 == 0


def test_writer_matches_the_go_layout(tmp_path):
    path = tmp_path / "t.safetensors"
    write_tensor(path, "w", "F32", np.zeros((2, 3), dtype=np.float32))
    b = path.read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    assert (
        b[8 : 8 + n].rstrip(b" ")
        == b'{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}'
    )


def test_reads_other_layouts_and_rejects_bad_sizes(tmp_path):
    header = {
        "__metadata__": {"format": "pt"},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "a": {"dtype": "U8", "shape": [3], "data_offsets": [8, 11]},
    }
    body = json.dumps(header).encode()
    path = tmp_path / "m.safetensors"
    path.write_bytes(
        struct.pack("<Q", len(body))
        + body
        + np.array([1.5, -2], "<f4").tobytes()
        + b"\x01\x02\x03"
    )
    h = read_header(path)
    assert set(h) == {"a", "b"}
    assert load_raw(path, h["b"]).tolist() == [1.5, -2.0]
    assert load_raw(path, h["a"]).tolist() == [1, 2, 3]

    header["a"]["shape"] = [4]
    body = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + bytes(11))
    with pytest.raises(ValueError, match="tensor a"):
        read_header(path)
