"""Tests for the reference delta codecs."""

import numpy as np
import pytest

from weightpack_worker import bit1_decode, bit1_encode, xor_decode, xor_encode


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    parent = rng.standard_normal((64, 32)).astype(np.float32)
    target = parent + 0.01 * rng.standard_normal(parent.shape).astype(np.float32)
    return parent, target


def test_xor_round_trip_is_exact(pair):
    parent, target = pair
    delta = xor_encode(parent, target)
    restored = xor_decode(parent, delta)
    assert restored.dtype == target.dtype
    assert np.array_equal(restored.view(np.uint32), target.view(np.uint32))


def test_xor_on_bf16_bytes():
    # BF16 tensors arrive as raw uint16 words.
    parent = np.arange(256, dtype=np.uint16)
    target = parent.copy()
    target[10] ^= 1
    delta = xor_encode(parent, target)
    assert int(np.count_nonzero(delta)) == 1
    assert np.array_equal(xor_decode(parent, delta), target)


def test_xor_of_identical_tensors_is_all_zero(pair):
    parent, _ = pair
    assert not np.any(xor_encode(parent, parent))


def test_bit1_is_small_and_close(pair):
    parent, target = pair
    delta = bit1_encode(parent, target)
    assert delta.nbytes < target.nbytes / 16
    approx = bit1_decode(parent, delta)
    err = np.linalg.norm(approx - target) / np.linalg.norm(target - parent)
    # 1-bit keeps the direction of every weight change; error stays bounded.
    assert err < 1.0


def test_shape_mismatch_rejected(pair):
    parent, _ = pair
    with pytest.raises(ValueError):
        xor_encode(parent, parent[:10])
    with pytest.raises(ValueError):
        bit1_encode(parent, parent[:10])
