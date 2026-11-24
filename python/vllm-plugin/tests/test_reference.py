"""Tests for the reference mix forward pass."""

import numpy as np
import pytest

import weightpack_vllm
from weightpack_vllm.reference import LoRAExpert, mix_forward, top_k_gates


def make_experts(rng, n, d_in, d_out, rank):
    return [
        LoRAExpert(
            a=rng.standard_normal((rank, d_in)).astype(np.float32),
            b=rng.standard_normal((d_out, rank)).astype(np.float32),
            scaling=0.5,
        )
        for _ in range(n)
    ]


def test_mix_matches_dense_merge_per_token():
    rng = np.random.default_rng(1)
    tokens, d_in, d_out = 5, 16, 12
    x = rng.standard_normal((tokens, d_in)).astype(np.float32)
    w = rng.standard_normal((d_out, d_in)).astype(np.float32)
    experts = make_experts(rng, 3, d_in, d_out, rank=4)
    gates = top_k_gates(rng.standard_normal((tokens, 3)).astype(np.float32), k=2)

    y = mix_forward(x, w, experts, gates)

    # Ground truth: merge the gated deltas into dense weights for each token.
    for t in range(tokens):
        dense = w.copy()
        for i, e in enumerate(experts):
            dense += gates[t, i] * e.scaling * (e.b @ e.a)
        np.testing.assert_allclose(y[t], dense @ x[t], rtol=1e-4, atol=1e-4)


def test_top_k_gates_sum_to_one_and_keep_k():
    gates = top_k_gates(np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]]), k=2)
    np.testing.assert_allclose(gates.sum(axis=1), 1.0, rtol=1e-6)
    assert (np.count_nonzero(gates, axis=1) == 2).all()
    assert gates[0, 1] == 0 and gates[1, 0] == 0


def test_bad_k_rejected():
    with pytest.raises(ValueError):
        top_k_gates(np.zeros((1, 3)), k=4)


def test_register_is_callable():
    weightpack_vllm.register()
