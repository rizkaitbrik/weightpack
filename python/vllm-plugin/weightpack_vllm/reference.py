"""Reference forward pass for a mix of LoRA experts.

For a targeted linear module the design defines::

    y = W x + sum_{i in TopK(g(x))} g_i(x) * B_i A_i x

The base product ``W x`` is computed once per token whatever the number of
experts. This NumPy version is the ground truth the Triton/CUDA kernels in
the vLLM plugin are tested against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoRAExpert:
    """A LoRA expert: ``delta = scaling * B @ A``."""

    a: np.ndarray  # (rank, in_features)
    b: np.ndarray  # (out_features, rank)
    scaling: float = 1.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.scaling * (x @ self.a.T) @ self.b.T


def top_k_gates(logits: np.ndarray, k: int) -> np.ndarray:
    """Softmax over the top-k logits per token; zeros elsewhere.

    ``logits`` has shape (tokens, experts).
    """
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError(f"k={k} out of range for {logits.shape[-1]} experts")
    idx = np.argpartition(-logits, k - 1, axis=-1)[..., :k]
    picked = np.take_along_axis(logits, idx, axis=-1)
    picked = np.exp(picked - picked.max(axis=-1, keepdims=True))
    picked /= picked.sum(axis=-1, keepdims=True)
    gates = np.zeros_like(logits, dtype=np.float32)
    np.put_along_axis(gates, idx, picked, axis=-1)
    return gates


def mix_forward(
    x: np.ndarray,
    base_weight: np.ndarray,
    experts: list[LoRAExpert],
    gates: np.ndarray,
) -> np.ndarray:
    """Compute the mixed output for ``x`` of shape (tokens, in_features).

    ``gates`` has shape (tokens, experts); zero entries skip that expert.
    """
    if gates.shape != (x.shape[0], len(experts)):
        raise ValueError(f"gates shape {gates.shape} != {(x.shape[0], len(experts))}")
    y = x @ base_weight.T
    for i, expert in enumerate(experts):
        rows = np.nonzero(gates[:, i])[0]
        if rows.size:
            y[rows] += gates[rows, i : i + 1] * expert.apply(x[rows])
    return y
