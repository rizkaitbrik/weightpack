"""The bit1 linear kernel on Qwen2.5-7B's projections, beside the base's own
product, with caches as cold as in a model step: each call reads its own
copy of the signs (and the base its own copy of the weights), the copies
well past any L2. Inside CUDA graphs, four slots, bfloat16.

    python bench_bit1_kernels.py

Prints each projection's time for decode batches (all tokens of one
fine-tune, or of four in turn) and prefills, and per decode step (28 layers
and the head) the base's products, the deltas, and the time the sign bytes
alone take to read. Needs an NVIDIA GPU.
"""

from __future__ import annotations

import time
from functools import partial

import torch

from weightpack_vllm import bit1_kernel

SLOTS = 4
COLD = 256 << 20  # bytes of copies per call site
SHAPES = {  # name: (rows, inputs)
    "qkv": (4608, 3584),
    "o": (3584, 3584),
    "gate_up": (37888, 3584),
    "down": (3584, 18944),
    "lm_head": (152064, 3584),
}
BATCHES = [
    (1, "one"),
    (4, "mixed"),
    (32, "one"),
    (32, "mixed"),
    (64, "mixed"),
    (128, "mixed"),
    (2048, "runs"),
]


def bandwidth() -> float:
    a = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    b.copy_(a)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    return 2 * a.numel() * 5 / (time.perf_counter() - start)


def ms(calls, reps=10) -> float:
    """Milliseconds a call, for calls each on their own data, in one graph."""
    for f in calls:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:
            f()
    graph.replay()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps / len(calls) * 1e3


def main() -> None:
    bw = bandwidth()
    print(f"{torch.cuda.get_device_name(0)}, copies at {bw / 1e9:.0f} GB/s")
    step: dict = {}
    for name, (n, k) in SHAPES.items():
        kb = (k + 7) // 8
        signs = [
            torch.randint(0, 256, (SLOTS, n, kb), dtype=torch.uint8, device="cuda")
            for _ in range(max(2, -(-COLD // (SLOTS * n * kb))))
        ]
        weights = [
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            for _ in range(max(2, -(-COLD // (n * k * 2))))
        ]
        scale = torch.rand(SLOTS, n, device="cuda") * 1e-2
        active = torch.ones(SLOTS, dtype=torch.int32, device="cuda")
        for t, how in BATCHES:
            if name == "lm_head" and t > 128:
                continue  # the head sees one row per sequence
            x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
            y = torch.zeros(t, n, device="cuda", dtype=torch.bfloat16)
            slots = {
                "one": torch.zeros(t, dtype=torch.long, device="cuda"),
                "mixed": torch.arange(t, device="cuda") % SLOTS,
                "runs": torch.arange(t, device="cuda") // (t // SLOTS),
            }[how]
            reps = 3 if t > 128 else 10
            base = ms([partial(torch.matmul, x, w.T) for w in weights], reps)
            linear = partial(bit1_kernel.linear_, y, x, slots)
            delta = ms([partial(linear, s, scale, active) for s in signs], reps)
            read = (1 if how == "one" else min(t, SLOTS)) * n * kb / bw * 1e3
            print(
                f"{name:8s} {t:5d} tokens, {how:5s}: base {base:.3f} ms, "
                f"delta {delta:.3f} ms, signs read in {read:.3f} ms"
            )
            if t <= 128:
                acc = step.setdefault((t, how), [0.0, 0.0, 0.0])
                times = 1 if name == "lm_head" else 28
                for i, v in enumerate((base, delta, read)):
                    acc[i] += v * times
        del signs, weights
        torch.cuda.empty_cache()
    print("per decode step, 28 layers and the head:")
    for (t, how), (base, delta, read) in step.items():
        print(
            f"  {t:3d} tokens, {how:5s}: base {base:.2f} ms, delta {delta:.2f} ms, "
            f"signs read in {read:.2f} ms"
        )


if __name__ == "__main__":
    main()
