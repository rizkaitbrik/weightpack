"""Scale calibration for bit1 deltas, as BitDelta (Liu et al., 2024) does.

A bit1 delta keeps a matrix's signs and one scale. The mean absolute delta
is the scale that minimises that matrix's own squared error, not the error
the model makes with every matrix compressed at once. Calibration tunes the
scales, signs fixed, so that the compressed model's next-token distributions
match the fine-tune's: it distils the fine-tune into a few hundred numbers.

The data is the fine-tune's own greedy replies to CALIBRATION_PROMPTS,
general prompts kept apart from the eval's (evaluate.SELF_PROMPTS), so the
self suite never scores a model tuned on its own examples. The loss is the
KL divergence from the fine-tune's next-token distribution to the
compressed model's, on the reply tokens. Each scale trains as a log factor on
its mean absolute delta: it stays positive and moves at the same relative
rate whatever its size. If training does not lower the loss on the
calibration replies, the mean absolute deltas are kept.
"""

from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .delta import Parent, bits
from .evaluate import MAX_TOKENS, _load, self_suite
from .lossy import bit1_scale
from .router_train import IGNORE, _batch, _encode, learnt_logits
from .safetensors_io import FLOAT_DTYPES, storage_dtype, to_float32

# General prompts the fine-tune answers to make the calibration data: none
# of them is in evaluate.SELF_PROMPTS.
CALIBRATION_PROMPTS = [
    "Explain how vaccines train the immune system.",
    "Write a limerick about a cat who learns to code.",
    "What is the difference between weather and climate?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "A shop sells pens at 3 for $2. How much do 12 pens cost?",
    "Summarise the causes of the First World War in a paragraph.",
    "How do noise-cancelling headphones work?",
    "Give five ideas for a rainy-day activity with children.",
    "Write a bash command that counts the lines in every .txt file in a folder.",
    "Why do leaves change colour in autumn?",
    "Translate 'Good morning, how are you?' into German and Italian.",
    "If every square is a rectangle, is every rectangle a square? Explain.",
    "Describe how a bill becomes a law in a parliamentary system.",
    "What are the pros and cons of working from home?",
    "Write a Go function that reverses a slice of integers.",
    "Solve for y: 2y - 5 = 11.",
    "What was the Silk Road, and why did it matter?",
    "Suggest a name for a hiking club and explain the choice.",
    "How does public-key cryptography work?",
    "Explain gravity to a seven-year-old.",
    "List the steps to make a cup of tea.",
    "Why is the Dead Sea so salty?",
    "Write a short story opening set on a night train.",
    "Compare electric cars and petrol cars in a few sentences.",
    "What does a database index do?",
    "Explain what a recession is.",
    "Write a regular expression that matches a US ZIP code, and explain it.",
    "A recipe needs 250 g of flour for 10 cookies. How much for 24 cookies?",
    "What are the main parts of a cell?",
    "Give advice to someone preparing for a job interview.",
    "Write a SQL statement that adds a column to a table.",
    "Why do we have time zones?",
]


@dataclass
class Calibration:
    scales: dict[str, float] = field(default_factory=dict)  # by tensor name
    loss_before: float = 0.0  # mean KL per reply token, mean absolute deltas
    loss_after: float = 0.0  # and with the calibrated scales
    examples: int = 0
    device: str = ""
    seconds: float = 0.0


class _AddScaledSign(torch.autograd.Function):
    """parent + scale * sign, rounded once to the parent's dtype as decoding
    does. Saves only the int8 signs for the backward pass, which gives the
    scale its gradient and the parent none."""

    @staticmethod
    def forward(ctx, w, scale, sign):
        ctx.save_for_backward(sign)
        return (w.float() + scale * sign).to(w.dtype)

    @staticmethod
    def backward(ctx, grad):
        (sign,) = ctx.saved_tensors
        return None, (grad.float() * sign).sum(), None


class _Bit1(torch.nn.Module):
    """A parametrization giving a matrix its parent's weights plus a bit1
    delta whose scale trains."""

    def __init__(self, sign: torch.Tensor, scale: float):  # a float32 value
        super().__init__()
        self.register_buffer("sign", sign, persistent=False)
        self.scale0 = scale
        self.log_factor = torch.nn.Parameter(torch.zeros((), device=sign.device))

    def scale(self) -> torch.Tensor:
        return self.scale0 * torch.exp(self.log_factor)

    def forward(self, w):
        return _AddScaledSign.apply(w, self.scale(), self.sign)


def calibration_examples(model, tok, prompts=CALIBRATION_PROMPTS) -> list[tuple]:
    """(input ids, labels) of the model's greedy replies to prompts, the
    replies learnt."""
    encoded = []
    for ex in self_suite(model, tok, prompts=prompts).examples:
        ids, labels = _encode(ex, tok)
        ids, labels = ids[:MAX_TOKENS], labels[:MAX_TOKENS]
        if any(lb != IGNORE for lb in labels[1:]):
            encoded.append((ids, labels))
    if not encoded:
        raise ValueError("the fine-tune's calibration replies are all empty")
    return encoded


def _kl(teacher, student, batch):
    """The summed KL divergence from teacher to student over the batch's
    learnt tokens, and how many tokens that is."""
    with torch.no_grad():
        t, targets = learnt_logits(teacher, batch)
    s, _ = learnt_logits(student, batch)
    keep = targets != IGNORE
    loss = torch.nn.functional.kl_div(
        torch.log_softmax(s[keep], -1),
        torch.log_softmax(t[keep], -1),
        log_target=True,
        reduction="sum",
    )
    return loss, int(keep.sum())


def mean_kl(teacher, student, encoded, pad: int, batch_size: int = 4) -> float:
    """The mean KL divergence per learnt token from teacher to student."""
    total, tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(encoded), batch_size):
            batch = _batch(encoded[i : i + batch_size], pad, teacher.device)
            loss, n = _kl(teacher, student, batch)
            total, tokens = total + float(loss), tokens + n
    return total / max(1, tokens)


def _bit1_student(student, parent: Parent, target: dict, device) -> dict:
    """Turns the student, a copy of the fine-tune, into the bit1 model: each
    changed float matrix gets its parent's weights plus a trainable bit1
    delta. A matrix tied to others (embeddings and the output head) gets one
    delta, shared. Returns the deltas by tensor name."""
    student.requires_grad_(False)
    owners: dict[int, list] = {}
    for module in student.modules():
        for pname, p in list(module._parameters.items()):
            if p is not None:
                owners.setdefault(id(p), []).append((module, pname))
    params = dict(student.named_parameters(remove_duplicate=False))
    deltas = {}
    for name in sorted(target):
        path, info = target[name]
        p = params.get(name)
        if len(info.shape) < 2 or info.dtype not in FLOAT_DTYPES or p is None:
            continue
        old, new = parent.bits(name), bits(path, info)
        if np.array_equal(old, new):
            continue
        storage = storage_dtype(info.dtype)
        w = to_float32(info.dtype, old.view(storage))
        delta = to_float32(info.dtype, new.view(storage)) - w
        scale = bit1_scale(delta)
        if scale == 0:
            continue
        sign = torch.from_numpy(np.where(delta >= 0, 1, -1).astype(np.int8))
        with torch.no_grad():
            p.copy_(torch.from_numpy(w).reshape(p.shape).to(p.dtype))
        bit1 = _Bit1(sign.reshape(p.shape).to(device), scale)
        for module, pname in owners[id(p)]:
            torch.nn.utils.parametrize.register_parametrization(module, pname, bit1)
        deltas[name] = bit1
    return deltas


def calibrate_bit1(
    source_dir: str,
    parent: Parent,
    target: dict,
    steps: int = 200,
    chat_template: str | None = None,
    device: str = "auto",
    batch_size: int = 4,
    learning_rate: float = 0.01,
    seed: int = 0,
) -> Calibration:
    """Calibrated bit1 scales for a fine-tune over its parent.

    source_dir is the fine-tune as a model directory (weights, config,
    tokenizer); target its tensors by name, as delta.fine_tune gives them.
    Returns a scale for each changed float matrix the model has as a
    parameter; others keep their mean absolute delta.
    """
    from .engine import pick_device

    start = time.monotonic()
    device = pick_device(device)
    teacher, tok = _load(source_dir, device, chat_template)
    student, _ = _load(source_dir, device, chat_template)
    try:
        deltas = _bit1_student(student, parent, target, device)
        initial = {name: d.scale0 for name, d in deltas.items()}
        encoded = calibration_examples(teacher, tok)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        before = mean_kl(teacher, student, encoded, pad, batch_size)
        after, scales = before, initial
        if deltas and steps > 0:
            opt = torch.optim.Adam(
                [d.log_factor for d in deltas.values()], lr=learning_rate
            )
            order = random.Random(seed)
            for _ in range(steps):
                sample = order.sample(encoded, min(batch_size, len(encoded)))
                loss, n = _kl(teacher, student, _batch(sample, pad, device))
                opt.zero_grad()
                (loss / max(1, n)).backward()
                opt.step()
            trained = mean_kl(teacher, student, encoded, pad, batch_size)
            if trained < before:
                after = trained
                scales = {n: float(d.scale().detach()) for n, d in deltas.items()}
        return Calibration(
            scales, before, after, len(encoded), device, time.monotonic() - start
        )
    finally:
        del teacher, student
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
