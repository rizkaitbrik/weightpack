"""ROUTER train: a MIX's router, trained with its experts frozen.

The router keeps Arrow's form (mixing.py): a token's logit for expert i in
a module is |<w_i, x>| + b_i. Training starts from Arrow, so it can only
move away from it where the data says so:

- w_i = s_i * u_i, with u_i starting at Arrow's prototype and s_i at a
  scale that puts the module's largest logits near START_LOGIT on the
  first batch. Raw hidden states make Arrow's logits large, so its softmax
  is nearly one-hot and would pass almost no gradient; the scale leaves
  each token's ranking of experts, and so Arrow's top-1 routing, as it is.
- b_i starts at 0.

Only the router learns: the base and the experts stay frozen. The loss is
the language-modelling loss of the routed model on the data. A held-out
tenth of the data (at most MAX_HELDOUT examples) measures the loss with
Arrow and with the trained router, as served.

Data is JSON Lines, one example per line, in one of three shapes:

    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
    {"prompt": "...", "completion": "..."}
    {"text": "..."}

Messages go through the tokenizer's chat template and only the last
assistant turn is learnt; a prompt and completion are plain text and only
the completion is learnt; a text is learnt whole.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Where training puts each module's largest logits at the start.
START_LOGIT = 4.0
MAX_HELDOUT = 200
IGNORE = -100


@dataclass
class Options:
    steps: int = 2000
    batch_size: int = 4
    max_tokens: int = 512  # per example; longer ones are cut
    learning_rate: float = 1e-3
    seed: int = 0


@dataclass
class Result:
    path: str
    examples: int  # trained on
    heldout: int
    arrow_loss: float  # held-out loss with Arrow, as served
    trained_loss: float  # and with the trained router
    device: str
    seconds: float


def load_examples(path: str, tokenizer, max_tokens: int) -> list[tuple[list, list]]:
    """(input ids, labels) per example; labels are IGNORE where nothing is
    learnt."""
    out = []
    with open(path) as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                ids, labels = _encode(ex, tokenizer)
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"{path}:{no}: {e}") from None
            ids, labels = ids[:max_tokens], labels[:max_tokens]
            if any(lb != IGNORE for lb in labels[1:]):
                out.append((ids, labels))
    return out


def split(examples: list, seed: int) -> tuple[list, list]:
    """The held-out examples and the rest, shuffled by seed."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    n = max(1, min(MAX_HELDOUT, len(examples) // 10))
    return examples[:n], examples[n:]


def _encode(ex: dict, tok) -> tuple[list, list]:
    if "messages" in ex:
        msgs = ex["messages"]
        last = max(
            (i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=None
        )
        if last is None:
            raise ValueError("messages have no assistant turn to learn")

        def chat(m, gen):
            return tok.apply_chat_template(
                m, tokenize=True, add_generation_prompt=gen, return_dict=True
            )["input_ids"]

        prompt, full = chat(msgs[:last], True), chat(msgs[: last + 1], False)
        n = len(prompt) if full[: len(prompt)] == prompt else 0
        return full, [IGNORE] * n + full[n:]
    if "prompt" in ex and "completion" in ex:
        prompt = tok(ex["prompt"])["input_ids"]
        completion = tok(ex["completion"], add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None:
            completion = completion + [tok.eos_token_id]
        return prompt + completion, [IGNORE] * len(prompt) + completion
    if "text" in ex:
        ids = tok(ex["text"])["input_ids"]
        return ids, list(ids)
    raise ValueError('want "messages", "prompt" and "completion", or "text"')


def _batch(examples, pad_id, device):
    import torch

    n = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), n), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        mask[i, : len(x)] = 1
    return ids.to(device), mask.to(device), labels.to(device)


def _loss(model, batch, reduction="mean"):
    """The loss on the learnt tokens. Logits are computed only where some
    example learns the next token: a vocabulary of 150k makes the rest cost
    more than the model."""
    import torch

    ids, mask, labels = batch
    targets = labels[:, 1:]
    where = (targets != IGNORE).any(dim=0).nonzero().flatten()
    try:
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=where).logits
    except TypeError:  # a model without logits_to_keep
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1][:, where]
    targets = targets[:, where]
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE,
        reduction=reduction,
    )


def heldout_loss(engine, mix, examples, batch_size, pad_id) -> float:
    """Mean loss per learnt token over examples, with the mix as served."""
    import torch

    total, tokens = 0.0, 0
    with torch.no_grad(), engine._routing(mix):
        for i in range(0, len(examples), batch_size):
            batch = _batch(examples[i : i + batch_size], pad_id, engine.device)
            total += float(_loss(engine.peft, batch, reduction="sum"))
            tokens += int((batch[2][:, 1:] != IGNORE).sum())
    return total / max(tokens, 1)


@contextlib.contextmanager
def _training(engine):
    """Training mode with activation checkpointing: each block's activations
    are computed again in the backward pass instead of kept. Otherwise every
    expert's output in every routed module stays alive across the network,
    which is several times the model's own size. Dropout stays off, as when
    the mix is served."""
    import torch

    model = engine.peft.get_base_model()
    checkpointed = bool(getattr(model, "supports_gradient_checkpointing", False))
    # transformers 5.x renamed gradient_checkpointing_* to activation_*.
    renamed = hasattr(model, "activation_checkpointing_enable")
    if checkpointed:
        if renamed:
            model.activation_checkpointing_enable({"use_reentrant": False})
        else:
            model.gradient_checkpointing_enable({"use_reentrant": False})
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
    try:
        yield
    finally:
        if checkpointed:
            if renamed:
                model.activation_checkpointing_disable()
            else:
                model.gradient_checkpointing_disable()
            model.eval()


def _release(device: str) -> None:
    """Hands Metal's cached blocks back: batches of every length otherwise
    grow its cache step by step, past what the machine has."""
    import torch

    if device == "mps":
        torch.mps.empty_cache()


def _calibrate(engine, mix, batch) -> list[float]:
    """Each route's mean largest Arrow logit on a batch."""
    import torch

    from .mixing import router_logits

    seen: dict[int, list[float]] = {}
    handles = []

    def watch(i, route):
        def hook(_module, args, _output):
            x = args[0].reshape(-1, args[0].shape[-1])
            logits = router_logits(x, route.weight, route.bias)
            seen.setdefault(i, []).append(float(logits.max(dim=1).values.mean()))

        return hook

    try:
        for i, r in enumerate(mix.routes):
            handles.append(r.module.register_forward_hook(watch(i, r)))
        with torch.no_grad(), engine._routing(mix):
            engine.peft(input_ids=batch[0], attention_mask=batch[1])
    finally:
        for h in handles:
            h.remove()
    means = [seen.get(i, [1.0]) for i in range(len(mix.routes))]
    return [sum(m) / len(m) for m in means]


def train(
    base_dir: str,
    experts: list[tuple[str, str]],
    targets: list[str],
    blocks: str,
    top_k: int,
    data: str,
    out_file: str,
    options: Options | None = None,
    device: str = "auto",
    progress: Callable[[int, int, float], None] | None = None,
    chat_template: str | None = None,
) -> Result:
    """Trains a router for a MIX of experts ((id, adapter dir), in mix order)
    over the base in base_dir, and writes it to out_file. chat_template, a
    file, replaces the tokenizer's, as the image's TEMPLATE does."""
    import torch
    from safetensors.torch import save_file

    from .engine import Engine, pick_device
    from .mixing import Mix

    opts = options or Options()
    start = time.monotonic()
    device = pick_device(device)
    torch.manual_seed(opts.seed)
    # Float32 on the CPU, where half precision is slow; the checkpoint's
    # dtype on GPUs. The router itself is always float32.
    dtype = "float32" if device == "cpu" else "auto"
    engine = Engine(base_dir, device=device, dtype=dtype, chat_template=chat_template)
    for name, path in experts:
        engine.load_adapter(name, path)
    names = [name for name, _ in experts]
    mix = Mix(engine.peft, names, top_k, targets, blocks)
    for p in engine.peft.parameters():
        p.requires_grad_(False)

    tok = engine.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    examples = load_examples(data, tok, opts.max_tokens)
    if len(examples) < 2:
        raise ValueError(f"{data}: {len(examples)} usable examples; want at least 2")
    heldout, train_set = split(examples, opts.seed)

    arrow_loss = heldout_loss(engine, mix, heldout, opts.batch_size, pad_id)

    order = random.Random(opts.seed + 1)
    queue: list = []

    def next_batch():
        nonlocal queue
        if len(queue) < opts.batch_size:
            refill = list(train_set)
            order.shuffle(refill)
            queue += refill
        batch, queue = queue[: opts.batch_size], queue[opts.batch_size :]
        return _batch(batch, pad_id, engine.device)

    first = next_batch()
    scales = _calibrate(engine, mix, first)
    params = []
    for r, c in zip(mix.routes, scales, strict=True):
        n = len(r.experts)
        r.u = torch.nn.Parameter(r.weight.clone())
        scale = math.log(START_LOGIT / max(c, 1e-6))
        r.log_s = torch.nn.Parameter(torch.full((n,), scale, device=r.weight.device))
        r.b = torch.nn.Parameter(torch.zeros(n, device=r.weight.device))
        params += [r.u, r.log_s, r.b]
    optimizer = torch.optim.AdamW(params, lr=opts.learning_rate, weight_decay=0.0)

    def use_trained():
        for r in mix.routes:
            r.weight = r.log_s.exp()[:, None] * r.u
            r.bias = r.b

    report = math.ceil(opts.steps / 50)  # at most 50 reports
    smooth = None
    batch = first
    with engine._routing(mix), _training(engine):
        for r in mix.routes:
            r.training = True
        for step in range(1, opts.steps + 1):
            use_trained()
            loss = _loss(engine.peft, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            del loss
            _release(device)
            smooth = value if smooth is None else 0.9 * smooth + 0.1 * value
            if progress and (step % report == 0 or step == opts.steps):
                progress(step, opts.steps, smooth)
            if step < opts.steps:
                batch = next_batch()
        for r in mix.routes:
            r.training = False

    with torch.no_grad():
        for r in mix.routes:
            r.weight = (r.log_s.exp()[:, None] * r.u).detach()
            r.bias = r.b.detach()
            del r.u, r.log_s, r.b
    trained_loss = heldout_loss(engine, mix, heldout, opts.batch_size, pad_id)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    save_file(mix.router_tensors(), out_file)
    return Result(
        path=out_file,
        examples=len(train_set),
        heldout=len(heldout),
        arrow_loss=arrow_loss,
        trained_loss=trained_loss,
        device=device,
        seconds=time.monotonic() - start,
    )
