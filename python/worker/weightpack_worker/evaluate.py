"""Fidelity evals: what a lossy layer costs, measured against its source.

An eval suite holds examples in the shapes ROUTER train reads
(router_train): JSON Lines of ``{"messages": [...]}``, ``{"prompt",
"completion"}`` or ``{"text"}``. A suite file is either such JSON Lines,
named after the file, or YAML that names them::

    name: finance-v2
    data: finance.jsonl   # relative to the YAML file

A model's score on a suite is its next-token accuracy on the learnt tokens
(the last assistant turn, the completion, or the whole text), teacher-forced,
in percent: how often its most likely next token is the example's. A lossy
layer's drop is the source fine-tune's score minus the score of the model
the layer gives back, in percentage points.

Without a suite, the source writes one (SELF): its greedy replies to a fixed
set of general prompts. The source scores about 100 on those, so the drop is
about the share of reply tokens on which the lossy model's first choice
differs from the source's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SELF = "self"
MAX_TOKENS = 512

# General prompts the source answers to make the SELF suite: writing, code,
# maths, facts and reasoning, so no one skill decides the drop.
SELF_PROMPTS = [
    "Explain how a refrigerator keeps food cold.",
    "Write a short poem about the sea at night.",
    "What are the main causes of inflation?",
    "Write a Python function that checks whether a string is a palindrome.",
    "A train travels 120 km in 1.5 hours. What is its average speed?",
    "Summarise the plot of Romeo and Juliet in three sentences.",
    "What is the difference between a virus and a bacterium?",
    "Give three tips for writing a clear email.",
    "Write a SQL query that finds the ten most recent orders.",
    "Why is the sky blue?",
    "Translate 'Where is the train station?' into French and Spanish.",
    "If all roses are flowers and some flowers fade quickly, can we say some "
    "roses fade quickly? Explain.",
    "Describe the water cycle.",
    "What should I consider when choosing a laptop for programming?",
    "Write a JavaScript function that removes duplicates from an array.",
    "Solve for x: 3x + 7 = 25.",
    "What were the main effects of the printing press on Europe?",
    "Suggest a name for a bakery and explain why.",
    "How does a hash table work?",
    "Explain photosynthesis to a ten-year-old.",
    "List the planets of the solar system in order from the Sun.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Write a haiku about autumn.",
    "Compare renting and buying a home in a few sentences.",
]


@dataclass
class Suite:
    name: str
    examples: list[dict]


@dataclass
class Score:
    accuracy: float  # percent of learnt tokens predicted
    loss: float  # mean loss per learnt token


@dataclass
class Result:
    suite: str
    examples: int
    tokens: int
    scores: list[Score] = field(default_factory=list)


def load_suite(path: str) -> Suite:
    """A suite file: JSON Lines of examples, or YAML naming them."""
    p = Path(path)
    name, data = p.stem, p
    if p.suffix in (".yaml", ".yml"):
        import yaml

        spec = yaml.safe_load(p.read_text()) or {}
        if not isinstance(spec, dict) or "data" not in spec:
            raise ValueError(f"{path}: want a mapping with data: <examples.jsonl>")
        name = str(spec.get("name") or p.stem)
        data = (p.parent / str(spec["data"])).resolve()
    examples = []
    with open(data) as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{data}:{no}: {e}") from None
    if not examples:
        raise ValueError(f"{data}: no examples")
    return Suite(name, examples)


def _load(model_dir: str, device: str, chat_template: str | None):
    import torch
    import transformers

    dtype = torch.float32 if device == "cpu" else "auto"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    model.to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    if chat_template:
        tok.chat_template = Path(chat_template).read_text()
    return model, tok


def greedy(model, ids: list[int], max_new_tokens: int, eos: set[int]) -> list[int]:
    """The model's first choice at each step, until an end token. Written
    out rather than through generate(), which merges in a checkpoint's own
    generation settings (Qwen's add a repetition penalty), so its replies
    would not be the model's first choices."""
    import torch

    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=True)
    past, nxt, tokens = out.past_key_values, out.logits[0, -1].argmax(), []
    for _ in range(max_new_tokens):
        t = int(nxt)
        if t in eos:
            break
        tokens.append(t)
        out = model(input_ids=nxt.view(1, 1), past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[0, -1].argmax()
    return tokens


def self_suite(model, tok, max_new_tokens: int = 128, prompts=SELF_PROMPTS) -> Suite:
    """The source's own greedy replies to prompts, SELF_PROMPTS by default."""
    import torch

    chat = bool(getattr(tok, "chat_template", None))
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    examples = []
    for prompt in prompts:
        if chat:
            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_dict=True
            )["input_ids"]
        else:
            ids = tok(prompt)["input_ids"]
        with torch.no_grad():
            reply = tok.decode(greedy(model, ids, max_new_tokens, eos))
        if chat:
            examples.append(
                {"messages": [*msgs, {"role": "assistant", "content": reply}]}
            )
        else:
            examples.append({"prompt": prompt, "completion": reply})
    return Suite(SELF, examples)


def score(model, tok, examples: list[dict], batch_size: int = 8) -> tuple[Score, int]:
    """A model's next-token accuracy and loss on the examples' learnt
    tokens, and how many tokens that is."""
    import torch

    from .router_train import IGNORE, _batch, _encode, learnt_logits

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encoded = []
    for ex in examples:
        ids, labels = _encode(ex, tok)
        ids, labels = ids[:MAX_TOKENS], labels[:MAX_TOKENS]
        if any(lb != IGNORE for lb in labels[1:]):
            encoded.append((ids, labels))
    if not encoded:
        raise ValueError("no example has a token to score")
    correct, tokens, loss = 0, 0, 0.0
    with torch.no_grad():
        for i in range(0, len(encoded), batch_size):
            batch = _batch(encoded[i : i + batch_size], pad, model.device)
            logits, targets = learnt_logits(model, batch)
            keep = targets != IGNORE
            correct += int((logits.argmax(-1)[keep] == targets[keep]).sum())
            tokens += int(keep.sum())
            loss += float(
                torch.nn.functional.cross_entropy(
                    logits, targets, ignore_index=IGNORE, reduction="sum"
                )
            )
    return Score(100.0 * correct / tokens, loss / tokens), tokens


def evaluate(
    models: list[str],
    suite: str = "",
    device: str = "auto",
    chat_template: str | None = None,
) -> Result:
    """Scores each model directory on a suite; without one, on the first
    model's SELF suite."""
    if not models:
        raise ValueError("no model to evaluate")
    import gc

    import torch

    from .engine import pick_device

    device = pick_device(device)
    the_suite = load_suite(suite) if suite else None
    result = None
    for model_dir in models:
        model, tok = _load(model_dir, device, chat_template)
        if the_suite is None:
            the_suite = self_suite(model, tok)
        s, tokens = score(model, tok, the_suite.examples)
        if result is None:
            result = Result(the_suite.name, len(the_suite.examples), tokens)
        result.scores.append(s)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    return result
