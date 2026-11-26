"""The token-mixing gate of the roadmap (docs/design.md): a 3-expert mix
beats each single expert on a mixed-domain eval.

Qwen2.5-0.5B-Instruct and the smoke test's three public LoRA experts
(python, sql, math), mixed in every projection with top_k 1. `wpk build`
trains the router (ROUTER train) on public data from the three domains;
`wpk export` hands it out. Held-out examples of each domain are then scored
by mean loss per answer token under the base, each expert alone, the mix
with Arrow and the mix with the trained router.

Opt-in: it downloads the models the smoke test uses (about 1 GB) and about
55 MB of data, cached in <repo>/tmp/smoke, and trains for a while:

    npx nx run worker:gate

WPK_GATE_STEPS (default 600) and WPK_GATE_TRAIN (examples per domain,
default 400) size the training.
"""

import json
import os
import random
import urllib.request
from pathlib import Path

import pytest

from .e2e import REPO, Weightpack, binaries
from .test_smoke_real_model import models  # noqa: F401 (a fixture)

pytestmark = pytest.mark.skipif(
    os.environ.get("WPK_GATE") != "1",
    reason="set WPK_GATE=1 (npx nx run worker:gate)",
)
torch = pytest.importorskip("torch")
hub = pytest.importorskip("huggingface_hub")

STEPS = int(os.environ.get("WPK_GATE_STEPS", "600"))
TRAIN = int(os.environ.get("WPK_GATE_TRAIN", "400"))
EVAL = 50
DATA = REPO / "tmp" / "smoke" / "data"
DOMAINS = ["python", "sql", "math"]

# Pinned so the test sees the same bytes every time.
GSM8K = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/{}.jsonl"
SQL = (
    "b-mc2/sql-create-context",
    "9d80a6a118b838d9defc3798d659a54a2ac2ff37",
    "sql_create_context_v4.json",
)
PYTHON = (
    "flytech/python-codes-25k",
    "0ed98ff2a76c5d133d8c157b814189a5a17ebd20",
    "python-codes-25k.jsonl",
)


def chat(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def hub_file(repo, rev, name) -> Path:
    return Path(
        hub.hf_hub_download(
            repo,
            name,
            repo_type="dataset",
            revision=rev,
            local_dir=DATA / repo.split("/")[1],
        )
    )


def gsm8k(split: str) -> list[dict]:
    path = DATA / f"gsm8k-{split}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GSM8K.format(split), path)  # noqa: S310 (a pinned https URL)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [chat(r["question"], r["answer"]) for r in rows]


def sql() -> list[dict]:
    rows = json.loads(hub_file(*SQL).read_text())
    return [
        chat(f"Given these tables:\n{r['context']}\n\n{r['question']}", r["answer"])
        for r in rows
    ]


def python() -> list[dict]:
    rows = [
        json.loads(line) for line in hub_file(*PYTHON).read_text().splitlines() if line
    ]
    return [
        chat(
            r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else ""),
            r["output"],
        )
        for r in rows
    ]


def splits() -> tuple[list[dict], dict[str, list[dict]]]:
    """A shuffled mix of TRAIN examples per domain, and EVAL held-out
    examples per domain that training never sees."""
    train, held = [], {}
    for name, rows in (("python", python()), ("sql", sql())):
        random.Random(0).shuffle(rows)
        held[name], train_rows = rows[:EVAL], rows[EVAL : EVAL + TRAIN]
        train += train_rows
    maths = gsm8k("train")
    random.Random(0).shuffle(maths)
    train += maths[:TRAIN]
    test = gsm8k("test")
    random.Random(0).shuffle(test)
    held["math"] = test[:EVAL]
    random.Random(1).shuffle(train)
    return train, held


def mean_loss(engine, name, examples, batch=8) -> float:
    """Mean loss per answer token of a model the engine serves."""
    from weightpack_worker import router_train

    pad = engine.tokenizer.pad_token_id or engine.tokenizer.eos_token_id
    encoded = [
        tuple(x[:512] for x in router_train._encode(ex, engine.tokenizer))
        for ex in examples
    ]
    model, ctx = engine._model_for(name)
    total, tokens = 0.0, 0
    with ctx, torch.no_grad():
        for i in range(0, len(encoded), batch):
            b = router_train._batch(encoded[i : i + batch], pad, engine.device)
            total += float(router_train._loss(model, b, reduction="sum"))
            tokens += int((b[2][:, 1:] != router_train.IGNORE).sum())
    return total / tokens


def routing_shares(engine, name, examples) -> list[float]:
    """The share of routing weight each expert gets over a domain's tokens,
    averaged over the mix's modules."""
    mix = engine.mixes[name]
    sums = torch.zeros(len(mix.experts))
    originals = {}
    for r in mix.routes:
        originals[r] = r.coefficients

        def wrapped(tokens, top_k, r=r):
            c = originals[r](tokens, top_k)
            for j, e in enumerate(r.experts):
                sums[mix.experts.index(e)] += float(c[:, j].sum())
            return c

        r.coefficients = wrapped
    try:
        mean_loss(engine, name, examples)
    finally:
        for r, f in originals.items():
            r.coefficients = f
    return (sums / sums.sum()).tolist()


def test_trained_router_beats_each_expert(models, tmp_path):  # noqa: F811
    from weightpack_worker.engine import Engine

    bins = binaries()
    if bins is None:
        pytest.skip("wpk/wpkd not built")
    train, held = splits()
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(ex) + "\n" for ex in train)
    )
    experts = "".join(f"  EXPERT {d} {models[d]}\n" for d in DOMAINS)
    (tmp_path / "Weightfile").write_text(
        f"FROM {models['base']}\n"
        "MIX domains targets=q,k,v,o,gate,up,down top_k=1\n"
        f"{experts}"
        f"  ROUTER train data=./train.jsonl steps={STEPS}\n"
    )
    w = Weightpack(*bins)
    try:
        out = w.wpk("build", "-t", "gate", str(tmp_path))
        print(out, flush=True)
        export = tmp_path / "export"
        w.wpk("export", "-o", str(export), "gate")
    finally:
        w.close()
    router = export / "routers" / "domains.safetensors"
    assert router.is_file()

    engine = Engine(models["base"])
    for d in DOMAINS:
        engine.load_adapter(d, models[d])
    engine.load_mix("arrow", DOMAINS, 1, ["q", "k", "v", "o", "gate", "up", "down"])
    engine.load_mix(
        "trained",
        DOMAINS,
        1,
        ["q", "k", "v", "o", "gate", "up", "down"],
        "",
        "trained",
        str(router),
    )
    names = ["base", *DOMAINS, "arrow", "trained"]
    losses = {n: {d: mean_loss(engine, n, held[d]) for d in DOMAINS} for n in names}
    shares = {
        n: {d: routing_shares(engine, n, held[d]) for d in DOMAINS}
        for n in ("arrow", "trained")
    }

    print(
        f"\nheld-out loss per answer token, {EVAL} examples per domain "
        f"(router trained on {len(train)} examples for {STEPS} steps)"
    )
    print(f"{'model':>8} " + " ".join(f"{d:>7}" for d in DOMAINS) + "    mean")
    means = {}
    for n in names:
        means[n] = sum(losses[n].values()) / len(DOMAINS)
        print(
            f"{n:>8} "
            + " ".join(f"{losses[n][d]:7.4f}" for d in DOMAINS)
            + f"  {means[n]:7.4f}"
        )
    for n in ("arrow", "trained"):
        print(f"\nrouting weight per expert ({', '.join(DOMAINS)}) under {n}:")
        for d in DOMAINS:
            print(f"  {d:>6} prompts: " + " ".join(f"{s:5.1%}" for s in shares[n][d]))

    best = min(means[d] for d in DOMAINS)
    assert means["trained"] < best, (
        f"the trained mix ({means['trained']:.4f}) does not beat the best "
        f"expert ({best:.4f})"
    )
