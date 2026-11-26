"""The delta-import gate of the roadmap (docs/design.md): a bit1 import stays
within max_drop on its eval.

Qwen2.5-0.5B-Instruct, fully fine-tuned here on GSM8K (every weight, a few
hundred steps), is imported over the base by `wpk build` as bit1 and as
sparse deltas, each gated on held-out GSM8K problems and on the fine-tune's
own replies to general prompts. The test prints each layer's size and drop.

Opt-in, and meant for a GPU: it downloads the model the smoke test uses and
about 5 MB of GSM8K, both cached in <repo>/tmp/smoke, and trains:

    npx nx run worker:import-gate

WPK_IMPORT_GATE_STEPS (default 300) sizes the fine-tune; WPK_IMPORT_GATE_MAX
(default 1.0) is max_drop, in points of next-token accuracy.
"""

import json
import os
import random
import shutil

import pytest

from .e2e import Weightpack, binaries
from .test_router_gate import gsm8k
from .test_smoke_real_model import models  # noqa: F401 (a fixture)

pytestmark = pytest.mark.skipif(
    os.environ.get("WPK_IMPORT_GATE") != "1",
    reason="set WPK_IMPORT_GATE=1 (npx nx run worker:import-gate)",
)
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

STEPS = int(os.environ.get("WPK_IMPORT_GATE_STEPS", "300"))
MAX_DROP = float(os.environ.get("WPK_IMPORT_GATE_MAX", "1.0"))
EVAL = 100


def full_finetune(base: str, out, examples, steps: int):
    """Every weight trained on examples, saved in the base's dtype with the
    base's own config and tokenizer files (the fine-tune changes neither)."""
    from weightpack_worker.router_train import IGNORE, _batch, _encode, learnt_logits

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(base)
    model = transformers.AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.to(device).train()
    model.gradient_checkpointing_enable()
    encoded = []
    for ex in examples:
        ids, labels = _encode(ex, tok)
        encoded.append((ids[:512], labels[:512]))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    order = random.Random(0)
    pad = tok.pad_token_id or tok.eos_token_id
    for step in range(steps):
        batch = _batch(order.sample(encoded, 8), pad, device)
        cuda = device == "cuda"
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=cuda):
            # Logits only where an answer token is learnt: all of them, over
            # a 150k vocabulary, would take gigabytes.
            logits, targets = learnt_logits(model, batch)
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"fine-tune step {step}: loss {float(loss):.4f}", flush=True)
    model.to(torch.bfloat16).save_pretrained(out)
    for f in os.listdir(base):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(base, f)):
            shutil.copyfile(os.path.join(base, f), os.path.join(out, f))
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()


def test_lossy_imports_stay_within_max_drop(models, tmp_path):  # noqa: F811
    bins = binaries()
    if bins is None:
        pytest.skip("wpk/wpkd not built")
    train = gsm8k("train")
    random.Random(0).shuffle(train)
    test = gsm8k("test")
    random.Random(0).shuffle(test)
    ft = tmp_path / "math"
    full_finetune(models["base"], ft, train[:2000], STEPS)
    (tmp_path / "gsm8k.jsonl").write_text(
        "".join(json.dumps(ex) + "\n" for ex in test[:EVAL])
    )

    suite = tmp_path / "gsm8k.jsonl"
    # bit1 calibrates its scales by default; the -raw layers keep the mean
    # absolute deltas, for comparison.
    builds = {
        "bit1": f"codec=bit1 eval={suite} max_drop={MAX_DROP}",
        "bit1-raw": f"codec=bit1 calibrate=0 eval={suite} max_drop={MAX_DROP}",
        "sparse": f"codec=sparse eval={suite} max_drop={MAX_DROP}",
        "bit1-self": f"codec=bit1 max_drop={MAX_DROP}",
        "bit1-raw-self": f"codec=bit1 calibrate=0 max_drop={MAX_DROP}",
        "sparse-self": f"codec=sparse max_drop={MAX_DROP}",
        "xor": "codec=xor",
    }
    w = Weightpack(*bins)
    results = {}
    try:
        for name, params in builds.items():
            d = tmp_path / name
            d.mkdir()
            (d / "Weightfile").write_text(
                f"FROM {models['base']}\nLAYER math {ft} {params}\n"
            )
            r = w.run("build", "-t", name, str(d))
            print(f"\n== {name}: exit {r.returncode}\n{r.stdout}{r.stderr}", flush=True)
            if r.returncode != 0:
                results[name] = None
                continue
            detail = w.inspect(name)
            layer = next(lr for lr in detail["config"]["layers"] if lr["id"] == "math")
            size = sum(
                d["size"]
                for d in detail["manifest"]["layers"]
                if d["annotations"].get("wpk.layer") == "math"
            )
            results[name] = (layer.get("fidelity"), size)
    finally:
        w.close()

    base_size = sum(
        os.path.getsize(os.path.join(models["base"], f))
        for f in os.listdir(models["base"])
        if f.endswith(".safetensors")
    )
    mb = base_size / 1e6
    print(f"\nfull fine-tune of {STEPS} steps; the base's weights are {mb:.0f} MB")
    head = ("layer", "size", "of base", "source", "layer", "drop")
    print("{:>14} {:>10} {:>8} {:>7} {:>7} {:>6}  suite".format(*head))
    for name, res in results.items():
        if res is None:
            print(f"{name:>14}  build failed")
            continue
        fid, size = res
        if fid is None:
            print(f"{name:>14} {size / 1e6:8.1f}MB {size / base_size:8.1%}  (exact)")
            continue
        src, got, drop = fid["source_score"], fid["score"], fid["measured_drop"]
        print(
            f"{name:>14} {size / 1e6:8.1f}MB {size / base_size:8.1%}"
            f" {src:7.2f} {got:7.2f} {drop:6.2f}  {fid['eval']}"
        )
    assert results["bit1"] is not None, "the bit1 import failed its gate"
    assert results["bit1"][0]["measured_drop"] <= MAX_DROP
