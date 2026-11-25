"""ROUTER train on a tiny model: two experts, each the better model of its
own data, and a router that learns which tokens are whose.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import json
import shutil
import tempfile
from pathlib import Path

import grpc
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from weightpack.worker.v1 import worker_pb2, worker_pb2_grpc  # noqa: E402
from weightpack_worker import router_train  # noqa: E402
from weightpack_worker.engine import Engine  # noqa: E402
from weightpack_worker.mixing import ROUTER_BIAS, ROUTER_WEIGHT, load_router  # noqa: E402
from weightpack_worker.server import serve  # noqa: E402

from .test_engine import tiny_model  # noqa: E402

TARGETS = ["q", "v", "up", "down"]


def counting(step, n, seed, length=12):
    """Texts that count by step from random starts, mod 60."""
    gen = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, 60, (n,), generator=gen).tolist()
    return [" ".join(f"t{(s + step * i) % 60}" for i in range(length)) for s in starts]


def fine_tune(base, path, texts, tok, seed):
    """A LoRA expert trained a little on texts, so it knows them."""
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=list(MODULES))
    model = peft.get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ids = torch.tensor([tok(t)["input_ids"] for t in texts])
    for step in range(150):
        batch = ids[(step * 16) % len(ids) :][:16]
        loss = model(input_ids=batch, labels=batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.save_pretrained(path)
    return str(path)


MODULES = ("q_proj", "v_proj", "up_proj", "down_proj")


@pytest.fixture(scope="module")
def mix(tmp_path_factory):
    d = tmp_path_factory.mktemp("router")
    base, _, tok = tiny_model(d)
    up, down = counting(1, 64, 0), counting(-2, 64, 1)
    experts = [
        ("one", fine_tune(base, d / "one", up, tok, 0)),
        ("two", fine_tune(base, d / "two", down, tok, 1)),
    ]
    engine = Engine(str(base), device="cpu")
    for name, path in experts:
        engine.load_adapter(name, path)
    texts = counting(1, 40, 2) + counting(-2, 40, 3)
    data = d / "mix.jsonl"
    data.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    return d, base, experts, data, engine


@pytest.mark.parametrize("top_k", [1, 2])
def test_trained_router_beats_arrow(mix, top_k):
    d, base, experts, data, engine = mix
    out = d / f"router-{top_k}.safetensors"
    seen = []
    res = router_train.train(
        str(base),
        experts,
        TARGETS,
        "",
        top_k,
        str(data),
        str(out),
        router_train.Options(steps=150, batch_size=8, learning_rate=0.02),
        device="cpu",
        progress=lambda *a: seen.append(a),
    )
    assert res.examples + res.heldout == 80 and res.heldout == 8
    # The gain varies with the platform's float math (0.08 to 0.17 measured
    # on macOS and Linux at 150 steps), so the margin is modest.
    assert res.trained_loss < res.arrow_loss - 0.03, res
    assert seen[-1][:2] == (150, 150) and len(seen) == 50

    # One row per expert, in mix order, for every routed module.
    tensors = load_router(str(out))
    modules = {n.removesuffix(ROUTER_WEIGHT) for n in tensors}
    modules = {m for m in modules if not m.endswith(ROUTER_BIAS)}
    assert len(modules) == 2 * len(TARGETS) and len(tensors) == 2 * len(modules)
    for m in modules:
        in_features = 128 if m.endswith("down_proj") else 64
        assert tensors[m + ROUTER_WEIGHT].shape == (2, in_features)
        assert tensors[m + ROUTER_BIAS].shape == (2,)

    # Served from the file, the router gives the loss training measured.
    name = f"trained-{top_k}"
    engine.load_mix(name, ["one", "two"], top_k, TARGETS, "", "trained", str(out))
    heldout, _ = router_train.split(
        router_train.load_examples(str(data), engine.tokenizer, 512), 0
    )
    pad = engine.tokenizer.pad_token_id
    served = router_train.heldout_loss(engine, engine.mixes[name], heldout, 8, pad)
    assert served == pytest.approx(res.trained_loss, rel=1e-4)


def test_router_file_must_cover_the_mix(mix, tmp_path):
    d, base, experts, data, engine = mix
    from safetensors.torch import save_file

    bad = tmp_path / "bad.safetensors"
    q = "model.layers.0.self_attn.q_proj"
    save_file({q + ROUTER_WEIGHT: torch.zeros(2, 64)}, bad)
    with pytest.raises(ValueError, match="no weights for"):
        engine.load_mix("bad", ["one", "two"], 1, TARGETS, "", "trained", str(bad))
    with pytest.raises(ValueError, match="router file"):
        engine.load_mix("bad", ["one", "two"], 1, TARGETS, "", "trained")
    assert "bad" not in engine.mixes


def test_examples(tmp_path):
    _, _, tok = tiny_model(tmp_path)
    data = tmp_path / "d.jsonl"
    lines = [
        {
            "messages": [
                {"role": "user", "content": "t1 t2"},
                {"role": "assistant", "content": "t3 t4"},
            ]
        },
        {"prompt": "t5 t6", "completion": "t7"},
        {"text": "t8 t9 t10"},
        {"text": "t11"},  # nothing to learn after the first token
    ]
    data.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    ex = router_train.load_examples(str(data), tok, 512)
    assert len(ex) == 3
    (ids, labels), (pids, plabels), (tids, tlabels) = ex
    assert labels == [-100, -100] + ids[2:] and len(ids) == 4
    assert pids == tok("t5 t6 t7")["input_ids"] + [tok.eos_token_id]
    assert plabels == [-100, -100] + pids[2:]
    assert tlabels == tids
    assert router_train.load_examples(str(data), tok, 3)[0][0] == ids[:3]

    data.write_text('{"text": "t1 t2"}\n{"nope": 1}\n')
    with pytest.raises(ValueError, match=r"d\.jsonl:2"):
        router_train.load_examples(str(data), tok, 512)
    data.write_text('{"messages": [{"role": "user", "content": "t1"}]}\n')
    with pytest.raises(ValueError, match="no assistant turn"):
        router_train.load_examples(str(data), tok, 512)


@pytest.fixture
def worker():
    # A short directory: Unix socket paths are limited to about 100 bytes.
    d = Path(tempfile.mkdtemp(prefix="wpkw"))
    server = serve(d / "w.sock")
    channel = grpc.insecure_channel(f"unix:{d / 'w.sock'}")
    yield worker_pb2_grpc.WorkerStub(channel)
    channel.close()
    server.stop(grace=None)
    shutil.rmtree(d)


def test_train_router_over_grpc(mix, worker):
    d, base, experts, data, _ = mix
    client = worker
    req = worker_pb2.TrainRouterRequest(
        base_dir=str(base),
        experts=[worker_pb2.Adapter(id=i, dir=p) for i, p in experts],
        targets=TARGETS,
        top_k=1,
        data=str(data),
        steps=20,
        out_file=str(d / "grpc" / "router.safetensors"),
        device="cpu",
    )
    events = list(client.TrainRouter(req))
    *steps, last = events
    assert [e.step for e in steps] == list(range(1, 21))
    assert not any(e.done for e in steps)
    assert last.done and last.examples == 72 and last.heldout == 8
    assert last.trained_loss < last.arrow_loss and last.device == "cpu"
    assert (d / "grpc" / "router.safetensors").is_file()

    req.data = str(d / "missing.jsonl")
    with pytest.raises(grpc.RpcError) as err:
        list(client.TrainRouter(req))
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "missing.jsonl" in err.value.details()


def test_loss_reads_only_learnt_positions(mix):
    _, _, _, _, engine = mix
    tok = engine.tokenizer
    examples = [
        router_train._encode({"prompt": "t1 t2 t3", "completion": "t4 t5"}, tok),
        router_train._encode({"text": "t6 t7 t8 t9 t10 t11"}, tok),
    ]
    batch = router_train._batch(examples, tok.pad_token_id, engine.device)
    with torch.no_grad():
        got = router_train._loss(engine.model, batch, reduction="sum")
        logits = engine.model(input_ids=batch[0], attention_mask=batch[1]).logits
        want = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            batch[2][:, 1:].reshape(-1),
            ignore_index=router_train.IGNORE,
            reduction="sum",
        )
    assert torch.allclose(got, want, rtol=1e-5)
