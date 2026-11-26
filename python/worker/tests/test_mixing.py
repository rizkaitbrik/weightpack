"""Token-level mixing: Arrow routing over LoRA experts in the reference
engine, checked against PEFT's own Arrow implementation and against plain
adapter arithmetic.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import json
import urllib.error

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from weightpack_worker.engine import Engine, serve  # noqa: E402
from weightpack_worker.mixing import (  # noqa: E402
    arrow_coefficients,
    arrow_prototype,
    in_domain,
)

from .test_engine import PROMPT, get, post, tiny_model  # noqa: E402

ALL = ["q", "k", "v", "o", "gate", "up", "down"]


def adapter(base, path, seed, r=4, targets=("q_proj", "v_proj"), saved=None):
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    cfg = peft.LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=list(targets),
        init_lora_weights=False,
        modules_to_save=saved,
    )
    peft.get_peft_model(model, cfg).save_pretrained(path)
    return path


def logits(engine, name, ids):
    model, ctx = engine._model_for(name)
    with ctx, torch.no_grad():
        return model(input_ids=ids).logits


def prompt_ids(tok):
    out = tok.apply_chat_template(PROMPT, add_generation_prompt=True, return_dict=True)
    return torch.tensor([list(out["input_ids"])])


def test_prototype_is_the_top_right_singular_vector():
    torch.manual_seed(0)
    a, b = torch.randn(4, 32), torch.randn(48, 4)
    v = arrow_prototype(a, b)
    _, _, vh = torch.linalg.svd(b @ a)
    assert torch.isclose(v.norm(), torch.tensor(1.0))
    assert abs(float(v @ vh[0])) == pytest.approx(1.0, abs=1e-5)
    # Signs are arbitrary, and the routing only reads |x . v|.
    assert abs(float(arrow_prototype(a, -b) @ v)) == pytest.approx(1.0, abs=1e-5)


def test_coefficients_pick_the_top_k():
    protos = torch.eye(3)
    x = torch.tensor([[3.0, -1.0, 0.5], [0.0, 0.0, -2.0]])
    c = arrow_coefficients(x, protos, top_k=2)
    assert torch.allclose(c.sum(dim=1), torch.ones(2))
    assert c[0, 2] == 0 and c[0, 0] > c[0, 1] > 0  # |3| > |-1| > |0.5|
    assert c[1, 2] > 0.5
    assert torch.equal(
        arrow_coefficients(x, protos, top_k=1).argmax(1), torch.tensor([0, 2])
    )


@pytest.mark.parametrize("n", [2, 3, 8])
def test_ties_go_to_the_first_expert(n):
    # Experts with the same prototype (same A, B scaled) tie for every token.
    protos = torch.ones(n, 4) / 2
    x = torch.randn(5, 4)
    c = arrow_coefficients(x, protos, top_k=1)
    assert torch.equal(c, torch.eye(n)[[0] * 5])
    c = arrow_coefficients(x, protos, top_k=2)
    assert torch.allclose(c[:, :2], torch.full((5, 2), 0.5)) and (c[:, 2:] == 0).all()


def test_domain():
    m = "base_model.model.model.layers.3.mlp.up_proj"
    assert in_domain(m, ["up"], "")
    assert in_domain(m, ["gate", "up", "down"], "0-1,3")
    assert not in_domain(m, ["up"], "0-2,4-9")
    assert not in_domain(m, ["q", "v"], "")
    assert not in_domain("base_model.model.lm_head", ["lm_head"], "0-31")


@pytest.mark.parametrize("top_k", [1, 2])
def test_mix_matches_peft_arrow(tmp_path, top_k):
    base, one, tok = tiny_model(tmp_path)
    two = adapter(base, tmp_path / "two", seed=1)
    engine = Engine(str(base), device="cpu", served_name="tiny")
    engine.load_adapter("one", str(one))
    engine.load_adapter("two", str(two))
    engine.load_mix("mix", ["one", "two"], top_k, ALL)
    ids = prompt_ids(tok)
    ours = logits(engine, "mix", ids)

    # PEFT's Arrow over the same experts, with the same (exact) prototypes:
    # it power-iterates from a random start, which only approximates them.
    ref = peft.create_arrow_model(
        transformers.AutoModelForCausalLM.from_pretrained(base),
        [str(one), str(two)],
        peft.ArrowConfig(top_k=top_k, router_temperature=1.0),
    ).eval()
    routed = 0
    for _, module in ref.named_modules():
        if hasattr(module, "lora_arrow"):
            layer = module.lora_arrow["arrow_router"]
            protos = torch.stack(
                [
                    arrow_prototype(module.lora_A[n].weight, module.lora_B[n].weight)
                    for n in layer.task_adapter_names
                ]
            )
            layer.register_buffer("prototypes", protos, persistent=False)
            layer._protos_ready = True
            routed += 1
    assert routed == len(engine.mixes["mix"].routes) == 4  # q and v in 2 blocks
    with torch.no_grad():
        want = ref(input_ids=ids).logits
    assert torch.allclose(ours, want, atol=1e-5), (ours - want).abs().max()

    # The routing does something: the mix is neither expert alone.
    if top_k == 1:
        for single in ("one", "two"):
            assert not torch.allclose(ours, logits(engine, single, ids), atol=1e-4)


def test_experts_with_different_ranks_and_targets(tmp_path):
    base, one, tok = tiny_model(tmp_path)  # q, v at rank 4
    mlp = adapter(
        base,
        tmp_path / "mlp",
        seed=2,
        r=8,
        targets=("gate_proj", "up_proj", "down_proj"),
    )
    engine = Engine(str(base), device="cpu", served_name="tiny")
    engine.load_adapter("one", str(one))
    engine.load_adapter("mlp", str(mlp))
    engine.load_mix("mix", ["one", "mlp"], 2, ALL)
    routes = {r.name.split(".", 3)[-1]: r.experts for r in engine.mixes["mix"].routes}
    assert routes["layers.0.self_attn.q_proj"] == ["one"]
    assert routes["layers.1.mlp.down_proj"] == ["mlp"]

    # Each module has a single expert, so the routing weight is 1 and the
    # mix is both adapters applied at once.
    ids = prompt_ids(tok)
    ours = logits(engine, "mix", ids)
    engine.peft.base_model.set_adapter(["one", "mlp"])
    with torch.no_grad():
        both = engine.peft(input_ids=ids).logits
    assert torch.allclose(ours, both, atol=1e-5), (ours - both).abs().max()

    # A mix over block 0 leaves block 1 alone.
    engine.load_mix("early", ["one", "mlp"], 1, ALL, blocks="0")
    assert {r.name.split(".")[4] for r in engine.mixes["early"].routes} == {"0"}
    with pytest.raises(ValueError, match="no expert has a LoRA module"):
        engine.load_mix("none", ["one"], 1, ["gate"])
    with pytest.raises(ValueError, match="top_k"):
        engine.load_mix("bad", ["one", "mlp"], 3, ALL)


def test_mix_endpoints(tmp_path):
    base, one, tok = tiny_model(tmp_path)
    two = adapter(base, tmp_path / "two", seed=1)
    engine = Engine(str(base), device="cpu", served_name="tiny")
    server = serve(engine)
    url = "http://{}:{}".format(*server.server_address[:2])
    try:
        for name, path in (("one", one), ("two", two)):
            post(
                url + "/v1/load_lora_adapter",
                {"lora_name": name, "lora_path": str(path)},
            )
        post(
            url + "/v1/load_mix",
            {"mix_name": "mix", "experts": ["one", "two"], "top_k": 1, "targets": ALL},
        )
        assert [m["id"] for m in get(url + "/v1/models")["data"]] == [
            "tiny",
            "one",
            "two",
            "mix",
        ]
        reply = json.loads(
            post(
                url + "/v1/chat/completions",
                {"model": "mix", "messages": PROMPT, "max_tokens": 6, "temperature": 0},
            )
        )
        assert reply["model"] == "mix" and reply["usage"]["completion_tokens"] > 0

        # An expert a mix uses stays loaded; the mix goes first.
        with pytest.raises(urllib.error.HTTPError) as err:
            post(url + "/v1/unload_lora_adapter", {"lora_name": "one"})
        assert err.value.code == 400 and "expert of mix" in err.value.read().decode()
        with pytest.raises(urllib.error.HTTPError) as err:
            post(
                url + "/v1/load_mix",
                {
                    "mix_name": "x",
                    "experts": ["one"],
                    "top_k": 1,
                    "targets": ALL,
                    "router": "train",
                },
            )
        assert err.value.code == 400
        post(url + "/v1/unload_mix", {"mix_name": "mix"})
        post(url + "/v1/unload_lora_adapter", {"lora_name": "one"})
        assert [m["id"] for m in get(url + "/v1/models")["data"]] == ["tiny", "two"]
    finally:
        server.shutdown()


def test_experts_with_saved_modules_are_not_mixed(tmp_path):
    """Token routing has no sum of whole-module replacements, and dropping
    them would serve an expert without weights it was trained with."""
    base, one, tok = tiny_model(tmp_path)
    head = adapter(base, tmp_path / "head", seed=3, saved=["lm_head"])
    engine = Engine(str(base), device="cpu", served_name="tiny")
    engine.load_adapter("one", str(one))
    engine.load_adapter("head", str(head))
    with pytest.raises(ValueError, match=r"'head' replaces whole modules \(lm_head\)"):
        engine.load_mix("mix", ["one", "head"], 1, ALL)
    assert "mix" not in engine.models()
    # The expert still serves as its own model, saved head included.
    ids = prompt_ids(tok)
    with_head = logits(engine, "head", ids)
    model = peft.PeftModel.from_pretrained(
        transformers.AutoModelForCausalLM.from_pretrained(base), str(head)
    ).eval()
    with torch.no_grad():
        assert torch.allclose(with_head, model(input_ids=ids).logits, atol=1e-5)
