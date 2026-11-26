"""Token routing as the plugin runs it inside vLLM's LoRA path, checked
against the NumPy reference with a stand-in for vLLM's punica wrapper."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from weightpack_vllm import reference  # noqa: E402
from weightpack_vllm.routing import (  # noqa: E402
    Routing,
    gated_lora_linear,
    in_domain,
    load_mix,
    stack_experts,
)

IN, MAX_RANK, SLOTS = 16, 32, 3


class FakePunica:
    """vLLM's add_shrink/add_expand semantics, by token: each token uses the
    adapter in its slot, none for -1."""

    def __init__(self, token_slots):
        self._token_lora_indices = torch.tensor(token_slots)

    def add_shrink(self, y, x, lora_a_stacked, scale, **_):
        for s, a in enumerate(lora_a_stacked):
            for t, slot in enumerate(self._token_lora_indices.tolist()):
                if slot >= 0:
                    y[s, t] = (x[t].float() @ a[slot, 0].float().T) * scale

    def add_expand(self, y, x, lora_b_stacked, output_slices, add_inputs=True, **_):
        offset = 0
        for s, (b, size) in enumerate(zip(lora_b_stacked, output_slices, strict=True)):
            for t, slot in enumerate(self._token_lora_indices.tolist()):
                if slot >= 0:
                    y[t, offset : offset + size] += (x[s, t] @ b[slot, 0].float().T).to(
                        y.dtype
                    )
            offset += size


def expert(rng, rank, out, scaling):
    return reference.LoRAExpert(
        a=rng.standard_normal((rank, IN)).astype(np.float32),
        b=rng.standard_normal((out, rank)).astype(np.float32),
        scaling=scaling,
    )


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


@pytest.mark.parametrize("router", ["arrow", "trained"])
@pytest.mark.parametrize("top_k", [1, 2])
def test_gated_lora_matches_the_reference(top_k, router):
    rng = np.random.default_rng(0)
    outs = (12, 8)  # two slices, as q and k of a packed qkv_proj
    # Three experts of different ranks: e0 touches only slice 0, e2 only
    # slice 1, e1 both.
    ranks = [4, 2, 6]
    scalings = [2.0, 0.5, 1.0]
    touches = [{0}, {0, 1}, {1}]
    experts = [
        {
            f"layers.0.self_attn.{'qk'[s]}_proj": expert(
                rng, ranks[e], outs[s], scalings[e]
            )
            for s in touches[e]
        }
        for e in range(3)
    ]
    as_tuples = [
        {n: (t(x.a), t(x.b), x.scaling) for n, x in e.items()} for e in experts
    ]
    stacked = stack_experts(as_tuples, ranks, lambda n: True, torch.float32)
    assert sorted(stacked) == ["layers.0.self_attn.k_proj", "layers.0.self_attn.q_proj"]

    # Slot 1 holds the mix, slot 0 a plain adapter of rank 3.
    a_stacked = tuple(torch.zeros(SLOTS, 1, MAX_RANK, IN) for _ in outs)
    b_stacked = tuple(torch.zeros(SLOTS, 1, o, MAX_RANK) for o in outs)
    plain = [expert(rng, 3, o, 1.0) for o in outs]
    for s, name in enumerate(
        ["layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj"]
    ):
        a, b = stacked[name]
        a_stacked[s][1, 0, : a.shape[0]] = a
        b_stacked[s][1, 0, :, : b.shape[1]] = b
        a_stacked[s][0, 0, :3] = t(plain[s].a)
        b_stacked[s][0, 0, :, :3] = t(plain[s].b)

    # A trained router's rows per slice, one per expert of the mix; the
    # rows of experts that do not touch a slice are never read.
    rows = None
    if router == "trained":
        rows = [
            (
                t(rng.standard_normal((3, IN)).astype(np.float32)),
                t(rng.standard_normal(3).astype(np.float32)),
            )
            for _ in outs
        ]
    routing = Routing(len(outs), SLOTS, MAX_RANK, IN, 4, "cpu")
    routing.set_slot(
        1,
        ranks,
        top_k,
        [a[1, 0] for a in a_stacked],
        [b[1, 0] for b in b_stacked],
        rows,
    )
    assert routing.touch[0, 1].tolist() == [True, True, False, False]
    assert routing.touch[1, 1].tolist() == [False, True, True, False]

    slots = [1, 1, 0, -1, 1, 0, 1, 1]
    x = rng.standard_normal((len(slots), IN)).astype(np.float32)
    y = torch.zeros(len(slots), sum(outs))
    gated_lora_linear(
        FakePunica(slots), routing, y, t(x), a_stacked, b_stacked, 1.0, outs
    )

    # The reference: per slice, Arrow over the experts that touch it.
    want = np.zeros((len(slots), sum(outs)), dtype=np.float64)
    offset = 0
    for s, out in enumerate(outs):
        name = f"layers.0.self_attn.{'qk'[s]}_proj"
        touching = [i for i, e in enumerate(experts) if name in e]
        here = [experts[i][name] for i in touching]
        if rows is None:
            protos = np.stack([reference.arrow_prototype(e) for e in here])
            gates = reference.arrow_gates(x, protos, top_k)
        else:
            w, b = (r.numpy()[touching] for r in rows[s])
            gates = reference.router_gates(x, w, b, top_k)
        for row, slot in enumerate(slots):
            if slot == 1:
                want[row, offset : offset + out] = sum(
                    gates[row, i] * e.apply(x[row : row + 1])[0]
                    for i, e in enumerate(here)
                )
            elif slot == 0:
                want[row, offset : offset + out] = plain[s].apply(x[row : row + 1])[0]
        offset += out
    np.testing.assert_allclose(y.numpy(), want, rtol=1e-4, atol=1e-4)

    # Routing changes the result: the mix is not the plain sum of its experts.
    flat = torch.zeros_like(y)
    routing.clear(1)
    gated_lora_linear(
        FakePunica(slots), routing, flat, t(x), a_stacked, b_stacked, 1.0, outs
    )
    assert not torch.allclose(flat[0], y[0], atol=1e-3)


def test_gates_leave_other_tokens_alone():
    routing = Routing(1, SLOTS, 8, IN, 4, "cpu")
    x = torch.randn(5, IN)
    g = routing.gates(x, torch.tensor([0, 1, -1, 2, 0]))
    assert torch.equal(g, torch.ones(1, 5, 8))


def test_stack_experts_keeps_the_domain_and_folds_scaling():
    a, b = torch.ones(2, IN), torch.ones(4, 2)
    experts = [
        {
            "model.layers.0.mlp.up_proj": (a, b, 3.0),
            "model.layers.5.mlp.up_proj": (a, b, 3.0),
        },
        {"model.layers.0.self_attn.q_proj": (a, b, 1.0)},
    ]

    def keep(name):
        return in_domain(name, ["up"], "0-3")

    stacked = stack_experts(experts, [2, 2], keep, torch.float32)
    assert list(stacked) == ["model.layers.0.mlp.up_proj"]
    sa, sb = stacked["model.layers.0.mlp.up_proj"]
    assert sa.shape == (4, IN) and sb.shape == (4, 4)
    assert torch.equal(sb[:, :2], 3.0 * b) and torch.equal(sb[:, 2:], torch.zeros(4, 2))
    with pytest.raises(ValueError, match="rank 3"):
        stack_experts(experts, [3, 2], lambda n: True, torch.float32)


def test_load_mix(tmp_path):
    assert load_mix(str(tmp_path)) is None
    mix = {
        "version": 1,
        "router": "arrow",
        "top_k": 1,
        "targets": ["q"],
        "experts": [{"id": "a", "path": "/x"}],
    }
    (tmp_path / "weightpack_mix.json").write_text(json.dumps(mix))
    assert load_mix(str(tmp_path))["top_k"] == 1
    trained = {**mix, "router": "trained", "router_file": "/r.safetensors"}
    (tmp_path / "weightpack_mix.json").write_text(json.dumps(trained))
    assert load_mix(str(tmp_path))["router_file"] == "/r.safetensors"
    for bad in ({"version": 2}, {"top_k": 2}, {"router": "trained"}, {"router": "x"}):
        (tmp_path / "weightpack_mix.json").write_text(json.dumps({**mix, **bad}))
        with pytest.raises(ValueError):
            load_mix(str(tmp_path))
    assert in_domain("model.layers.3.mlp.down_proj", ["down"], "0-1,3")
    assert not in_domain("model.layers.3.mlp.down_proj", ["up"], "")
    assert not in_domain("lm_head", ["lm_head"], "0-31")


@pytest.mark.parametrize("n", [2, 3, 8])
def test_ties_pick_exactly_top_k(n):
    # Experts with the same A and differently scaled B have the same
    # prototype, so they tie for every token; top_k of them are picked, the
    # first ones listed, as weightpack-engine does.
    rng = np.random.default_rng(0)
    a = t(rng.standard_normal((1, IN)).astype(np.float32))
    b = t(rng.standard_normal((4, 1)).astype(np.float32))
    a_slot = torch.cat([a] * n)
    b_slot = torch.cat([b * (i + 1) for i in range(n)], dim=1)
    routing = Routing(1, SLOTS, n, IN, n, "cpu")
    x = torch.randn(5, IN)
    for top_k in (1, 2):
        routing.set_slot(0, [1] * n, top_k, [a_slot], [b_slot])
        g = routing.gates(x, torch.zeros(5, dtype=torch.long))[0]
        want = torch.zeros(5, n)
        want[:, :top_k] = 1 / top_k
        torch.testing.assert_close(g, want)
        np.testing.assert_allclose(
            reference.arrow_gates(x.numpy(), np.ones((n, IN)), top_k), want.numpy()
        )
