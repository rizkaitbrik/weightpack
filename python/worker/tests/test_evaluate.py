"""Fidelity evals on a tiny model: suites, scores and the SELF suite.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from weightpack_worker import evaluate  # noqa: E402

from .test_engine import tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """The tiny model, a copy with noise on every matrix, and another model
    of the same shape, initialised from another seed."""
    d = tmp_path_factory.mktemp("eval")
    base, _, tok = tiny_model(d)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.add_(torch.randn(p.shape, generator=gen) * p.std())
    noisy = d / "noisy"
    model.save_pretrained(noisy)
    tok.save_pretrained(noisy)
    torch.manual_seed(1)
    other = d / "other"
    transformers.LlamaForCausalLM(model.config).save_pretrained(other)
    tok.save_pretrained(other)
    return str(base), str(noisy), str(other)


def test_load_suite(tmp_path):
    data = tmp_path / "math.jsonl"
    data.write_text('{"text": "t1 t2 t3"}\n\n{"prompt": "t1", "completion": "t2"}\n')
    s = evaluate.load_suite(str(data))
    assert s.name == "math" and len(s.examples) == 2
    (tmp_path / "evals").mkdir()
    spec = tmp_path / "evals" / "suite.yaml"
    spec.write_text("name: finance-v2\ndata: ../math.jsonl\n")
    s = evaluate.load_suite(str(spec))
    assert s.name == "finance-v2" and len(s.examples) == 2
    spec.write_text("name: x\n")
    with pytest.raises(ValueError, match="data:"):
        evaluate.load_suite(str(spec))
    data.write_text('{"text": "t1"}\nnot json\n')
    with pytest.raises(ValueError, match=r"math\.jsonl:2"):
        evaluate.load_suite(str(data))


def test_scores_on_a_suite(models, tmp_path):
    base, noisy, _ = models
    suite = tmp_path / "s.jsonl"
    suite.write_text(
        "".join(
            json.dumps({"text": " ".join(f"t{(s + i) % 60}" for i in range(12))}) + "\n"
            for s in range(0, 60, 3)
        )
    )
    res = evaluate.evaluate([base, base, noisy], str(suite), device="cpu")
    assert res.suite == "s" and res.examples == 20 and res.tokens == 20 * 11
    a, b, c = res.scores
    assert a == b  # the same model scores the same
    assert 0 <= a.accuracy <= 100 and a.loss > 0
    assert c.loss != a.loss


def test_self_suite(models):
    base, _, other = models
    res = evaluate.evaluate([base, other], device="cpu")
    assert res.suite == evaluate.SELF
    assert res.examples == len(evaluate.SELF_PROMPTS) and res.tokens > 0
    source, other = res.scores
    # The source agrees with its own greedy replies; another model does not.
    assert source.accuracy > 95
    assert other.accuracy < source.accuracy - 5
