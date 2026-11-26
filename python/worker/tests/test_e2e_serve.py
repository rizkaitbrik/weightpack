"""End to end: wpk run through wpkd's gateway, with the real reference
engine, gives what transformers and PEFT generate directly.

Needs the parity dependency group and the wpk/wpkd binaries:
npx nx run worker:parity
"""

import json
import re
import urllib.error
import urllib.request

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from .test_engine import PROMPT, direct, tiny_model  # noqa: E402


def chat(endpoint, model, **extra):
    body = {
        "model": model,
        "messages": PROMPT,
        "max_tokens": 8,
        "temperature": 0,
        **extra,
    }
    req = urllib.request.Request(
        endpoint + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_run_serves_images_on_one_engine(weightpack, tmp_path):
    base, adapter, tok = tiny_model(tmp_path)
    # A second adapter, for a mix.
    torch.manual_seed(1)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    lora = peft.LoraConfig(
        r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )
    peft.get_peft_model(model, lora).save_pretrained(tmp_path / "adapter2")

    (tmp_path / "Weightfile").write_text("FROM ./base\nLAYER styled ./adapter\n")
    weightpack.wpk("build", "-t", "styled", str(tmp_path))
    (tmp_path / "Weightfile").write_text("FROM ./base\n")
    weightpack.wpk("build", "-t", "plain", str(tmp_path))
    (tmp_path / "Weightfile").write_text(
        "FROM ./base\nMIX domain targets=attn top_k=1\n"
        "  EXPERT one ./adapter\n  EXPERT two ./adapter2\n"
    )
    weightpack.wpk("build", "-t", "domains", str(tmp_path))

    out = weightpack.wpk("run", "--device", "cpu", "styled")
    assert "Running styled on the reference engine (cpu)" in out, out
    endpoint = re.search(r"endpoint\s+(\S+)", out).group(1)
    weightpack.wpk("run", "--device", "cpu", "plain")
    out = weightpack.wpk("run", "--device", "cpu", "domains")
    assert "domains/one" in out and "domains/two" in out

    ps = weightpack.wpk("ps")
    rows = [line.split() for line in ps.splitlines()[1:]]
    assert sorted(r[0] for r in rows) == ["domains", "plain", "styled"], ps
    assert len({r[4] for r in rows}) == 1, "the images should share one engine"

    styled = chat(endpoint, "styled")
    assert styled["model"] == "styled"
    want_styled = direct(base, adapter, tok)
    assert styled["choices"][0]["message"]["content"] == want_styled
    assert chat(endpoint, "plain")["choices"][0]["message"]["content"] == direct(
        base, None, tok
    )
    # The mix's first expert is the same adapter as styled's layer.
    assert (
        chat(endpoint, "domains/one")["choices"][0]["message"]["content"] == want_styled
    )
    assert chat(endpoint, "domains/two")["choices"][0]["message"]["content"] == direct(
        base, tmp_path / "adapter2", tok
    )

    with urllib.request.urlopen(endpoint + "/models") as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    assert ids == ["domains/one", "domains/two", "plain", "styled"]
    with pytest.raises(urllib.error.HTTPError) as err:
        chat(endpoint, "nope")
    assert err.value.code == 404

    assert "listening on" in weightpack.wpk("logs", "styled")

    weightpack.wpk("stop", "styled", "plain", "domains")
    assert weightpack.wpk("ps").strip().splitlines()[1:] == []
    with urllib.request.urlopen(endpoint + "/models") as r:
        assert json.loads(r.read())["data"] == []
    out = weightpack.wpk("prune")
    assert "of serving files no container uses" in out, out


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Silicon")
def test_run_on_metal(weightpack, tmp_path):
    base, adapter, tok = tiny_model(tmp_path)
    (tmp_path / "Weightfile").write_text("FROM ./base\nLAYER styled ./adapter\n")
    weightpack.wpk("build", "-t", "styled", str(tmp_path))
    out = weightpack.wpk("run", "--device", "mps", "styled")
    assert "Running styled on the reference engine (mps)" in out, out
    endpoint = re.search(r"endpoint\s+(\S+)", out).group(1)
    got = chat(endpoint, "styled")["choices"][0]["message"]["content"]
    # Metal kernels round differently from the CPU's, so a tiny random
    # model may pick other tokens; it must still generate.
    assert got
    assert "(device mps)" in weightpack.wpk("logs", "styled")
    weightpack.wpk("stop", "styled")
