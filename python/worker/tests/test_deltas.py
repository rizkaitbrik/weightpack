"""Full fine-tunes served as bit1 deltas over the reference engine's base.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from weightpack_worker import deltas, lossy  # noqa: E402
from weightpack_worker.engine import Engine, serve  # noqa: E402
from weightpack_worker.safetensors_io import (  # noqa: E402
    index_tensors,
    load_raw,
    write_tensors,
)

from .test_calibrate import _store_head_too, finetune  # noqa: E402
from .test_engine import PROMPT, direct, get, post  # noqa: E402


def delta_dir(tmp_path, base, ft, name="delta"):
    """The bit1 layer wpkd would export for ft over base, and the model its
    decoding gives (as wpk squash would)."""
    out, recon = tmp_path / name, tmp_path / (name + "-dense")
    out.mkdir()
    recon.mkdir()
    res = lossy.encode_lossy(
        lossy.BIT1,
        [str(f) for f in sorted(base.glob("*.safetensors"))],
        [],
        [str(f) for f in sorted(ft.glob("*.safetensors"))],
        out,
        reconstruct_dir=recon,
    )
    meta = {
        "version": 1,
        "codec": "bit1",
        "layer": name,
        "files": [f for _, f in res.tensors],
    }
    (out / deltas.DELTA_FILE).write_text(json.dumps(meta))
    return out, recon


def another_finetune(tmp_path, base, seed, name):
    """base nudged with another seed's noise, under tmp_path/name."""
    model = transformers.AutoModelForCausalLM.from_pretrained(base, dtype="auto")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(1 + 0.05 * torch.randn(p.shape, generator=gen).to(p.dtype))
            else:
                p.add_(0.01 * torch.randn(p.shape, generator=gen).to(p.dtype))
    out = tmp_path / name
    model.save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(base).save_pretrained(out)
    if (base / "model.safetensors").is_file() and "lm_head.weight" in index_tensors(
        [str(base / "model.safetensors")]
    ):
        _store_head_too(out / "model.safetensors")
    return out


def weights(model_dir, dtype):
    """A model's tensors as transformers loads them, by name."""
    m = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    out = dict(m.named_parameters(remove_duplicate=False))
    out.update(m.named_buffers(remove_duplicate=False))
    return {n: t.detach().clone() for n, t in out.items()}


def assert_weights(engine, want):
    for name, t in engine.switcher.tensors.items():
        assert torch.equal(t.detach().cpu(), want[name]), name


@pytest.mark.parametrize(
    "variant,dtype",
    [("untied", "F32"), ("tied", "F32"), ("tied-both", "F32"), ("untied", "BF16")],
)
def test_switching_gives_the_decoded_model(tmp_path, variant, dtype):
    """A fine-tune's requests run on exactly the model decoding its bit1
    layer gives, and the base's on the base, bit for bit: tied matrices
    included, whether the checkpoint stores the head under its own name too
    or not."""
    base, ft, _, _ = finetune(tmp_path, variant, dtype)
    path, dense = delta_dir(tmp_path, base, ft)
    engine = Engine(str(base), device="cpu")
    torch_dtype = engine.model.dtype
    want_base, want_ft = weights(base, torch_dtype), weights(dense, torch_dtype)
    assert any(not torch.equal(want_base[n], want_ft[n]) for n in want_base)

    engine.load_adapter("ft", str(path))
    assert engine.models() == ["base", "ft"]
    assert_weights(engine, want_base)  # loading switches nothing
    engine._model_for("ft")
    assert_weights(engine, want_ft)
    engine._model_for("base")
    assert_weights(engine, want_base)

    tok = transformers.AutoTokenizer.from_pretrained(base)
    ids = engine.encode_chat(PROMPT)
    text, _, _ = engine.generate("ft", ids, {"max_tokens": 8, "temperature": 0})
    assert text == direct(dense, None, tok)
    assert_weights(engine, want_ft)  # the fine-tune stays until another needs it

    engine.unload_adapter("ft")
    assert engine.models() == ["base"]
    assert_weights(engine, want_base)


def test_fine_tunes_adapters_and_the_base_share_one_model(tmp_path):
    """Two fine-tunes, a LoRA adapter and the base, in any order: each
    request gets its own weights, and the engine holds one copy of them."""
    base, ft, _, _ = finetune(tmp_path)
    ft2 = another_finetune(tmp_path, base, 1, "ft2")
    path1, dense1 = delta_dir(tmp_path, base, ft, "one")
    path2, dense2 = delta_dir(tmp_path, base, ft2, "two")
    tok = transformers.AutoTokenizer.from_pretrained(base)
    engine = Engine(str(base), device="cpu")
    engine.load_adapter("one", str(path1))
    engine.load_adapter("lora", str(tmp_path / "adapter"))
    engine.load_adapter("two", str(path2))
    params = sum(p.numel() for p in engine.model.parameters())

    want = {
        "base": direct(base, None, tok),
        "one": direct(dense1, None, tok),
        "two": direct(dense2, None, tok),
        "lora": direct(base, tmp_path / "adapter", tok),
    }
    assert len(set(want.values())) == 4
    ids = engine.encode_chat(PROMPT)
    for name in ["one", "two", "lora", "one", "base", "two", "two", "lora"]:
        text, _, _ = engine.generate(name, ids, {"max_tokens": 8, "temperature": 0})
        assert text == want[name], name
    assert sum(p.numel() for p in engine.model.parameters()) == params

    # Unloading the active fine-tune puts the base back.
    engine.unload_adapter("two")
    assert_weights(engine, weights(base, engine.model.dtype))


def test_served_over_http(tmp_path):
    base, ft, _, _ = finetune(tmp_path)
    path, dense = delta_dir(tmp_path, base, ft)
    tok = transformers.AutoTokenizer.from_pretrained(base)
    server = serve(Engine(str(base), device="cpu", served_name="base-x"))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        post(url + "/v1/load_lora_adapter", {"lora_name": "ft", "lora_path": str(path)})
        models = get(url + "/v1/models")["data"]
        assert [(m["id"], m["parent"]) for m in models] == [
            ("base-x", None),
            ("ft", "base-x"),
        ]
        body = {"model": "ft", "messages": PROMPT, "max_tokens": 8, "temperature": 0}
        reply = json.loads(post(url + "/v1/chat/completions", body))
        assert reply["choices"][0]["message"]["content"] == direct(dense, None, tok)
        post(url + "/v1/unload_lora_adapter", {"lora_name": "ft"})
        assert [m["id"] for m in get(url + "/v1/models")["data"]] == ["base-x"]
    finally:
        server.shutdown()


def test_refuses_what_it_cannot_serve(tmp_path):
    base, ft, _, _ = finetune(tmp_path)
    path, _ = delta_dir(tmp_path, base, ft)
    engine = Engine(str(base), device="cpu")
    meta = json.loads((path / deltas.DELTA_FILE).read_text())

    def variant(name, **changes):
        d = tmp_path / name
        shutil.copytree(path, d)
        (d / deltas.DELTA_FILE).write_text(json.dumps({**meta, **changes}))
        return d

    with pytest.raises(ValueError, match="version 2"):
        engine.load_adapter("v2", str(variant("v2", version=2)))
    with pytest.raises(ValueError, match="sparse delta.*wpk squash"):
        engine.load_adapter("sparse", str(variant("sparse", codec="sparse")))

    # A tensor the base model does not have.
    d = variant("stranger")
    write_tensors(
        d / "x.safetensors",
        [
            ("model.extra.weight.signs", "U8", np.zeros(1, np.uint8)),
            ("model.extra.weight.scale", "F32", np.ones(1, np.float32)),
        ],
    )
    (d / deltas.DELTA_FILE).write_text(
        json.dumps({**meta, "files": [*meta["files"], "x.safetensors"]})
    )
    with pytest.raises(ValueError, match="model.extra.weight.*no tensor"):
        engine.load_adapter("stranger", str(d))

    # Signs that do not cover the tensor.
    d = variant("short")
    first = d / meta["files"][0]
    name, entries = lossy.read_delta(str(first))
    if lossy.SIGNS in entries:
        write_tensors(
            first,
            [
                (name + ".signs", "U8", np.asarray(entries[lossy.SIGNS][1])[:-1]),
                (name + ".scale", "F32", np.asarray(entries[lossy.SCALE][1])),
            ],
        )
        with pytest.raises(ValueError, match="sign bytes"):
            engine.load_adapter("short", str(d))
    assert engine.models() == ["base"]


def test_tied_names_must_agree(tmp_path):
    """A head stored under its own name and tied to the embeddings is one
    tensor: a delta that changes the two names differently is refused."""
    base, ft, _, _ = finetune(tmp_path, "tied-both")
    path, _ = delta_dir(tmp_path, base, ft)
    meta = json.loads((path / deltas.DELTA_FILE).read_text())
    for rel in meta["files"]:
        name, entries = lossy.read_delta(str(path / rel))
        if name == "lm_head.weight":
            signs = np.asarray(entries[lossy.SIGNS][1]) ^ 0xFF
            scale = np.asarray(entries[lossy.SCALE][1])
            write_tensors(
                path / rel,
                [(name + ".signs", "U8", signs), (name + ".scale", "F32", scale)],
            )
    engine = Engine(str(base), device="cpu")
    with pytest.raises(ValueError, match="one tensor in the model"):
        engine.load_adapter("ft", str(path))


def test_scale_raw(tmp_path):
    """The scale a switch adds is the stored float32, not a rounded copy."""
    base, ft, _, _ = finetune(tmp_path)
    path, _ = delta_dir(tmp_path, base, ft)
    engine = Engine(str(base), device="cpu")
    delta = engine.switcher.load("ft", str(path))
    for entry in delta.entries.values():
        if entry.signs:
            stored = load_raw(entry.file, _info(entry.file, entry.name + ".scale"))
            assert entry.scale == float(np.float32(stored[0]))


def _info(file, key):
    from weightpack_worker.safetensors_io import read_header

    return read_header(file)[key]


def test_a_fine_tune_that_changed_nothing(tmp_path):
    """A bit1 layer with no deltas, as an import of an unchanged checkpoint
    gives: the engine serves it as the base."""
    base, _, _, _ = finetune(tmp_path)
    path = tmp_path / "same"
    path.mkdir()
    meta = {"version": 1, "codec": "bit1", "layer": "same", "files": []}
    (path / deltas.DELTA_FILE).write_text(json.dumps(meta))
    engine = Engine(str(base), device="cpu")
    engine.load_adapter("same", str(path))
    ids = engine.encode_chat(PROMPT)
    params = {"max_tokens": 8, "temperature": 0}
    assert engine.generate("same", ids, params) == engine.generate("base", ids, params)
    assert_weights(engine, weights(base, engine.model.dtype))
