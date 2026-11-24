"""End to end: wpk import stores a full fine-tune losslessly.

Through the real wpk and wpkd, with this worker: the fine-tune is stored as
compressed XOR deltas, and squash gives every tensor back bit for bit.
"""

import json

import numpy as np

from weightpack_worker.safetensors_io import from_float32, to_float32

from .test_squash import write_many

HIDDEN, INTER, VOCAB, BLOCKS = 256, 512, 64, 2
MATRICES = {
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.v_proj": (HIDDEN, HIDDEN),
    "mlp.up_proj": (INTER, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTER),
}


def write_model(d, tensors):
    d.mkdir()
    write_many(
        d / "model.safetensors",
        {n: ("BF16", from_float32("BF16", w)) for n, w in tensors.items()},
    )
    (d / "config.json").write_text(
        json.dumps(
            {"model_type": "llama", "num_hidden_layers": BLOCKS, "hidden_size": HIDDEN}
        )
    )
    (d / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')


def base_and_finetune(tmp_path, rng):
    base = {
        "model.embed_tokens.weight": rng.standard_normal((VOCAB, HIDDEN)).astype(
            np.float32
        ),
        "model.norm.weight": np.ones(HIDDEN, np.float32),
    }
    for b in range(BLOCKS):
        base[f"model.layers.{b}.input_layernorm.weight"] = np.ones(HIDDEN, np.float32)
        for m, shape in MATRICES.items():
            base[f"model.layers.{b}.{m}.weight"] = (
                rng.standard_normal(shape).astype(np.float32) * 0.02
            )
    # Round to BF16 first, so the fine-tune starts from the stored base.
    base = {n: to_float32("BF16", from_float32("BF16", w)) for n, w in base.items()}
    # A full fine-tune nudges every matrix by about 1%; norms and the
    # embedding stay as they were.
    tuned = {
        n: (
            w * (1 + 0.01 * rng.standard_normal(w.shape).astype(np.float32))
            if w.ndim == 2 and "embed" not in n
            else w
        )
        for n, w in base.items()
    }
    write_model(tmp_path / "base", base)
    write_model(tmp_path / "finetune", tuned)
    return base, tuned


def test_import_is_bit_exact(weightpack, tmp_path):
    rng = np.random.default_rng(3)
    base, tuned = base_and_finetune(tmp_path, rng)
    out = weightpack.wpk(
        "import",
        "--parent",
        str(tmp_path / "base"),
        "--id",
        "finance",
        "-t",
        "finance",
        str(tmp_path / "finetune"),
    )
    assert "Imported" in out and "Tagged finance:latest" in out

    detail = weightpack.inspect("finance")
    layer = next(lr for lr in detail["config"]["layers"] if lr["id"] == "finance")
    assert layer["kind"] == "delta.xor" and "fidelity" not in layer
    blobs = [
        d
        for d in detail["manifest"]["layers"]
        if d["annotations"].get("wpk.layer") == "finance"
    ]
    changed = {d["annotations"]["wpk.tensor"] for d in blobs}
    assert changed == {n for n, w in base.items() if w.ndim == 2 and "embed" not in n}
    raw = sum(base[n].size * 2 for n in changed)
    stored = sum(d["size"] for d in blobs)
    print(f"XOR deltas: {raw} raw bytes stored in {stored} ({stored / raw:.1%})")
    assert stored < 0.8 * raw

    weightpack.wpk("squash", "-t", "finance-dense", "finance")
    got = weightpack.tensors("finance-dense")
    assert set(got) == set(tuned)
    for name, w in tuned.items():
        assert got[name][1].tobytes() == from_float32("BF16", w).tobytes(), name

    # An XOR layer has no PEFT form: export says to squash first.
    r = weightpack.run("export", "-o", str(tmp_path / "x"), "finance")
    assert r.returncode == 1 and "squash the image" in r.stderr


def test_lora_over_an_imported_finetune(weightpack, tmp_path):
    rng = np.random.default_rng(4)
    base, tuned = base_and_finetune(tmp_path, rng)
    weightpack.wpk(
        "import",
        "--parent",
        str(tmp_path / "base"),
        "-t",
        "tuned",
        str(tmp_path / "finetune"),
    )

    rank, alpha = 4, 8
    module = "model.layers.1.self_attn.q_proj"
    a = rng.standard_normal((rank, HIDDEN)).astype(np.float32) * 0.1
    b = rng.standard_normal((HIDDEN, rank)).astype(np.float32) * 0.1
    (tmp_path / "adapter").mkdir()
    write_many(
        tmp_path / "adapter" / "adapter_model.safetensors",
        {
            f"base_model.model.{module}.lora_A.weight": ("F32", a),
            f"base_model.model.{module}.lora_B.weight": ("F32", b),
        },
    )
    (tmp_path / "adapter" / "adapter_config.json").write_text(
        json.dumps(
            {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "bias": "none"}
        )
    )
    (tmp_path / "Weightfile").write_text("FROM tuned\nLAYER style ./adapter\n")
    weightpack.wpk("build", "-t", "styled", str(tmp_path))
    weightpack.wpk("squash", "-t", "styled-dense", "styled")

    got = weightpack.tensors("styled-dense")
    name = module + ".weight"
    stored = to_float32("BF16", from_float32("BF16", tuned[name]))  # as imported
    want = from_float32("BF16", stored + np.float32(alpha / rank) * (b @ a))
    assert got[name][1].tobytes() == want.tobytes()
    other = "model.layers.0.mlp.up_proj.weight"
    assert got[other][1].tobytes() == from_float32("BF16", tuned[other]).tobytes()
