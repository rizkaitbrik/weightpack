"""End to end: wpk build and wpk squash through wpkd and this worker."""

import json

import numpy as np

from weightpack_worker.safetensors_io import from_float32, to_float32

from .test_squash import write_many

HIDDEN, INTER, VOCAB, BLOCKS = 256, 512, 64, 2
SHAPES = {  # out, in
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.v_proj": (HIDDEN, HIDDEN),
    "mlp.up_proj": (INTER, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTER),
}


def write_project(d, rng):
    """A BF16 Llama-shaped base and an F32 PEFT LoRA on q and v."""
    base = {}
    for name, shape in [
        ("model.embed_tokens.weight", (VOCAB, HIDDEN)),
        ("lm_head.weight", (VOCAB, HIDDEN)),
        ("model.norm.weight", (HIDDEN,)),
    ]:
        base[name] = rng.standard_normal(shape).astype(np.float32)
    for b in range(BLOCKS):
        base[f"model.layers.{b}.input_layernorm.weight"] = np.ones(HIDDEN, np.float32)
        for module, shape in SHAPES.items():
            base[f"model.layers.{b}.{module}.weight"] = (
                rng.standard_normal(shape).astype(np.float32) * 0.02
            )
    (d / "base").mkdir()
    write_many(
        d / "base" / "model.safetensors",
        {n: ("BF16", from_float32("BF16", w)) for n, w in base.items()},
    )
    (d / "base" / "config.json").write_text(
        json.dumps(
            {"model_type": "llama", "num_hidden_layers": BLOCKS, "hidden_size": HIDDEN}
        )
    )
    (d / "base" / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')

    rank, alpha = 4, 8
    lora = {}
    for b in range(BLOCKS):
        for module in ("self_attn.q_proj", "self_attn.v_proj"):
            out_f, in_f = SHAPES[module]
            p = f"base_model.model.model.layers.{b}.{module}"
            lora[p + ".lora_A.weight"] = (
                rng.standard_normal((rank, in_f)).astype(np.float32) * 0.1
            )
            lora[p + ".lora_B.weight"] = (
                rng.standard_normal((out_f, rank)).astype(np.float32) * 0.1
            )
    (d / "adapter").mkdir()
    write_many(
        d / "adapter" / "adapter_model.safetensors",
        {n: ("F32", a) for n, a in lora.items()},
    )
    (d / "adapter" / "adapter_config.json").write_text(
        json.dumps(
            {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "bias": "none"}
        )
    )
    (d / "Weightfile").write_text("FROM ./base\nLAYER french ./adapter\n")
    return base, lora, alpha / rank


def test_squash_end_to_end(weightpack, tmp_path):
    rng = np.random.default_rng(42)
    base, lora, scaling = write_project(tmp_path, rng)

    weightpack.wpk("build", "-t", "french", str(tmp_path))
    out = weightpack.wpk("squash", "-t", "french-dense", "french")
    assert "Squashed into sha256:" in out and "Tagged french-dense:latest" in out

    detail = weightpack.inspect("french-dense")
    assert [layer["kind"] for layer in detail["config"]["layers"]] == ["base"]
    assert detail["manifest"]["annotations"]["wpk.squashed"] == "french"

    src = weightpack.tensors("french")
    got = weightpack.tensors("french-dense")
    assert set(got) == set(src) == set(base)
    for name, (dtype, raw, digest) in got.items():
        assert dtype == "BF16"
        module = name.removesuffix(".weight")
        a = lora.get(f"base_model.model.{module}.lora_A.weight")
        if a is None:
            # Untouched: the very same blob and bytes as the source image.
            assert digest == src[name][2], name
            assert np.array_equal(raw, src[name][1]), name
            continue
        b = lora[f"base_model.model.{module}.lora_B.weight"]
        w = to_float32("BF16", src[name][1]).astype(np.float64)
        want = to_float32(
            "BF16",
            from_float32(
                "BF16", (w + scaling * (b.astype(np.float64) @ a)).astype(np.float32)
            ),
        )
        have = to_float32("BF16", raw)
        np.testing.assert_allclose(have, want, rtol=2**-7, atol=0, err_msg=name)
        assert np.mean(have == want) > 0.95, name
        assert digest != src[name][2], name
