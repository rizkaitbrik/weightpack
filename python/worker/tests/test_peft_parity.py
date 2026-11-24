"""Parity with PEFT: wpk squash gives the weights merge_and_unload gives.

Builds a tiny BF16 Llama with transformers and a LoRA over every linear
module with PEFT, then compares PEFT's merged weights with the image
`wpk build` + `wpk squash` produce, tensor by tensor.

Needs the parity dependency group (torch, transformers, peft) and the wpk
and wpkd binaries: npx nx run worker:parity
"""

import json

import numpy as np
import pytest

from weightpack_worker.safetensors_io import to_float32

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def bf16_bits(t):
    return t.detach().contiguous().view(torch.int16).numpy().view(np.uint16)


@pytest.mark.parametrize("use_rslora", [False, True])
def test_squash_matches_peft_merge(weightpack, tmp_path, use_rslora):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(tmp_path / "base")
    (tmp_path / "base" / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}})
    )

    lora = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=TARGETS,
        lora_dropout=0.0,
        use_rslora=use_rslora,
        init_lora_weights=False,  # random A and B, so every delta is non-zero
    )
    peft_model = peft.get_peft_model(model, lora)
    peft_model.save_pretrained(tmp_path / "adapter")
    want = peft_model.merge_and_unload().state_dict()

    (tmp_path / "Weightfile").write_text("FROM ./base\nLAYER lora ./adapter\n")
    weightpack.wpk("build", "-t", "tiny", str(tmp_path))
    weightpack.wpk("squash", "-t", "tiny-dense", "tiny")
    got = weightpack.tensors("tiny-dense")
    base = weightpack.tensors("tiny")

    assert set(got) == set(want)
    adapted, identical, total = 0, 0, 0
    for name, tensor in want.items():
        dtype, raw, _ = got[name]
        assert dtype == "BF16", name
        peft_bits = bf16_bits(tensor)
        if not any(name.endswith(f"{t}.weight") for t in TARGETS):
            assert np.array_equal(raw, peft_bits), name  # untouched: exact
            continue
        adapted += 1
        assert not np.array_equal(raw, base[name][1]), f"{name} was not merged"
        ours, theirs = to_float32("BF16", raw), to_float32("BF16", peft_bits)
        # At most one BF16 step apart, from float32 summation order alone.
        np.testing.assert_allclose(ours, theirs, rtol=2**-7, atol=1e-12, err_msg=name)
        identical += int(np.sum(raw == peft_bits))
        total += raw.size
    assert adapted == 2 * len(TARGETS)
    share = identical / total
    print(f"rslora={use_rslora}: {share:.4%} of {total} merged weights match PEFT")
    assert share > 0.99, f"only {share:.4%} of merged weights match PEFT bit for bit"
