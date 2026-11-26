"""bit1 scale calibration on a tiny model.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from weightpack_worker import calibrate, evaluate, lossy  # noqa: E402
from weightpack_worker.delta import Parent, fine_tune  # noqa: E402
from weightpack_worker.safetensors_io import (  # noqa: E402
    index_tensors,
    load_raw,
    read_header,
    write_tensors,
)

from .test_engine import tiny_model  # noqa: E402


def _store_head_too(path):
    """Rewrites a tied checkpoint to store its output head under its own
    name as well, as some exporters do."""
    tensors = index_tensors([str(path)])
    entries = [
        (n, info.dtype, np.array(load_raw(p, info))) for n, (p, info) in tensors.items()
    ]
    embed = next(e for e in entries if e[0] == "model.embed_tokens.weight")
    entries.append(("lm_head.weight", embed[1], embed[2]))
    tmp = path.with_suffix(".tmp")
    write_tensors(tmp, entries)
    tmp.replace(path)


def finetune(tmp_path, variant: str = "untied", dtype: str = "F32"):
    """The tiny Llama and a "full fine-tune" of it: every matrix nudged by
    about 5%. variant ties its embeddings to its output head ("tied"), and
    also stores the head under its own name ("tied-both"); dtype is the
    checkpoints' (F32 or BF16)."""
    base, _, tok = tiny_model(tmp_path)
    if variant != "untied" or dtype != "F32":
        cfg = transformers.AutoConfig.from_pretrained(base)
        cfg.tie_word_embeddings = variant != "untied"
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(cfg)
        if dtype == "BF16":
            model.to(torch.bfloat16)
        base = tmp_path / "variant"
        model.save_pretrained(base)
        tok.save_pretrained(base)
    model = transformers.AutoModelForCausalLM.from_pretrained(base, dtype="auto")
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                noise = torch.randn(p.shape, generator=gen).to(p.dtype)
                p.mul_(1 + 0.05 * noise)
    ft = tmp_path / "ft"
    model.save_pretrained(ft)
    tok.save_pretrained(ft)
    if variant == "tied-both":
        _store_head_too(base / "model.safetensors")
        _store_head_too(ft / "model.safetensors")
    parent = Parent([str(base / "model.safetensors")])
    return base, ft, parent, fine_tune(parent, [str(ft / "model.safetensors")])


@pytest.mark.parametrize(
    "variant,dtype",
    [("untied", "F32"), ("tied", "F32"), ("tied-both", "F32"), ("untied", "BF16")],
)
def test_the_student_is_the_bit1_model(tmp_path, variant, dtype):
    """Before training, the model calibration tunes is exactly the one an
    uncalibrated bit1 layer gives back: tied matrices included, and on a
    CPU, where a BF16 checkpoint computes in float32, rounded as decoding
    rounds."""
    base, ft, parent, target = finetune(tmp_path, variant, dtype)
    assert {info.dtype for _, info in target.values()} == {dtype}
    cal = calibrate.calibrate_bit1(str(ft), parent, target, steps=0, device="cpu")
    matrices = {n for n, (_, info) in target.items() if len(info.shape) == 2}
    assert set(cal.scales) == matrices
    assert cal.loss_before == cal.loss_after > 0
    if variant == "tied-both":
        assert cal.scales["lm_head.weight"] == cal.scales["model.embed_tokens.weight"]

    out, recon = tmp_path / "out", tmp_path / "recon"
    out.mkdir()
    recon.mkdir()
    lossy.encode_lossy(
        lossy.BIT1,
        [str(base / "model.safetensors")],
        [],
        [str(ft / "model.safetensors")],
        out,
        reconstruct_dir=recon,
    )
    teacher, tok = evaluate._load(str(ft), "cpu", None)
    student, _ = evaluate._load(str(recon), "cpu", None)
    encoded = calibrate.calibration_examples(teacher, tok)
    want = calibrate.mean_kl(teacher, student, encoded, tok.pad_token_id)
    assert cal.loss_before == pytest.approx(want, rel=1e-5)


def test_calibration_lowers_the_loss(tmp_path):
    base, ft, parent, target = finetune(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    res = lossy.encode_lossy(
        lossy.BIT1,
        [str(base / "model.safetensors")],
        [],
        [str(ft / "model.safetensors")],
        out,
        calibration_steps=60,
        device="cpu",
    )
    cal = res.calibration
    print(f"KL per reply token {cal.loss_before:.3e} -> {cal.loss_after:.3e}")
    assert cal.loss_after < cal.loss_before
    assert cal.examples == len(calibrate.CALIBRATION_PROMPTS)
    # The layer stores the calibrated scales.
    for name, file in res.tensors:
        if name in cal.scales:
            header = read_header(out / file)
            scale = load_raw(out / file, header[name + lossy.SCALE])[0]
            assert scale == pytest.approx(cal.scales[name], rel=1e-6) and scale > 0


def test_calibration_prompts_are_not_eval_prompts():
    assert not set(calibrate.CALIBRATION_PROMPTS) & set(evaluate.SELF_PROMPTS)
    assert len(set(calibrate.CALIBRATION_PROMPTS)) == len(calibrate.CALIBRATION_PROMPTS)


def test_tied_names_train_one_scale(tmp_path):
    """A head stored under its own name and tied to the embeddings trains
    with them: one scale, and an optimizer that takes it once."""
    base, ft, parent, target = finetune(tmp_path, "tied-both")
    cal = calibrate.calibrate_bit1(str(ft), parent, target, steps=5, device="cpu")
    assert cal.scales["lm_head.weight"] == cal.scales["model.embed_tokens.weight"]
