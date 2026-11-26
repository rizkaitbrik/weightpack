"""bit1 scale calibration on a tiny model.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from weightpack_worker import calibrate, evaluate, lossy  # noqa: E402
from weightpack_worker.delta import Parent, fine_tune  # noqa: E402
from weightpack_worker.safetensors_io import load_raw, read_header  # noqa: E402

from .test_engine import tiny_model  # noqa: E402


def finetune(tmp_path, tied: bool):
    """The tiny Llama (its embeddings tied to its output head, or not) and a
    "full fine-tune" of it: every matrix nudged by about 5%."""
    base, _, tok = tiny_model(tmp_path)
    if tied:
        cfg = transformers.AutoConfig.from_pretrained(base)
        cfg.tie_word_embeddings = True
        torch.manual_seed(0)
        base = tmp_path / "tied"
        transformers.LlamaForCausalLM(cfg).save_pretrained(base)
        tok.save_pretrained(base)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(1 + 0.05 * torch.randn(p.shape, generator=gen))
    ft = tmp_path / "ft"
    model.save_pretrained(ft)
    tok.save_pretrained(ft)
    parent = Parent([str(base / "model.safetensors")])
    return base, ft, parent, fine_tune(parent, [str(ft / "model.safetensors")])


@pytest.mark.parametrize("tied", [False, True])
def test_the_student_is_the_bit1_model(tmp_path, tied):
    """Before training, the model calibration tunes is exactly the one an
    uncalibrated bit1 layer gives back, tied matrices included."""
    base, ft, parent, target = finetune(tmp_path, tied)
    cal = calibrate.calibrate_bit1(str(ft), parent, target, steps=0, device="cpu")
    matrices = {n for n, (_, info) in target.items() if len(info.shape) == 2}
    assert set(cal.scales) == matrices
    assert cal.loss_before == cal.loss_after > 0

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
    base, ft, parent, target = finetune(tmp_path, tied=False)
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
