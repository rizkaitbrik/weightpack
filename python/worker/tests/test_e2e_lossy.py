"""End to end: a lossy import of a full fine-tune, gated on an eval.

Through the real wpk and wpkd, with this worker: a fine-tune of the tiny
Llama is stored as a bit1 delta; the build scores it against the fine-tune
on the fine-tune's own replies and records the drop; and squash gives back
exactly the model the eval scored.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from weightpack_worker import lossy  # noqa: E402
from weightpack_worker.safetensors_io import index_tensors, load_raw  # noqa: E402

from .test_engine import tiny_model  # noqa: E402


def finetune(tmp_path):
    """The tiny Llama, and a "full fine-tune" of it: every matrix nudged by
    about 1%."""
    base, _, tok = tiny_model(tmp_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(1 + 0.01 * torch.randn(p.shape, generator=gen))
    ft = tmp_path / "ft"
    model.save_pretrained(ft)
    tok.save_pretrained(ft)
    return base, ft


def test_bit1_import(weightpack, tmp_path):
    base, ft = finetune(tmp_path)
    (tmp_path / "Weightfile").write_text(
        "FROM ./base\nLAYER ft ./ft codec=bit1 max_drop=100\n"
    )
    out = weightpack.wpk("build", "-t", "ft", str(tmp_path))
    assert "bit1 delta:" in out and "a drop of" in out, out

    detail = weightpack.inspect("ft")
    layer = next(lr for lr in detail["config"]["layers"] if lr["id"] == "ft")
    fid = layer["fidelity"]
    assert layer["kind"] == "delta.bit1"
    assert fid["codec"] == "bit1" and fid["eval"] == "self" and fid["max_drop"] == 100
    assert fid["source_digest"].startswith("sha256:")
    assert fid["measured_drop"] == pytest.approx(fid["source_score"] - fid["score"])
    blobs = [
        d
        for d in detail["manifest"]["layers"]
        if d["annotations"].get("wpk.layer") == "ft"
    ]
    assert {d["mediaType"] for d in blobs} == {
        "application/vnd.weightpack.delta.bit1.v1"
    }
    tensors = index_tensors([str(ft / "model.safetensors")])
    matrices = {n for n, (_, info) in tensors.items() if len(info.shape) == 2}
    assert {d["annotations"]["wpk.tensor"] for d in blobs} == matrices
    raw = sum(tensors[n][1].size for n in matrices)
    stored = sum(d["size"] for d in blobs)
    print(f"bit1 deltas: {raw} bytes of matrices stored in {stored}")
    assert stored < 0.05 * raw  # one bit per 32-bit weight, plus headers

    # Squash gives back exactly the model the eval scored.
    weightpack.wpk("squash", "-t", "ft-dense", "ft")
    got = weightpack.tensors("ft-dense")
    out_dir, recon = tmp_path / "out", tmp_path / "recon"
    out_dir.mkdir()
    recon.mkdir()
    lossy.encode_lossy(
        lossy.BIT1,
        [str(base / "model.safetensors")],
        [],
        [str(ft / "model.safetensors")],
        out_dir,
        reconstruct_dir=recon,
    )
    want = index_tensors([str(p) for p in recon.glob("model-*.safetensors")])
    assert set(got) == set(want)
    for name, (path, info) in want.items():
        assert got[name][1].tobytes() == np.array(load_raw(path, info)).tobytes(), name
