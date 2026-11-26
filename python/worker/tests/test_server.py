"""Tests for the worker's gRPC server, over a real Unix socket."""

import os
import shutil
import stat
import tempfile
from pathlib import Path

import grpc
import numpy as np
import pytest

from weightpack.worker.v1 import worker_pb2, worker_pb2_grpc
from weightpack_worker.safetensors_io import load_raw, read_header, write_tensor
from weightpack_worker.server import serve

from .test_squash import lora_blob


@pytest.fixture
def stub():
    # Unix socket paths are limited to about 100 bytes, so use a short
    # temporary directory rather than pytest's tmp_path.
    d = Path(tempfile.mkdtemp(prefix="wpkw"))
    sock = d / "w.sock"
    server = serve(sock)
    channel = grpc.insecure_channel(f"unix:{sock}")
    yield worker_pb2_grpc.WorkerStub(channel), sock
    channel.close()
    server.stop(grace=None)
    shutil.rmtree(d)


def test_info_and_private_socket(stub):
    s, sock = stub
    info = s.Info(worker_pb2.InfoRequest(), timeout=5)
    assert info.version
    assert list(info.codecs) == [
        worker_pb2.CODEC_LORA,
        worker_pb2.CODEC_XOR,
        worker_pb2.CODEC_BIT1,
        worker_pb2.CODEC_SPARSE,
    ]
    assert stat.S_IMODE(os.stat(sock).st_mode) & 0o077 == 0


def test_squash_over_grpc(stub, tmp_path):
    s, _ = stub
    rng = np.random.default_rng(0)
    base = tmp_path / "w.safetensors"
    write_tensor(
        base, "m.weight", "F32", rng.standard_normal((4, 3)).astype(np.float32)
    )
    blob = lora_blob(
        tmp_path / "l.safetensors",
        "m",
        rng.standard_normal((2, 3)),
        rng.standard_normal((4, 2)),
    )
    out = tmp_path / "out"
    out.mkdir()
    req = worker_pb2.SquashRequest(
        base=[worker_pb2.Blob(path=str(base))],
        layers=[
            worker_pb2.LoRALayer(
                id="x", scaling=1.0, blobs=[worker_pb2.Blob(path=blob)]
            )
        ],
        out_dir=str(out),
    )
    resp = s.Squash(req, timeout=10)
    assert [(t.name, (out / t.file).is_file()) for t in resp.tensors] == [
        ("m.weight", True)
    ]


def test_squash_errors_are_failed_precondition(stub, tmp_path):
    s, _ = stub
    req = worker_pb2.SquashRequest(
        base=[worker_pb2.Blob(path=str(tmp_path / "missing"))], out_dir=str(tmp_path)
    )
    with pytest.raises(grpc.RpcError) as err:
        s.Squash(req, timeout=5)
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    with pytest.raises(grpc.RpcError) as err:
        s.Squash(worker_pb2.SquashRequest(out_dir=str(tmp_path / "nope")), timeout=5)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_evaluate_needs_a_model(stub):
    s, _ = stub
    with pytest.raises(grpc.RpcError) as err:
        s.Evaluate(worker_pb2.EvaluateRequest(), timeout=60)
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "no model" in err.value.details()


def test_lossy_encode_and_squash_over_grpc(stub, tmp_path):
    """A bit1 layer over gRPC: the deltas, the reconstruction, and a squash
    that gives the reconstruction back."""
    s, _ = stub
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    base = tmp_path / "base.safetensors"
    write_tensor(base, "m.weight", "F32", w)
    src = tmp_path / "src"
    src.mkdir()
    write_tensor(src / "model.safetensors", "m.weight", "F32", w + 0.1)
    out, recon, merged = tmp_path / "out", tmp_path / "recon", tmp_path / "merged"
    for d in (out, recon, merged):
        d.mkdir()
    blob = worker_pb2.Blob(path=str(base))
    resp = s.EncodeLayer(
        worker_pb2.EncodeLayerRequest(
            codec=worker_pb2.CODEC_BIT1,
            base=[blob],
            target_files=[str(src / "model.safetensors")],
            out_dir=str(out),
            reconstruct_dir=str(recon),
        ),
        timeout=30,
    )
    assert [t.name for t in resp.tensors] == ["m.weight"] and resp.unchanged == 0
    assert resp.relative_error == pytest.approx(0, abs=1e-6)  # a constant shift
    delta = worker_pb2.Blob(path=str(out / resp.tensors[0].file))
    sq = s.Squash(
        worker_pb2.SquashRequest(
            base=[blob],
            out_dir=str(merged),
            lossy_layers=[worker_pb2.DeltaLayer(id="ft", blobs=[delta])],
        ),
        timeout=30,
    )
    (t,) = sq.tensors
    got = load_raw(merged / t.file, read_header(merged / t.file)["m.weight"])
    np.testing.assert_allclose(np.array(got), w + 0.1, rtol=1e-6)

    with pytest.raises(grpc.RpcError) as err:
        s.EncodeLayer(
            worker_pb2.EncodeLayerRequest(
                codec=worker_pb2.CODEC_LORA, out_dir=str(out)
            ),
            timeout=5,
        )
    assert err.value.code() == grpc.StatusCode.UNIMPLEMENTED
