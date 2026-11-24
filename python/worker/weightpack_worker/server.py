"""gRPC server for the wpkd <-> worker contract (proto/weightpack/worker/v1).

Run it as ``weightpack-worker --socket PATH`` (or ``python -m
weightpack_worker``). ``wpkd --worker CMD`` starts it for you. The socket is
created readable and writable by its owner only.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from concurrent import futures
from pathlib import Path

import grpc

from weightpack.worker.v1 import worker_pb2, worker_pb2_grpc
from weightpack_worker.squash import LoRALayer, squash

__version__ = "0.1.0"


class Worker(worker_pb2_grpc.WorkerServicer):
    """Implements Info and Squash; EncodeLayer and Evaluate are not done yet
    and answer UNIMPLEMENTED (the generated base class does that)."""

    def Info(self, request, context):  # noqa: N802 (gRPC method name)
        return worker_pb2.InfoResponse(
            version=__version__, codecs=[worker_pb2.CODEC_LORA]
        )

    def Squash(self, request, context):  # noqa: N802
        out_dir = Path(request.out_dir)
        if not out_dir.is_dir():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"out_dir {out_dir} is not a directory",
            )
        layers = [
            LoRALayer(
                id=layer.id, scaling=layer.scaling, blobs=[b.path for b in layer.blobs]
            )
            for layer in request.layers
        ]
        try:
            merged = squash([b.path for b in request.base], layers, out_dir)
        except (ValueError, OSError, KeyError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return worker_pb2.SquashResponse(
            tensors=[
                worker_pb2.MergedTensor(name=name, file=file) for name, file in merged
            ]
        )


def serve(socket: str | Path, max_workers: int = 2) -> grpc.Server:
    """Start a worker server on a Unix socket and return it."""
    socket = Path(socket).absolute()
    socket.unlink(missing_ok=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    worker_pb2_grpc.add_WorkerServicer_to_server(Worker(), server)
    # Create the socket private to its owner, with no window in which
    # someone else could connect.
    old = os.umask(0o077)
    try:
        server.add_insecure_port(f"unix:{socket}")
        server.start()
    finally:
        os.umask(old)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weightpack-worker", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--socket", required=True, help="Unix socket to listen on")
    args = parser.parse_args(argv)

    server = serve(args.socket)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    print(
        f"weightpack-worker {__version__} listening on {args.socket}",
        file=sys.stderr,
        flush=True,
    )
    stop.wait()
    server.stop(grace=2).wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
