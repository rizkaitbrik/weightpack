"""gRPC server for the wpkd <-> worker contract (proto/weightpack/worker/v1).

Run it as ``weightpack-worker --socket PATH`` (or ``python -m
weightpack_worker``). ``wpkd --worker CMD`` starts it for you. The socket is
created readable and writable by its owner only.
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
from concurrent import futures
from pathlib import Path

import grpc

from weightpack.worker.v1 import worker_pb2, worker_pb2_grpc
from weightpack_worker.delta import encode_xor
from weightpack_worker.squash import LoRALayer, squash

__version__ = "0.1.0"


class Cancelled(Exception):
    """The daemon went away while a router trained."""


class Worker(worker_pb2_grpc.WorkerServicer):
    """Implements Info, Squash, EncodeLayer (XOR) and TrainRouter; Evaluate
    is not done yet and answers UNIMPLEMENTED (the generated base class does
    that)."""

    def Info(self, request, context):  # noqa: N802 (gRPC method name)
        return worker_pb2.InfoResponse(
            version=__version__,
            codecs=[worker_pb2.CODEC_LORA, worker_pb2.CODEC_XOR],
        )

    def EncodeLayer(self, request, context):  # noqa: N802
        if request.codec != worker_pb2.CODEC_XOR:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"codec {worker_pb2.Codec.Name(request.codec)} is not supported yet",
            )
        out_dir = Path(request.out_dir)
        if not out_dir.is_dir():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"out_dir {out_dir} is not a directory",
            )
        try:
            encoded, unchanged = encode_xor(
                [b.path for b in request.base],
                [[b.path for b in layer.blobs] for layer in request.parents],
                list(request.target_files),
                out_dir,
            )
        except (ValueError, OSError, KeyError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return worker_pb2.EncodeLayerResponse(
            tensors=[
                worker_pb2.EncodedTensor(name=name, file=file) for name, file in encoded
            ],
            unchanged=unchanged,
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
            merged = squash(
                [b.path for b in request.base],
                layers,
                out_dir,
                [[b.path for b in layer.blobs] for layer in request.xor_layers],
            )
        except (ValueError, OSError, KeyError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return worker_pb2.SquashResponse(
            tensors=[
                worker_pb2.MergedTensor(name=name, file=file) for name, file in merged
            ]
        )

    def TrainRouter(self, request, context):  # noqa: N802
        """Trains in a thread of its own and streams its progress."""
        try:
            from weightpack_worker import router_train
        except ImportError as exc:  # pragma: no cover - needs a bare install
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"training a router needs torch, transformers and peft "
                f"(pip install 'weightpack-worker[serve]'): {exc}",
            )
        events: queue.Queue = queue.Queue()
        outcome: dict = {}

        def progress(step, steps, loss):
            if not context.is_active():
                raise Cancelled
            event = worker_pb2.TrainRouterProgress(step=step, steps=steps, loss=loss)
            events.put(event)

        def run():
            try:
                outcome["result"] = router_train.train(
                    request.base_dir,
                    [(e.id, e.dir) for e in request.experts],
                    list(request.targets),
                    request.blocks,
                    request.top_k,
                    request.data,
                    request.out_file,
                    router_train.Options(steps=request.steps or 2000),
                    device=request.device or "auto",
                    progress=progress,
                    chat_template=request.chat_template or None,
                )
            except Exception as exc:  # reported to the daemon below
                outcome["error"] = exc
            finally:
                events.put(None)

        threading.Thread(target=run, daemon=True).start()
        while (event := events.get()) is not None:
            yield event
        if "error" in outcome:
            exc = outcome["error"]
            if isinstance(exc, Cancelled):
                return
            code = (
                grpc.StatusCode.FAILED_PRECONDITION
                if isinstance(exc, (ValueError, OSError, KeyError))
                else grpc.StatusCode.INTERNAL
            )
            context.abort(code, f"{type(exc).__name__}: {exc}")
        res = outcome["result"]
        yield worker_pb2.TrainRouterProgress(
            step=request.steps,
            steps=request.steps,
            done=True,
            examples=res.examples,
            heldout=res.heldout,
            arrow_loss=res.arrow_loss,
            trained_loss=res.trained_loss,
            device=res.device,
            seconds=res.seconds,
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
