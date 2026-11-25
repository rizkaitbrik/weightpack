"""Reference serving engine: an OpenAI-compatible server over transformers and
PEFT, on NVIDIA GPUs (cuda), Apple Silicon GPUs (mps) or CPUs.

It serves one base model and any number of LoRA adapters loaded at runtime,
through the same endpoints vLLM exposes with runtime LoRA updating
(``/v1/load_lora_adapter``, ``/v1/unload_lora_adapter``), so wpkd drives it
and vLLM alike. The base and every adapter are separate model names; a
request picks one with its ``model`` field. Requests run one at a time.

It is the engine for machines without vLLM (Macs, CPU servers), and the
reference vLLM is checked against.

    weightpack-engine --model DIR --port 8001 [--device auto] [--adapter name=DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__version__ = "0.1.0"


def pick_device(device: str = "auto") -> str:
    """cuda if there is an NVIDIA GPU, else mps on Apple Silicon, else cpu."""
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check() -> dict:
    """What this engine can run on: its libraries and the devices torch
    sees. wpk doctor reports it."""
    import peft
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    return {
        "engine": __version__,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": cuda,
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if cuda
        else [],
        "mps": torch.backends.mps.is_available(),
    }


class Engine:
    """One base model and its LoRA adapters, ready to generate."""

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        dtype: str = "auto",
        served_name: str = "base",
        chat_template: str | None = None,
    ):
        import torch
        import transformers

        self.torch = torch
        self.device = pick_device(device)
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch_dtype
        ).to(self.device)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        if chat_template:
            self.tokenizer.chat_template = Path(chat_template).read_text()
        self.served_name = served_name
        self.peft = None  # a PeftModel once the first adapter is loaded
        self.adapters: dict[str, str] = {}
        self.lock = threading.Lock()

    # Adapters -----------------------------------------------------------

    def load_adapter(self, name: str, path: str) -> None:
        import peft

        with self.lock:
            if name == self.served_name or name in self.adapters:
                raise ValueError(f"model name {name!r} is already in use")
            if self.peft is None:
                self.peft = peft.PeftModel.from_pretrained(
                    self.model, path, adapter_name=name
                ).to(self.device)
            else:
                self.peft.load_adapter(path, adapter_name=name)
                self.peft.to(self.device)
            self.peft.eval()
            self.adapters[name] = path

    def unload_adapter(self, name: str) -> None:
        with self.lock:
            if name not in self.adapters:
                raise KeyError(name)
            self.peft.delete_adapter(name)
            del self.adapters[name]

    def models(self) -> list[str]:
        return [self.served_name, *self.adapters]

    # Generation ---------------------------------------------------------

    def _model_for(self, name: str):
        """The module to run for a model name, and a context that selects
        the adapter (or turns adapters off for the base)."""
        import contextlib

        if name == self.served_name:
            if self.peft is None:
                return self.model, contextlib.nullcontext()
            return self.peft, self.peft.disable_adapter()
        if name not in self.adapters:
            raise KeyError(name)
        self.peft.set_adapter(name)
        return self.peft, contextlib.nullcontext()

    def encode_chat(
        self, messages: list[dict], chat_template: str | None = None
    ) -> list[int]:
        """Token ids of a chat prompt. chat_template overrides the
        tokenizer's for this request, as vLLM's request field does."""
        out = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            chat_template=chat_template or None,
        )
        return list(out["input_ids"])

    def encode_text(self, prompt: str) -> list[int]:
        return self.tokenizer(prompt)["input_ids"]

    def generate(self, name: str, input_ids: list[int], params: dict, on_text=None):
        """Generate a completion. Returns (text, completion tokens, finish
        reason). With on_text, text pieces are passed to it as they come."""
        import transformers

        torch = self.torch
        max_tokens = int(
            params.get("max_tokens") or params.get("max_completion_tokens") or 256
        )
        temperature = float(params.get("temperature", 1.0))
        kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if kwargs["pad_token_id"] is None:
            kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        if temperature <= 0:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=float(params.get("top_p", 1.0)),
            )
        stop = params.get("stop")
        if stop:
            kwargs["stop_strings"] = [stop] if isinstance(stop, str) else list(stop)
            kwargs["tokenizer"] = self.tokenizer
        ids = torch.tensor([input_ids], device=self.device)
        with self.lock:
            model, ctx = self._model_for(name)
            if params.get("seed") is not None:
                torch.manual_seed(int(params["seed"]))
            with ctx, torch.no_grad():
                if on_text is None:
                    out = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs
                    )
                    new = out[0, ids.shape[1] :].tolist()
                    text = self.tokenizer.decode(new, skip_special_tokens=True)
                else:
                    streamer = transformers.TextIteratorStreamer(
                        self.tokenizer, skip_prompt=True, skip_special_tokens=True
                    )
                    result = {}

                    def run():
                        result["out"] = model.generate(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            streamer=streamer,
                            **kwargs,
                        )

                    t = threading.Thread(target=run)
                    t.start()
                    pieces = []
                    for piece in streamer:
                        if piece:
                            pieces.append(piece)
                            on_text(piece)
                    t.join()
                    new = result["out"][0, ids.shape[1] :].tolist()
                    text = "".join(pieces)
        if stop:
            for s in kwargs["stop_strings"]:
                if s in text:
                    text = text[: text.index(s)]
        eos = self.tokenizer.eos_token_id
        finish = (
            "length"
            if len(new) >= max_tokens and (not new or new[-1] != eos)
            else "stop"
        )
        return text, len(new), finish


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than the default
            sys.stderr.write("engine: " + (fmt % args) + "\n")

        def _json(self, status: int, body) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _text(self, status: int, body: str) -> None:
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, status: int, message: str) -> None:
            self._json(
                status, {"error": {"message": message, "type": "invalid_request_error"}}
            )

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._json(
                    200,
                    {"status": "ok", "engine": "reference", "device": engine.device},
                )
            elif self.path == "/v1/models":
                data = [
                    {
                        "id": m,
                        "object": "model",
                        "owned_by": "weightpack",
                        "parent": None
                        if m == engine.served_name
                        else engine.served_name,
                    }
                    for m in engine.models()
                ]
                self._json(200, {"object": "list", "data": data})
            else:
                self._error(404, f"no route {self.path}")

        def do_POST(self):  # noqa: N802
            try:
                body = self._body()
            except json.JSONDecodeError as exc:
                return self._error(400, f"bad JSON: {exc}")
            try:
                if self.path == "/v1/load_lora_adapter":
                    engine.load_adapter(body["lora_name"], body["lora_path"])
                    return self._text(
                        200,
                        f"Success: LoRA adapter '{body['lora_name']}' added.",
                    )
                if self.path == "/v1/unload_lora_adapter":
                    engine.unload_adapter(body["lora_name"])
                    return self._text(
                        200,
                        f"Success: LoRA adapter '{body['lora_name']}' removed.",
                    )
                if self.path == "/v1/chat/completions":
                    return self._complete(body, chat=True)
                if self.path == "/v1/completions":
                    return self._complete(body, chat=False)
            except KeyError as exc:
                return self._error(404, f"unknown model or field {exc}")
            except (ValueError, OSError) as exc:
                return self._error(400, str(exc))
            self._error(404, f"no route {self.path}")

        def _complete(self, body: dict, chat: bool) -> None:
            name = body.get("model") or engine.served_name
            if name not in engine.models():
                return self._error(
                    404, f"model {name!r} is not loaded (have {engine.models()})"
                )
            ids = (
                engine.encode_chat(body["messages"], body.get("chat_template"))
                if chat
                else engine.encode_text(body["prompt"])
            )
            created, rid = (
                int(time.time()),
                ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
            )
            kind = "chat.completion" if chat else "text_completion"
            if not body.get("stream"):
                text, n, finish = engine.generate(name, ids, body)
                choice = {"index": 0, "finish_reason": finish}
                choice.update(
                    {"message": {"role": "assistant", "content": text}}
                    if chat
                    else {"text": text}
                )
                usage = {
                    "prompt_tokens": len(ids),
                    "completion_tokens": n,
                    "total_tokens": len(ids) + n,
                }
                return self._json(
                    200,
                    {
                        "id": rid,
                        "object": kind,
                        "created": created,
                        "model": name,
                        "choices": [choice],
                        "usage": usage,
                    },
                )

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send(choice):
                chunk = {
                    "id": rid,
                    "object": kind + ".chunk" if chat else kind,
                    "created": created,
                    "model": name,
                    "choices": [choice],
                }
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()

            if chat:
                send(
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                )
            _, _, finish = engine.generate(
                name,
                ids,
                body,
                on_text=lambda piece: send(
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    if chat
                    else {"index": 0, "text": piece, "finish_reason": None}
                ),
            )
            send(
                {"index": 0, "delta": {}, "finish_reason": finish}
                if chat
                else {"index": 0, "text": "", "finish_reason": finish}
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def serve(
    engine: Engine, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Start serving in a background thread and return the server."""
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="weightpack-engine", description=__doc__.splitlines()[0]
    )
    p.add_argument("--model", help="base model directory (Hugging Face layout)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"]
    )
    p.add_argument("--served-model-name", default="base")
    p.add_argument("--chat-template")
    p.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="load a LoRA adapter at start (repeatable)",
    )
    p.add_argument(
        "--exit-with-stdin",
        action="store_true",
        help="exit when stdin closes: the engine never outlives its wpkd",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="print, as JSON, what this engine can run on, and exit",
    )
    args = p.parse_args(argv)
    if args.check:
        print(json.dumps(check()))
        return 0
    if not args.model:
        p.error("--model is required")

    engine = Engine(
        args.model, args.device, args.dtype, args.served_model_name, args.chat_template
    )
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        engine.load_adapter(name, path)
    server = serve(engine, args.host, args.port)
    host, port = server.server_address[:2]
    print(
        f"weightpack-engine {__version__} listening on {host}:{port}"
        f" (device {engine.device})",
        file=sys.stderr,
        flush=True,
    )
    stop = threading.Event()
    if args.exit_with_stdin:

        def watch() -> None:
            with contextlib.suppress(Exception):
                while sys.stdin.buffer.read(4096):
                    pass
            stop.set()

        threading.Thread(target=watch, daemon=True).start()
    with contextlib.suppress(KeyboardInterrupt):
        stop.wait()
    server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
