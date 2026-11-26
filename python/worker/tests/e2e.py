"""Harness for end-to-end tests: the real wpkd and wpk binaries, with this
Python environment as wpkd's build worker.

The binaries come from $WPK_BIN_DIR, else <repo>/dist/apps (built by
``npx nx run-many -t build --projects=wpk,wpkd``). Tests that need them are
skipped when they are missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from weightpack_worker.safetensors_io import load_raw, read_header

REPO = Path(__file__).resolve().parents[3]


def binaries() -> tuple[Path, Path] | None:
    bin_dir = Path(os.environ.get("WPK_BIN_DIR", REPO / "dist" / "apps"))
    wpk, wpkd = bin_dir / "wpk", bin_dir / "wpkd"
    return (wpk, wpkd) if wpk.is_file() and wpkd.is_file() else None


class Weightpack:
    """A running wpkd on a fresh store, and a way to call wpk against it."""

    def __init__(self, wpk: Path, wpkd: Path):
        self.wpk_bin = wpk
        # Unix socket paths are limited to about 100 bytes: keep root short.
        self.root = Path(tempfile.mkdtemp(prefix="wpk"))
        self.socket = self.root / "wpkd.sock"
        self.env = {**os.environ, "WPK_SOCKET": str(self.socket)}
        self.log = open(self.root / "wpkd.log", "wb")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [
                str(wpkd),
                "--root",
                str(self.root),
                "--worker",
                f"{sys.executable} -m weightpack_worker",
                # A free port, so tests never meet a wpkd already running.
                "--gateway",
                "127.0.0.1:0",
                "--engine-reference",
                f"{sys.executable} -m weightpack_worker.engine",
            ],
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 60
        while not self.socket.exists():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"wpkd did not start:\n{self.logs()}")
            time.sleep(0.05)

    def logs(self) -> str:
        return (self.root / "wpkd.log").read_text(errors="replace")

    def run(self, *args: str) -> subprocess.CompletedProcess:
        """Run wpk and return the result, whatever its exit code."""
        return subprocess.run(
            [str(self.wpk_bin), *args],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def wpk(self, *args: str) -> str:
        r = self.run(*args)
        if r.returncode != 0:
            raise AssertionError(
                f"wpk {' '.join(args)} exited {r.returncode}:\n{r.stderr}\n"
                f"wpkd log:\n{self.logs()}"
            )
        return r.stdout

    def inspect(self, ref: str) -> dict:
        return json.loads(self.wpk("inspect", ref))

    def blob_path(self, digest: str) -> Path:
        hexpart = digest.removeprefix("sha256:")
        return self.root / "blobs" / "sha256" / hexpart[:2] / hexpart

    def tensors(self, ref: str) -> dict[str, tuple[str, np.ndarray, str]]:
        """Every tensor of an image's base layer: name -> (dtype, raw values,
        digest of the blob holding it)."""
        out = {}
        for d in self.inspect(ref)["manifest"]["layers"]:
            if not d["mediaType"].startswith("application/vnd.weightpack.tensor.v1"):
                continue
            path = self.blob_path(d["digest"])
            for name, info in read_header(path).items():
                out[name] = (info.dtype, np.array(load_raw(path, info)), d["digest"])
        return out

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def weightpack():
    bins = binaries()
    if bins is None:
        pytest.skip("wpk/wpkd not built (npx nx run-many -t build --projects=wpk,wpkd)")
    wp = Weightpack(*bins)
    yield wp
    wp.close()
