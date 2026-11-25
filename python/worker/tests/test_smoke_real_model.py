"""Smoke test on a real model: Qwen2.5-0.5B-Instruct and three public LoRA
adapters from the Hugging Face Hub, through the whole of Weightpack.

It builds a mix of the adapters, pushes it to a local registry, pulls it
into a second store, squashes one expert, exports, and checks the results
against transformers and PEFT: weights, and greedy generations. It prints a
report of sizes and timings.

Opt-in, since it downloads about 1 GB (cached in <repo>/tmp/smoke):

    npx nx run worker:smoke
"""

import json
import os
import re
import subprocess
import time
import urllib.request

import numpy as np
import pytest

from .e2e import REPO, Weightpack, binaries

pytestmark = pytest.mark.skipif(
    os.environ.get("WPK_SMOKE") != "1",
    reason="set WPK_SMOKE=1 (npx nx run worker:smoke)",
)
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
hub = pytest.importorskip("huggingface_hub")

# Pinned so the test sees the same bytes every time.
BASE = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
ADAPTERS = {
    # rank 16 on every projection
    "python": (
        "ali4566544/qwen2.5-0.5b-instruct-python-lora",
        "5ecafffc2e23deebae61af586617143e596743a1",
    ),
    "sql": (
        "vindows/qwen2.5-0.5b-text-to-sql",
        "427e5eda47e6f5752fab024e54da34ab08d27a13",
    ),
    # rank 8 on q and v only
    "math": (
        "tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora",
        "06abad5db457724a01606284f7a21e0b27f07c56",
    ),
}
PROMPTS = {
    "python": "Write a Python function that reverses a string.",
    "sql": "Write a SQL query that counts the orders per customer.",
    "math": "Tom has 3 apples and buys 5 more. How many apples does he have?",
}


@pytest.fixture(scope="module")
def models():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    root = REPO / "tmp" / "smoke" / "hf"
    out = {}
    repo, rev = BASE
    out["base"] = hub.snapshot_download(repo, revision=rev, local_dir=root / "base")
    for name, (repo, rev) in ADAPTERS.items():
        out[name] = hub.snapshot_download(
            repo,
            revision=rev,
            local_dir=root / name,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        )
    return out


class Registry:
    """A throwaway in-memory OCI registry (go-containerregistry's).

    It logs every request, so its output goes to a file: a pipe nobody
    drains would fill up and stall the registry mid-push.
    """

    def __init__(self, log_path):
        self.log = open(log_path, "w")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [
                "go",
                "run",
                "github.com/google/go-containerregistry/cmd/registry",
                "-port",
                "0",
            ],
            cwd=REPO,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and self.proc.poll() is None:
            m = re.search(r"serving on port (\d+)", log_path.read_text())
            if m:
                self.host = f"localhost:{m.group(1)}"
                return
            time.sleep(0.2)
        self.close()
        raise RuntimeError(f"registry did not start:\n{log_path.read_text()}")

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.log.close()


@pytest.fixture
def registry(tmp_path):
    r = Registry(tmp_path / "registry.log")
    yield r
    r.close()


@pytest.fixture
def stores():
    bins = binaries()
    if bins is None:
        pytest.skip("wpk/wpkd not built")
    # wpkd caches hf:// downloads here, across runs.
    os.environ["WPK_HF_CACHE"] = str(REPO / "tmp" / "smoke" / "hf-cache")
    a, b = Weightpack(*bins), Weightpack(*bins)
    yield a, b
    a.close()
    b.close()


class Report(list):
    """Report lines, printed as they come so a later failure hides none."""

    def append(self, line):
        print(line, flush=True)
        super().append(line)

    def __iadd__(self, lines):
        for line in lines:
            self.append(line)
        return self


class Timer:
    def __init__(self):
        self.rows = []

    def __call__(self, label, fn, *args):
        t = time.monotonic()
        out = fn(*args)
        self.rows.append((label, time.monotonic() - t))
        return out


def post_json(url, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def store_bytes(wp):
    return sum(p.stat().st_size for p in (wp.root / "blobs").rglob("*") if p.is_file())


def bits(t):
    return t.detach().contiguous().view(torch.int16).numpy().view(np.uint16)


def compare(got, want):
    """Share of identical weights, and the largest difference in BF16 steps."""
    assert set(got) == set(want)
    same = total = 0
    worst = 0
    for k, w in want.items():
        g = got[k]
        same += int((bits(g) == bits(w)).sum())
        total += w.numel()
        diff = (g.float() - w.float()).abs()
        step = w.float().abs().clamp_min(1e-30) * 2**-7
        worst = max(worst, float((diff / step).max()))
    return same / total, worst


def generate(model, tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    out = model.generate(**enc, max_new_tokens=32, do_sample=False)
    return out[0, enc["input_ids"].shape[1] :].tolist()


def load(path):
    return transformers.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16
    ).eval()


@torch.no_grad()
def test_real_model_end_to_end(models, registry, stores, tmp_path):
    a, b = stores
    timer = Timer()
    report = Report()
    ref = f"{registry.host}/smoke/qwen-tasks:v1"

    # 1. Build a mix of the three adapters, and a plain stack of one.
    (tmp_path / "Weightfile").write_text(
        f"FROM {models['base']}\nMIX tasks top_k=1\n"
        + "".join(f"  EXPERT {n} {models[n]}\n" for n in ADAPTERS)
    )
    timer("build mix (cold store)", a.wpk, "build", "-t", ref, str(tmp_path))
    mix = a.inspect(ref)
    sizes = {}
    for d in mix["manifest"]["layers"]:
        layer = d["annotations"].get("wpk.layer", "(image)")
        sizes[layer] = sizes.get(layer, 0) + d["size"]
    report += [f"  {k:8s} {v / 1e6:9.1f} MB" for k, v in sizes.items()]
    store_after_mix = store_bytes(a)
    report.append(
        f"  store after mix: {store_after_mix / 1e6:.1f} MB; "
        f"three full copies would be {3 * sizes['base'] / 1e6:.1f} MB"
    )

    # 1b. The same mix straight from the Hub, by hf:// references at the
    #     same commits, gives the same image.
    hub_ctx = tmp_path / "hub"
    hub_ctx.mkdir()
    (hub_ctx / "Weightfile").write_text(
        f"FROM hf://{BASE[0]}@{BASE[1]}\nMIX tasks top_k=1\n"
        + "".join(f"  EXPERT {n} hf://{r}@{rev}\n" for n, (r, rev) in ADAPTERS.items())
    )
    hub_out = timer("build mix from hf://", a.wpk, "build", str(hub_ctx))
    assert mix["digest"] in hub_out
    report.append("  hf:// build gives the same digest as the downloaded files")

    # 2. Push, then pull into a second, empty store.
    timer("push", a.wpk, "push", ref)
    pulled = timer("pull (empty store)", b.wpk, "pull", "--exact", ref)
    assert mix["digest"] in pulled

    # 3. Squash the python expert in the pulled store and export it.
    timer(
        "squash --expert python",
        b.wpk,
        "squash",
        "-t",
        "py-dense",
        "--expert",
        "python",
        ref,
    )
    dense_dir = tmp_path / "py-dense"
    timer("export squashed", b.wpk, "export", "-o", str(dense_dir), "py-dense")

    tok = transformers.AutoTokenizer.from_pretrained(models["base"])
    ours = load(dense_dir / "base")
    theirs = peft.PeftModel.from_pretrained(
        load(models["base"]), models["python"]
    ).merge_and_unload()
    share, worst = compare(ours.state_dict(), theirs.state_dict())
    report.append(
        f"  squash vs PEFT merge (python): {share:.4%} identical,"
        f" worst {worst:.2f} BF16 steps"
    )
    assert worst <= 1.0 and share > 0.99
    gen_ours, gen_theirs = (
        generate(ours, tok, PROMPTS["python"]),
        generate(theirs, tok, PROMPTS["python"]),
    )
    report.append(f"  python, squashed: {tok.decode(gen_ours)!r}")
    assert gen_ours == gen_theirs

    # 4. Export the mix as base + adapters; every adapter loads in PEFT and
    #    generates exactly what the original adapter does.
    split = tmp_path / "split"
    timer("export mix", b.wpk, "export", "-o", str(split), ref)
    base_exported = load(split / "base")
    base_original = load(models["base"])
    want_tokens = {}
    share, worst = compare(base_exported.state_dict(), base_original.state_dict())
    assert share == 1.0, "the exported base must be the original, bit for bit"
    for name in ADAPTERS:
        cfg = json.loads(
            (split / "adapters" / name / "adapter_config.json").read_text()
        )
        mine = peft.PeftModel.from_pretrained(
            load(split / "base"), split / "adapters" / name
        )
        orig = peft.PeftModel.from_pretrained(load(models["base"]), models[name])
        g1, g2 = generate(mine, tok, PROMPTS[name]), generate(orig, tok, PROMPTS[name])
        report.append(
            f"  {name} adapter (target_modules {cfg['target_modules']}):"
            f" {tok.decode(g1)!r}"
        )
        assert g1 == g2, name
        want_tokens[name] = g2

    # 5. The export rebuilds to the same image.
    rebuilt = timer("rebuild from export", b.wpk, "build", str(split))
    assert mix["digest"] in rebuilt

    # 6. Serve the mix: one engine holds the base, each expert is an adapter
    #    in it, and the gateway answers what PEFT generates with the
    #    original adapters. CPU, so the numbers match PEFT's exactly.
    out = timer(
        "wpk run (cpu)", b.wpk, "run", "--device", "cpu", "--name", "tasks", ref
    )
    endpoint = re.search(r"endpoint\s+(\S+)", out).group(1)
    for name in ADAPTERS:
        body = {
            "model": f"tasks/{name}",
            "messages": [{"role": "user", "content": PROMPTS[name]}],
            "max_tokens": 32,
            "temperature": 0,
        }
        t = time.monotonic()
        got = post_json(endpoint + "/chat/completions", body)
        secs = time.monotonic() - t
        text = got["choices"][0]["message"]["content"]
        report.append(f"  served tasks/{name} in {secs:.1f} s: {text!r}")
        assert text == tok.decode(want_tokens[name], skip_special_tokens=True), name
    b.wpk("stop", "tasks")

    # 6b. The same on this machine's GPU, if it has one: Metal or CUDA
    #     kernels round differently, so only check that it generates.
    gpu = "mps" if torch.backends.mps.is_available() else None
    gpu = "cuda" if torch.cuda.is_available() else gpu
    if gpu:
        out = timer(
            f"wpk run ({gpu})", b.wpk, "run", "--device", gpu, "--name", "tasks", ref
        )
        endpoint = re.search(r"endpoint\s+(\S+)", out).group(1)
        t = time.monotonic()
        got = post_json(
            endpoint + "/chat/completions",
            {
                "model": "tasks/sql",
                "messages": [{"role": "user", "content": PROMPTS["sql"]}],
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        text = got["choices"][0]["message"]["content"]
        report.append(
            f"  served tasks/sql on {gpu} in {time.monotonic() - t:.1f} s: {text!r}"
        )
        assert text
        b.wpk("stop", "tasks")

    report.append("  timings:")
    report += [f"    {label:24s} {secs:6.1f} s" for label, secs in timer.rows]
