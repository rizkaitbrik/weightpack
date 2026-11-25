"""Tests for the reference serving engine: OpenAI-compatible endpoints over
transformers + PEFT, with adapters loaded at runtime as vLLM does.

Needs the parity dependency group (torch, transformers, peft):
npx nx run worker:parity
"""

import json
import urllib.error
import urllib.request

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

from weightpack_worker.engine import Engine, pick_device, serve  # noqa: E402

PROMPT = [{"role": "user", "content": "t1 t2 t3 t4"}]


def tiny_model(d):
    """A tiny random Llama with a word-level tokenizer and a chat template,
    and a PEFT LoRA over it with random weights."""
    vocab = {f"t{i}": i for i in range(60)}
    vocab.update({"<unk>": 60, "<s>": 61, "</s>": 62, "<pad>": 63})
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tk.decoder = tokenizers.decoders.WordPiece(prefix="##")
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        eos_token_id=62,
        bos_token_id=61,
        pad_token_id=63,
    )
    model = transformers.LlamaForCausalLM(cfg)
    model.save_pretrained(d / "base")
    tok.save_pretrained(d / "base")
    lora = peft.LoraConfig(
        r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )
    peft.get_peft_model(model, lora).save_pretrained(d / "adapter")
    return d / "base", d / "adapter", tok


def post(url, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def direct(base_dir, adapter_dir, tok, max_tokens=8):
    """What transformers (and PEFT) generate, greedily, without the engine."""
    model = transformers.AutoModelForCausalLM.from_pretrained(base_dir).eval()
    if adapter_dir:
        model = peft.PeftModel.from_pretrained(model, adapter_dir).eval()
    ids = torch.tensor(
        [
            tok.apply_chat_template(
                PROMPT, add_generation_prompt=True, return_dict=True
            )["input_ids"]
        ]
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=63,
        )
    return tok.decode(out[0, ids.shape[1] :].tolist(), skip_special_tokens=True)


@pytest.fixture
def running(tmp_path):
    base, adapter, tok = tiny_model(tmp_path)
    engine = Engine(str(base), device="cpu", served_name="tiny")
    server = serve(engine)
    url = "http://{}:{}".format(*server.server_address[:2])
    yield url, base, adapter, tok
    server.shutdown()


def test_endpoints_and_runtime_adapters(running):
    url, base, adapter, tok = running
    assert get(url + "/health")["device"] == "cpu"
    assert [m["id"] for m in get(url + "/v1/models")["data"]] == ["tiny"]

    post(
        url + "/v1/load_lora_adapter",
        {"lora_name": "styled", "lora_path": str(adapter)},
    )
    models = get(url + "/v1/models")["data"]
    assert [(m["id"], m["parent"]) for m in models] == [
        ("tiny", None),
        ("styled", "tiny"),
    ]

    def chat(model, **extra):
        body = {
            "model": model,
            "messages": PROMPT,
            "max_tokens": 8,
            "temperature": 0,
            **extra,
        }
        return json.loads(post(url + "/v1/chat/completions", body))

    base_out, styled_out = chat("tiny"), chat("styled")
    assert base_out["choices"][0]["message"]["content"] == direct(base, None, tok)
    assert styled_out["choices"][0]["message"]["content"] == direct(base, adapter, tok)
    assert (
        base_out["choices"][0]["message"]["content"]
        != styled_out["choices"][0]["message"]["content"]
    )
    assert (
        styled_out["usage"]["completion_tokens"] == 8
        and styled_out["choices"][0]["finish_reason"] == "length"
    )

    # Streaming gives the same text, then [DONE].
    raw = post(
        url + "/v1/chat/completions",
        {
            "model": "styled",
            "messages": PROMPT,
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        },
    )
    events = [
        line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == styled_out["choices"][0]["message"]["content"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"

    comp = json.loads(
        post(
            url + "/v1/completions",
            {"model": "tiny", "prompt": "t5 t6", "max_tokens": 4, "temperature": 0},
        )
    )
    assert comp["object"] == "text_completion" and comp["choices"][0]["text"]

    # A request's own chat template wins over the tokenizer's: this one
    # drops the prompt, so the model sees nothing and the output changes.
    templated = chat("tiny", chat_template="t1 t2 t3")
    assert templated["usage"]["prompt_tokens"] == 3
    assert base_out["usage"]["prompt_tokens"] != 3

    with pytest.raises(urllib.error.HTTPError) as err:
        chat("nope")
    assert err.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as err:
        post(
            url + "/v1/load_lora_adapter",
            {"lora_name": "styled", "lora_path": str(adapter)},
        )
    assert err.value.code == 400  # the name is taken

    # Bad parameters are refused before a stream starts.
    for bad in ({"max_tokens": -1}, {"temperature": -0.5}, {"top_p": 5}):
        with pytest.raises(urllib.error.HTTPError) as err:
            chat("tiny", **bad, stream=True)
        assert err.value.code == 400, bad

    post(url + "/v1/unload_lora_adapter", {"lora_name": "styled"})
    assert [m["id"] for m in get(url + "/v1/models")["data"]] == ["tiny"]


def test_pick_device():
    assert pick_device("cpu") == "cpu"
    want = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert pick_device("auto") == want


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Silicon")
def test_generates_on_metal(tmp_path):
    base, adapter, _ = tiny_model(tmp_path)
    engine = Engine(str(base), device="mps", served_name="tiny")
    engine.load_adapter("styled", str(adapter))
    ids = engine.encode_chat(PROMPT)
    text, n, _ = engine.generate("styled", ids, {"max_tokens": 6, "temperature": 0})
    assert n == 6 and text
    assert next(engine.peft.parameters()).device.type == "mps"


def test_failed_generation_frees_the_engine(tmp_path):
    """A generation that fails mid-stream ends the stream with an error,
    and the engine serves the next request: the lock is released."""
    base, _, tok = tiny_model(tmp_path)
    engine = Engine(str(base), device="cpu", served_name="tiny")
    server = serve(engine)
    url = "http://{}:{}".format(*server.server_address[:2])
    real = engine.model.generate

    def failing(*args, streamer=None, **kwargs):
        streamer.put(torch.tensor([[5]]))  # the prompt, which is skipped
        streamer.put(torch.tensor([7]))
        raise RuntimeError("CUDA out of memory (simulated)")

    try:
        engine.model.generate = failing
        body = {"model": "tiny", "messages": PROMPT, "max_tokens": 8, "temperature": 0}
        raw = post(url + "/v1/chat/completions", {**body, "stream": True})
        events = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]", events
        assert "out of memory" in json.loads(events[-2])["error"]["message"]
        with pytest.raises(urllib.error.HTTPError) as err:
            post(url + "/v1/chat/completions", body)
        assert err.value.code == 500
        engine.model.generate = real
        out = json.loads(post(url + "/v1/chat/completions", body))
        assert out["choices"][0]["message"]["content"] == direct(base, None, tok)
    finally:
        server.shutdown()


def test_stop_filter():
    from weightpack_worker.engine import StopFilter

    def run(stops, pieces):
        out = []
        f = StopFilter(stops, out.append)
        for p in pieces:
            f.feed(p)
        f.close()
        return out

    # A stop string split across pieces is never partly sent.
    assert "".join(run(["END"], ["hello E", "N", "D world"])) == "hello "
    assert run(["END"], ["hello E", "N", "D world"])[0] == "hello "
    # A held-back tail that turns out not to be a stop string is sent.
    assert "".join(run(["END"], ["a E", "x", "y"])) == "a Exy"
    # The first of several stop strings wins.
    assert "".join(run(["zz", "b"], ["aab", "zz"])) == "aa"
    # No stop strings: everything passes at once.
    assert run([], ["a", "b"]) == ["a", "b"]


def test_streamed_stop_matches_unstreamed(running):
    url, *_ = running
    body = {"model": "tiny", "messages": PROMPT, "max_tokens": 8, "temperature": 0}
    full = json.loads(post(url + "/v1/chat/completions", body))
    words = full["choices"][0]["message"]["content"].split()
    stop = words[len(words) // 2]  # a word the reply reaches
    plain = json.loads(post(url + "/v1/chat/completions", {**body, "stop": stop}))
    want = plain["choices"][0]["message"]["content"]
    assert stop not in want
    raw = post(url + "/v1/chat/completions", {**body, "stop": stop, "stream": True})
    chunks = [
        json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")
    ]
    got = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert got == want
