#!/usr/bin/env bash
# End to end: bit1 fine-tunes served per token through real vLLM, with this
# plugin installed, against weightpack-engine, which serves exactly the
# models decoding gives.
#
# It makes a tiny Llama (tied embeddings, biased attention projections, as
# Qwen2's are, and grouped-query attention) and two
# full fine-tunes of it that change every tensor, norms and biases
# included, imports each as a bit1 layer as wpk build does, and exports it
# as wpkd does. weightpack-engine says what the base and each fine-tune
# generate; then vLLM serves the base with delta slots
# (WEIGHTPACK_DELTA_SLOTS=1), loads both fine-tunes and a LoRA adapter,
# and must give the same replies: one request at a time, then all at once,
# so fine-tune, adapter and base tokens share batches.
#
# Needs uv and jq, and either docker (vLLM's CPU image; IMAGE picks it, as
# in vllm_e2e.sh) or VLLM=<path to a vllm executable> whose environment has
# this plugin installed. DTYPE (default float32) is the dtype vLLM and
# weightpack-engine run in; on a GPU, bfloat16 compares the replies of two
# different roundings, so a few may part ways late.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

IMAGE=${IMAGE:-public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:v0.29.0}
VLLM=${VLLM:-}
DTYPE=${DTYPE:-float32}
NAME=wpk-vllm-bit1-e2e
PORT=${PORT:-18012}
work=$PWD/tmp/vllm-bit1-e2e
rm -rf "$work" && mkdir -p "$work"
srv=/work
if [ -n "$VLLM" ]; then srv=$work; fi
logs() {
  if [ -n "$VLLM" ]; then cat "$work/vllm.log"; else docker logs "$NAME" 2>&1; fi
}
finish() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "== vLLM log (exit $status)"
    logs | grep -vE "^\s*$" | tail -80
  fi
  if [ -n "$VLLM" ]; then
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null
  else
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  return 0
}
trap finish EXIT

echo "== tiny model, two bit1 fine-tunes and what weightpack-engine generates"
uv run --project python/worker --group parity python - "$work" "$DTYPE" <<'PY'
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "python/worker")
import peft
import torch
import transformers
from tests.test_engine import PROMPT, tiny_model
from weightpack_worker import lossy
from weightpack_worker.engine import Engine

work, dtype = Path(sys.argv[1]), sys.argv[2]
_, _, tok = tiny_model(work)
base = work / "base"
shutil.rmtree(base)
torch.manual_seed(0)
transformers.LlamaForCausalLM(transformers.LlamaConfig(
    vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
    attention_bias=True, tie_word_embeddings=True,
    eos_token_id=62, bos_token_id=61, pad_token_id=63,
)).save_pretrained(base)
tok.save_pretrained(base)

# Two full fine-tunes: every matrix, norm and bias changed.
for seed in (1, 2):
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * (p.abs().mean() + 0.02) * torch.randn(p.shape, generator=g))
    model.save_pretrained(work / f"ft{seed}")
    # The layer as wpk build stores it, exported as wpkd exports it.
    out = work / f"delta{seed}"
    out.mkdir()
    res = lossy.encode_lossy(
        lossy.BIT1, [str(base / "model.safetensors")], [],
        [str(work / f"ft{seed}" / "model.safetensors")], out,
    )
    (out / "weightpack_delta.json").write_text(json.dumps(
        {"version": 1, "codec": "bit1", "layer": f"ft{seed}", "files": [f for _, f in res.tensors]}))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 1, "lora_alpha": 1,
        "target_modules": [], "bias": "none", "use_rslora": False,
        "fan_in_fan_out": False, "base_model_name_or_path": "",
    }))
    print(f"ft{seed}: {len(res.tensors)} tensors changed")

# A LoRA adapter too, so adapters and fine-tunes share batches: the first
# seed whose adapter says something of its own.
device = "cpu" if dtype == "float32" else "cuda"
engine = Engine(str(base), device=device, dtype=dtype, served_name="base")
engine.load_adapter("ft1", str(work / "delta1"))
engine.load_adapter("ft2", str(work / "delta2"))
ids = engine.encode_chat(PROMPT)
want = {}
for name in ("base", "ft1", "ft2"):
    want[name], _, _ = engine.generate(name, ids, {"max_tokens": 8, "temperature": 0})
for seed in range(3, 40):
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    cfg = peft.LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    shutil.rmtree(work / "lora", ignore_errors=True)
    peft.get_peft_model(model, cfg).save_pretrained(work / "lora")
    if "lora" in engine.models():
        engine.unload_adapter("lora")
    engine.load_adapter("lora", str(work / "lora"))
    reply, _, _ = engine.generate("lora", ids, {"max_tokens": 8, "temperature": 0})
    if reply and reply not in want.values():
        want["lora"] = reply
        break
(work / "want.json").write_text(json.dumps({"prompt": PROMPT, "replies": want}))
print(json.dumps(want, indent=1))
assert len(set(want.values())) == len(want), "replies are not all different"
PY

serve_args=(--served-model-name base --enable-lora --max-lora-rank 16 --max-loras 4 --dtype "$DTYPE" --max-model-len 128)
if [ -n "${VLLM_ARGS:-}" ]; then
  read -r -a extra <<<"$VLLM_ARGS"
  serve_args+=("${extra[@]}")
fi
if [ -n "$VLLM" ]; then
  echo "== vLLM ($VLLM) with weightpack-vllm and delta slots"
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True WEIGHTPACK_DELTA_SLOTS=1 "$VLLM" serve "$work/base" --port "$PORT" "${serve_args[@]}" \
    >"$work/vllm.log" 2>&1 &
  pid=$!
  alive() { kill -0 "$pid" 2>/dev/null; }
else
  echo "== vLLM ($IMAGE) with weightpack-vllm and delta slots"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" \
    -v "$work:/work" -v "$PWD/python/vllm-plugin:/plugin:ro" \
    -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True -e VLLM_CPU_KVCACHE_SPACE=1 -e WEIGHTPACK_DELTA_SLOTS=1 \
    --entrypoint bash "$IMAGE" -c \
    "pip install -q --no-deps /plugin && exec vllm serve /work/base ${serve_args[*]}" >/dev/null
  alive() { docker ps -q -f name="$NAME" | grep -q .; }
fi
api=http://127.0.0.1:$PORT
for i in $(seq 450); do
  curl -sf "$api/health" >/dev/null && break
  alive || exit 1
  sleep 2
done
curl -sf "$api/health" >/dev/null || exit 1

for pair in ft1:delta1 ft2:delta2 lora:lora; do
  name=${pair%%:*} dir=${pair#*:}
  curl -sf "$api/v1/load_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"$name\", \"lora_path\": \"$srv/$dir\"}" >/dev/null ||
    { echo "load $name failed"; exit 1; }
done

names=(base ft1 ft2 lora)
ask() {
  curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$(jq -c --arg m "$1" \
    '{model: $m, messages: .prompt, max_tokens: 8, temperature: 0}' "$work/want.json")" >"$work/reply-$1-$2.json"
}
fail=0
compare() {
  for name in "${names[@]}"; do
    got=$(jq -r '.choices[0].message.content // ("ERROR " + (.error.message // .message // tostring)) | ltrimstr(" ") | rtrimstr(" ")' "$work/reply-$name-$1.json")
    want=$(jq -r --arg m "$name" '.replies[$m] | ltrimstr(" ") | rtrimstr(" ")' "$work/want.json")
    if [ "$got" = "$want" ]; then mark=same; else mark=DIFFERENT; fail=1; fi
    printf '%-5s %-9s vllm %-30q engine %q\n' "$name" "$mark" "$got" "$want"
  done
}
echo "== replies, one request at a time"
for name in "${names[@]}"; do ask "$name" seq; done
compare seq
echo "== replies, all requests at once (shared batches)"
asks=()
for name in "${names[@]}"; do ask "$name" par & asks+=($!); done
wait "${asks[@]}"
compare par
echo "== plugin log lines"
logs | grep -E "weightpack" | sed 's/^.*weightpack/weightpack/' | sort -u
exit $fail
