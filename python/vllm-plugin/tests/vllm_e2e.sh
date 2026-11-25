#!/usr/bin/env bash
# End to end: token routing through real vLLM, with this plugin installed,
# against weightpack-engine's routing.
#
# It makes a tiny Llama, two LoRA experts and a router trained on them,
# asks weightpack-engine what the base, each expert and three mixes of
# them (Arrow top_k 1 and 2, and the trained router) generate, then serves
# the same files with vLLM's CPU image in Docker, loads the experts and the
# mix directories as wpkd writes them, and compares.
#
# Needs docker, uv and jq. IMAGE picks the vLLM image (default: vLLM's
# arm64 CPU release; on x86, public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

IMAGE=${IMAGE:-public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:v0.29.0}
NAME=wpk-vllm-e2e
PORT=${PORT:-18011}
work=$PWD/tmp/vllm-e2e
rm -rf "$work" && mkdir -p "$work"
finish() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "== vLLM log (exit $status)"
    docker logs "$NAME" 2>&1 | grep -vE "^\s*$" | tail -60
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap finish EXIT

echo "== tiny model, experts and what weightpack-engine generates"
uv run --project python/worker --group parity python - "$work" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "python/worker")
import peft
import torch
import transformers
from tests.test_engine import PROMPT, tiny_model
from weightpack_worker import router_train
from weightpack_worker.engine import Engine

work = Path(sys.argv[1])
# The worker's tiny model and tokenizer, with a head size vLLM's CPU
# attention supports (64, not 16).
base, _, tok = tiny_model(work)
torch.manual_seed(0)
transformers.LlamaForCausalLM(transformers.LlamaConfig(
    vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
    tie_word_embeddings=False, eos_token_id=62, bos_token_id=61, pad_token_id=63,
)).save_pretrained(base)
for seed, name in ((0, "one"), (1, "two")):
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_pretrained(base)
    cfg = peft.LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    peft.get_peft_model(model, cfg).save_pretrained(work / name)

# A router trained on text that counts, so it has something to learn.
data = work / "mix.jsonl"
data.write_text("".join(
    json.dumps({"text": " ".join(f"t{(s + step * i) % 60}" for i in range(12))}) + "\n"
    for step in (1, -2) for s in range(0, 60, 2)
))
router_train.train(
    str(base), [("one", str(work / "one")), ("two", str(work / "two"))], ["q", "k", "v", "o"],
    "", 1, str(data), str(work / "router.safetensors"),
    router_train.Options(steps=30, batch_size=8, learning_rate=0.05), device="cpu",
)

# Mix directories, as wpkd writes them, with the container's paths: the
# same experts routed to the top one by Arrow (mix) and by the trained
# router (mix3), and blended over both by Arrow (mix2).
for name, top_k in (("mix", 1), ("mix2", 2), ("mix3", 1)):
    mix = work / name
    mix.mkdir()
    (mix / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"], "bias": "none", "use_rslora": False,
        "fan_in_fan_out": False, "base_model_name_or_path": "",
    }))
    desc = {
        "version": 1, "router": "arrow", "top_k": top_k, "targets": ["q", "k", "v", "o"], "blocks": "",
        "experts": [{"id": "one", "rank": 4, "path": "/work/one"}, {"id": "two", "rank": 4, "path": "/work/two"}],
    }
    if name == "mix3":
        desc.update(router="trained", router_file="/work/router.safetensors")
    (mix / "weightpack_mix.json").write_text(json.dumps(desc))

engine = Engine(str(base), device="cpu", served_name="base")
engine.load_adapter("one", str(work / "one"))
engine.load_adapter("two", str(work / "two"))
engine.load_mix("mix", ["one", "two"], 1, ["q", "k", "v", "o"])
engine.load_mix("mix2", ["one", "two"], 2, ["q", "k", "v", "o"])
engine.load_mix("mix3", ["one", "two"], 1, ["q", "k", "v", "o"], "", "trained", str(work / "router.safetensors"))
ids = engine.encode_chat(PROMPT)
want = {}
for name in ("base", "one", "two", "mix", "mix2", "mix3"):
    want[name], _, _ = engine.generate(name, ids, {"max_tokens": 8, "temperature": 0})
(work / "want.json").write_text(json.dumps({"prompt": PROMPT, "replies": want}))
print(json.dumps(want, indent=1))
# Otherwise matching replies would not show that vLLM routes at all.
assert len(set(want.values())) == len(want), "replies are not all different"
PY

echo "== vLLM ($IMAGE) with weightpack-vllm"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" \
  -v "$work:/work" -v "$PWD/python/vllm-plugin:/plugin:ro" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True -e VLLM_CPU_KVCACHE_SPACE=1 \
  --entrypoint bash "$IMAGE" -c \
  "pip install -q --no-deps /plugin && exec vllm serve /work/base --served-model-name base \
   --enable-lora --max-lora-rank 16 --max-loras 4 --dtype float32 --max-model-len 128" >/dev/null
api=http://127.0.0.1:$PORT
for i in $(seq 300); do
  curl -sf "$api/health" >/dev/null && break
  if ! docker ps -q -f name="$NAME" | grep -q .; then docker logs "$NAME" | tail -40; exit 1; fi
  sleep 2
done
curl -sf "$api/health" >/dev/null || { docker logs "$NAME" | tail -40; exit 1; }

for name in one two mix mix2 mix3; do
  curl -sf "$api/v1/load_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"$name\", \"lora_path\": \"/work/$name\"}" >/dev/null ||
    { echo "load $name failed"; docker logs "$NAME" | tail -30; exit 1; }
done

echo "== replies"
fail=0
for name in base one two mix mix2 mix3; do
  reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$(jq -c --arg m "$name" \
    '{model: $m, messages: .prompt, max_tokens: 8, temperature: 0}' "$work/want.json")")
  # Trimmed: vLLM's detokenizer keeps the first token's leading space.
  got=$(jq -r '.choices[0].message.content // ("ERROR " + (.error.message // .message // tostring)) | ltrimstr(" ") | rtrimstr(" ")' <<<"$reply")
  want=$(jq -r --arg m "$name" '.replies[$m] | ltrimstr(" ") | rtrimstr(" ")' "$work/want.json")
  if [ "$got" = "$want" ]; then mark=same; else mark=DIFFERENT; fail=1; fi
  printf '%-5s %-9s vllm %-30q engine %q\n' "$name" "$mark" "$got" "$want"
done
echo "== plugin log lines"
docker logs "$NAME" 2>&1 | grep -E "weightpack" | sed 's/^.*weightpack/weightpack/' | sort -u
exit $fail
