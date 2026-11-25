#!/usr/bin/env bash
# End-to-end test of the Helm chart on a local kind cluster.
#
# It builds the CPU agent and control plane images, starts a kind cluster
# with two workers and a registry beside it, builds three tiny images with
# wpk (two on one base, one on another), pushes them, installs the chart
# with them in the catalogue, and checks through wpk-control that:
#   - the catalogue's fixed replica starts,
#   - an on-demand model on the same base lands on the same node (base
#     affinity), and one on another base goes to the other node,
#   - chat requests (plain and streamed) are answered.
#
# Needs docker, kind, helm, kubectl, go, uv and jq. KEEP=1 leaves the cluster.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CLUSTER=${CLUSTER:-wpk-e2e}
REGISTRY=$CLUSTER-registry
NS=wpk
work=$(mktemp -d)
pids=()

log() { printf '\n== %s\n' "$*"; }
cleanup() {
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  if [ -z "${KEEP:-}" ]; then
    kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true
    docker rm -f "$REGISTRY" >/dev/null 2>&1 || true
    rm -rf "$work"
  else
    echo "kept: kind cluster $CLUSTER, registry $REGISTRY, files in $work"
  fi
}
trap cleanup EXIT

log "binaries and images"
go build -o "$work/bin/" ./apps/wpk ./apps/wpkd
docker build -q -f docker/Dockerfile --target cpu -t weightpack:e2e-cpu .
docker build -q -f docker/Dockerfile --target control -t weightpack:e2e-control .

log "kind cluster and registry"
kind create cluster --name "$CLUSTER" --config deploy/kind/cluster.yaml --wait 3m
docker run -d --name "$REGISTRY" --network kind -p 127.0.0.1:5001:5000 registry:2 >/dev/null
reg_ip=$(docker inspect -f '{{.NetworkSettings.Networks.kind.IPAddress}}' "$REGISTRY")
kind load docker-image --name "$CLUSTER" weightpack:e2e-cpu weightpack:e2e-control

log "tiny images"
uv run --project python/worker --group parity python - "$work/model" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "python/worker")
import peft
import torch
import transformers
from tests.test_engine import tiny_model

d = Path(sys.argv[1])
d.mkdir(parents=True)
base, adapter, tok = tiny_model(d)
torch.manual_seed(1)
model = transformers.AutoModelForCausalLM.from_pretrained(base)
lora = peft.LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
peft.get_peft_model(model, lora).save_pretrained(d / "adapter2")
PY
export WPK_SOCKET=$work/wpkd.sock
"$work/bin/wpkd" --root "$work/store" --gateway "" >"$work/wpkd.log" 2>&1 &
pids+=($!)
for _ in $(seq 50); do [ -S "$WPK_SOCKET" ] && break; sleep 0.1; done
wpk() { "$work/bin/wpk" "$@"; }
printf 'FROM ./base\nLAYER styled ./adapter\n' >"$work/model/Weightfile"
wpk build -t localhost:5001/e2e/styled:v1 "$work/model" >/dev/null
printf 'FROM ./base\nMIX domain targets=attn top_k=1\n  EXPERT one ./adapter\n  EXPERT two ./adapter2\n' >"$work/model/Weightfile"
wpk build -t localhost:5001/e2e/domains:v1 "$work/model" >/dev/null
# Another base: the styled layer baked in, with adapter2 on top.
printf 'FROM ./base\nLAYER styled ./adapter\nLAYER second ./adapter2\n' >"$work/model/Weightfile"
wpk build -t localhost:5001/e2e/stacked:v1 "$work/model" >/dev/null
for img in styled domains stacked; do wpk push "localhost:5001/e2e/$img:v1" | tail -1; done

log "install the chart"
reg=$reg_ip:5000
helm install wpk deploy/helm/weightpack -n "$NS" --create-namespace --wait --timeout 10m \
  --set image.repository=weightpack --set image.tag=e2e --set image.pullPolicy=Never \
  --set agent.variant=cpu --set agent.gpus=0 --set agent.maxEngines=1 \
  --set agent.storePath=/var/lib/weightpack-e2e \
  --set "agent.insecureRegistries[0]=$reg" --set control.interval=2s --set auth.apiKey=e2e-key \
  --set "catalogue.models[0].name=styled" --set "catalogue.models[0].image=$reg/e2e/styled:v1" --set "catalogue.models[0].replicas=1" \
  --set "catalogue.models[1].name=domains" --set "catalogue.models[1].image=$reg/e2e/domains:v1" \
  --set "catalogue.models[2].name=stacked" --set "catalogue.models[2].image=$reg/e2e/stacked:v1"
kubectl -n "$NS" get pods -o wide

kubectl -n "$NS" port-forward svc/wpk-weightpack 18080:80 >/dev/null 2>&1 &
pids+=($!)
api=http://127.0.0.1:18080
auth=(-H "Authorization: Bearer e2e-key")
for _ in $(seq 50); do curl -sf "${auth[@]}" "$api/healthz" >/dev/null && break; sleep 0.2; done

cluster() { curl -sf "${auth[@]}" "$api/v1/cluster"; }
node_of() { cluster | jq -r --arg m "$1" '[.models[]? | select(.name == $m) | .running[]?] | join(",")'; }
chat() {
  curl -sf "${auth[@]}" "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$1\", \"messages\": [{\"role\": \"user\", \"content\": \"t1 t2 t3\"}], \"max_tokens\": 4, \"temperature\": 0 ${2:-}}"
}

log "the fixed replica starts"
for _ in $(seq 150); do [ -n "$(node_of styled)" ] && break; sleep 2; done
styled_node=$(node_of styled)
[ -n "$styled_node" ] || { echo "styled never started"; cluster; exit 1; }
echo "styled runs on $styled_node"
chat styled | jq -e '.model == "styled"' >/dev/null || { echo "styled did not answer"; exit 1; }
echo "styled answered"

log "an on-demand model on the same base joins its engine"
chat domains/two | jq -e '.model == "domains/two"' >/dev/null || { echo "domains/two did not answer"; exit 1; }
echo "domains/two answered"
[ "$(node_of domains)" = "$styled_node" ] || { echo "domains went to $(node_of domains), not $styled_node"; exit 1; }
echo "domains runs on $styled_node too"

log "a model on another base goes to the other node (one engine per node)"
chat stacked ', "stream": true' | grep -q 'data: \[DONE\]' || { echo "stacked: no streamed reply"; cluster; exit 1; }
stacked_node=$(node_of stacked)
[ -n "$stacked_node" ] && [ "$stacked_node" != "$styled_node" ] || { echo "stacked on '$stacked_node'"; exit 1; }
echo "stacked runs on $stacked_node"

curl -sf "${auth[@]}" "$api/metrics" | grep -E '^wpk_control_(replicas|placements_total)'
log "passed"
