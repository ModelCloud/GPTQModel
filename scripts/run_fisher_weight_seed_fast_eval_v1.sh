#!/usr/bin/env bash
set -euo pipefail

ROOT="${QVQ_FAST_SEED_EVAL_ROOT:-/root/QvQ-score-updates}"
PINNED_COMMIT="${QVQ_FAST_SEED_EVAL_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
RESULTS="${QVQ_FAST_SEED_EVAL_RESULTS:-/root/qvq-results/calibration-fisher-weight-seed-v1}"
CANARY_SUMMARY="/root/qvq-results/calibration-fisher-composition-v1/llama32-1b-f9_yaqa182_nm2048_yaqa1x-anchor-up4-l6-l8/post_quant_eval_gsm8k_platinum_fa2_decode_graph_v4_comparison.json"
WORKBASE=""
WORKTREE=""

ids=(
  f13_yaqa182_nm4096_yaqa15x
  f14_yaqa182_nm4096_yaqa3x
  f15_yaqa182_nm4096_yaqa2x_seed1
  f16_yaqa182_nm4096_yaqa2x_seed2
)
gpus=(0 1 2 3)

while [ ! -f "$CANARY_SUMMARY" ]; do
  echo "[$(date -u +%FT%TZ)] waiting for fast-eval canary acceptance"
  sleep 60
done
python - "$CANARY_SUMMARY" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
required = (
    "metric_parity",
    "continuous_batching_verified",
    "paged_attention_verified",
    "graph_policy_verified",
    "cache_policy_compatible",
)
if not all(payload.get(key) is True for key in required):
    raise SystemExit(f"fast-eval canary was not accepted: {payload}")
PY

cleanup() {
  if [ -n "$WORKTREE" ] && [ -d "$WORKTREE" ]; then
    git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  fi
  if [ -n "$WORKBASE" ] && [ -d "$WORKBASE" ]; then
    rmdir "$WORKBASE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

WORKBASE="$(mktemp -d /tmp/qvq-fast-seed-eval-v1.XXXXXX)"
WORKTREE="$WORKBASE/source"
git -C "$ROOT" worktree add --detach "$WORKTREE" "$PINNED_COMMIT" >/dev/null

gpu_idle() {
  local gpu="$1" uuid apps utilization memory
  uuid="$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  grep -q "^${uuid}," <<<"$apps" && return 1
  IFS=, read -r utilization memory < <(
    nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
  )
  utilization="${utilization//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  [ "$utilization" -eq 0 ] && [ "$memory" -le 256 ]
}

run_one() {
  local index="$1" arm gpu checkpoint legacy baseline fast comparison stable=0
  arm="${ids[$index]}"
  gpu="${gpus[$index]}"
  checkpoint="$RESULTS/llama32-1b-${arm}-anchor-up4-l6-l8"
  legacy="$checkpoint/post_quant_eval_gsm8k_platinum.json"
  baseline="$checkpoint/post_quant_eval_gsm8k_platinum_fa2_graph_off_v4.json"
  fast="$checkpoint/post_quant_eval_gsm8k_platinum_fa2_decode_graph_v4.json"
  comparison="$checkpoint/post_quant_eval_gsm8k_platinum_fa2_decode_graph_v4_comparison.json"

  while [ ! -f "$legacy" ]; do
    echo "[$(date -u +%FT%TZ)] arm=$arm waiting for pinned legacy evaluation"
    sleep 60
  done
  while [ "$stable" -lt 3 ]; do
    if gpu_idle "$gpu"; then
      stable=$((stable + 1))
      sleep 1
    else
      stable=0
      sleep 60
    fi
  done

  if [ ! -f "$baseline" ]; then
    env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_evaluate.py -- tasks \
        --checkpoint "$checkpoint" --output "$baseline" --task gsm8k_platinum_cot \
        --batch-size 64 --device cuda:0 --attn-implementation 'paged|flash_attention_2' \
        --cuda-graph-mode off --no-use-async-batching --max-blocks-per-request 32 \
        --max-batch-tokens 8192
  fi
  if [ ! -f "$fast" ]; then
    env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_evaluate.py -- tasks \
        --checkpoint "$checkpoint" --output "$fast" --task gsm8k_platinum_cot \
        --batch-size 64 --device cuda:0 --attn-implementation 'paged|flash_attention_2' \
        --cuda-graph-mode decode --no-use-async-batching --max-blocks-per-request 32 \
        --max-batch-tokens 8192
  fi
  python - "$baseline" "$fast" "$comparison" <<'PY'
import json
import sys
from pathlib import Path

baseline_path, fast_path, comparison_path = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
fast = json.loads(fast_path.read_text(encoding="utf-8"))
baseline_task = baseline["tasks"]["gsm8k_platinum_cot"]
fast_task = fast["tasks"]["gsm8k_platinum_cot"]
baseline_accuracy = float(baseline_task["metrics"]["acc,num"])
fast_accuracy = float(fast_task["metrics"]["acc,num"])
baseline_engine = baseline_task.get("engine", {})
fast_engine = fast_task.get("engine", {})
baseline_cb = baseline_engine.get("execution", {}).get("continuous_batching_config", {})
fast_cb = fast_engine.get("execution", {}).get("continuous_batching_config", {})
baseline_graphs = baseline_cb.get("cuda_graph_booleans", baseline_cb.get("use_cuda_graph"))
fast_graphs = fast_cb.get("cuda_graph_booleans", fast_cb.get("use_cuda_graph"))
cache_policy_compatible = (
    baseline_cb.get("block_size") == fast_cb.get("block_size")
    and baseline_cb.get("max_batch_tokens") == fast_cb.get("max_batch_tokens")
    and baseline_cb.get("max_blocks_per_request") == fast_cb.get("max_blocks_per_request")
    and baseline_cb.get("allow_block_sharing") == fast_cb.get("allow_block_sharing")
    and baseline_cb.get("use_async_batching") is False
    and fast_cb.get("use_async_batching") is False
    and int(baseline_cb.get("num_blocks", 0)) >= int(baseline_cb.get("max_blocks_per_request", 0))
    and int(fast_cb.get("num_blocks", 0)) >= int(fast_cb.get("max_blocks_per_request", 0))
)
payload = {
    "baseline_result": str(baseline_path),
    "fast_result": str(fast_path),
    "baseline_accuracy": baseline_accuracy,
    "fast_accuracy": fast_accuracy,
    "metric_parity": baseline_accuracy == fast_accuracy,
    "baseline_seconds": baseline_task["seconds"],
    "fast_seconds": fast_task["seconds"],
    "speedup": baseline_task["seconds"] / fast_task["seconds"],
    "continuous_batching_verified": (
        baseline_engine.get("execution", {}).get("generation_backend") == "continuous_batching"
        and fast_engine.get("execution", {}).get("generation_backend") == "continuous_batching"
    ),
    "paged_attention_verified": (
        baseline_engine.get("execution", {}).get("paged_attention") is True
        and fast_engine.get("execution", {}).get("paged_attention") is True
    ),
    "graph_policy_verified": baseline_graphs == [False, False] and fast_graphs == [False, True],
    "cache_policy_compatible": cache_policy_compatible,
    "baseline_graphs": baseline_graphs,
    "fast_graphs": fast_graphs,
    "baseline_engine": baseline_engine,
    "fast_engine": fast_engine,
}
comparison_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not all(
    payload[key]
    for key in (
        "metric_parity",
        "continuous_batching_verified",
        "paged_attention_verified",
        "graph_policy_verified",
        "cache_policy_compatible",
    )
):
    raise SystemExit(f"fast/FA2 graph-off metric mismatch: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

pids=()
for index in "${!ids[@]}"; do
  run_one "$index" >"$RESULTS/${ids[$index]}_fast_eval.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
