#!/usr/bin/env bash
set -euo pipefail

ROOT="${QVQ_FAST_EVAL_ROOT:-/root/QvQ-score-updates}"
PINNED_COMMIT="${QVQ_FAST_EVAL_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
CHECKPOINT="${QVQ_FAST_EVAL_CHECKPOINT:-/root/qvq-results/calibration-fisher-composition-v1/llama32-1b-f9_yaqa182_nm2048_yaqa1x-anchor-up4-l6-l8}"
GPU="${QVQ_FAST_EVAL_GPU:-0}"
MAX_ROWS="${QVQ_FAST_EVAL_MAX_ROWS:-128}"
BASELINE_OUTPUT="$CHECKPOINT/post_quant_eval_gsm8k_platinum_fa2_graph_off_v4.json"
CANDIDATE_OUTPUT="$CHECKPOINT/post_quant_eval_gsm8k_platinum_fa2_decode_graph_v4.json"
SUMMARY="$CHECKPOINT/post_quant_eval_gsm8k_platinum_fa2_decode_graph_v4_comparison.json"
WORKBASE=""
WORKTREE=""

export CUDA_DEVICE_ORDER=PCI_BUS_ID

gpu_idle() {
  local uuid apps utilization memory
  uuid="$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  grep -q "^${uuid}," <<<"$apps" && return 1
  IFS=, read -r utilization memory < <(
    nvidia-smi --id="$GPU" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
  )
  utilization="${utilization//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  [ "$utilization" -eq 0 ] && [ "$memory" -le 256 ]
}

stable=0
while [ "$stable" -lt 3 ]; do
  if gpu_idle; then
    stable=$((stable + 1))
    echo "[$(date -u +%FT%TZ)] GPU $GPU canary idle sample=$stable/3"
    sleep 1
  else
    stable=0
    echo "[$(date -u +%FT%TZ)] fast-eval canary waiting for GPU $GPU"
    sleep 60
  fi
done

cleanup() {
  if [ -n "$WORKTREE" ] && [ -d "$WORKTREE" ]; then
    git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  fi
  if [ -n "$WORKBASE" ] && [ -d "$WORKBASE" ]; then
    rmdir "$WORKBASE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

WORKBASE="$(mktemp -d /tmp/qvq-fast-eval-v1.XXXXXX)"
WORKTREE="$WORKBASE/source"
git -C "$ROOT" worktree add --detach "$WORKTREE" "$PINNED_COMMIT" >/dev/null

if [ ! -f "$BASELINE_OUTPUT" ]; then
  env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_evaluate.py -- tasks \
      --checkpoint "$CHECKPOINT" --output "$BASELINE_OUTPUT" --task gsm8k_platinum_cot \
      --batch-size 64 --device cuda:0 --attn-implementation 'paged|flash_attention_2' \
      --cuda-graph-mode off --no-use-async-batching --max-batch-tokens 2048 \
      --max-blocks-per-request 32 --max-rows "$MAX_ROWS"
fi

if [ ! -f "$CANDIDATE_OUTPUT" ]; then
  env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_evaluate.py -- tasks \
      --checkpoint "$CHECKPOINT" --output "$CANDIDATE_OUTPUT" --task gsm8k_platinum_cot \
      --batch-size 64 --device cuda:0 --attn-implementation 'paged|flash_attention_2' \
      --cuda-graph-mode decode --no-use-async-batching --max-blocks-per-request 32 \
      --max-batch-tokens 2048 --max-rows "$MAX_ROWS"
fi

python - "$BASELINE_OUTPUT" "$CANDIDATE_OUTPUT" "$SUMMARY" "$MAX_ROWS" <<'PY'
import json
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
max_rows = int(sys.argv[4])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
baseline_task = baseline["tasks"]["gsm8k_platinum_cot"]
candidate_task = candidate["tasks"]["gsm8k_platinum_cot"]
baseline_accuracy = float(baseline_task["metrics"]["acc,num"])
candidate_accuracy = float(candidate_task["metrics"]["acc,num"])
baseline_correct = round(baseline_accuracy * max_rows)
candidate_correct = round(candidate_accuracy * max_rows)
baseline_engine = baseline_task.get("engine", {})
candidate_engine = candidate_task.get("engine", {})
baseline_execution = baseline_engine.get("execution", {})
candidate_execution = candidate_engine.get("execution", {})
baseline_cb = baseline_execution.get("continuous_batching_config", {})
candidate_cb = candidate_execution.get("continuous_batching_config", {})
baseline_graphs = baseline_cb.get("cuda_graph_booleans", baseline_cb.get("use_cuda_graph"))
candidate_graphs = candidate_cb.get("cuda_graph_booleans", candidate_cb.get("use_cuda_graph"))
payload = {
    "baseline_result": str(baseline_path),
    "candidate_result": str(candidate_path),
    "rows": max_rows,
    "baseline_correct": baseline_correct,
    "candidate_correct": candidate_correct,
    "baseline_accuracy": baseline_accuracy,
    "candidate_accuracy": candidate_accuracy,
    "baseline_seconds": baseline_task["seconds"],
    "candidate_seconds": candidate_task["seconds"],
    "speedup": baseline_task["seconds"] / candidate_task["seconds"],
    "metric_parity": baseline_accuracy == candidate_accuracy,
    "continuous_batching_verified": (
        baseline_execution.get("generation_backend") == "continuous_batching"
        and candidate_execution.get("generation_backend") == "continuous_batching"
    ),
    "paged_attention_verified": (
        baseline_execution.get("paged_attention") is True
        and candidate_execution.get("paged_attention") is True
    ),
    "baseline_graphs": baseline_graphs,
    "candidate_graphs": candidate_graphs,
    "graph_policy_verified": baseline_graphs == [False, False] and candidate_graphs == [False, True],
    "cache_policy_compatible": (
        baseline_cb.get("block_size") == candidate_cb.get("block_size")
        and baseline_cb.get("max_batch_tokens") == candidate_cb.get("max_batch_tokens")
        and baseline_cb.get("max_blocks_per_request") == candidate_cb.get("max_blocks_per_request")
        and baseline_cb.get("allow_block_sharing") == candidate_cb.get("allow_block_sharing")
        and baseline_cb.get("use_async_batching") is False
        and candidate_cb.get("use_async_batching") is False
        and int(baseline_cb.get("num_blocks", 0)) >= int(baseline_cb.get("max_blocks_per_request", 0))
        and int(candidate_cb.get("num_blocks", 0)) >= int(candidate_cb.get("max_blocks_per_request", 0))
    ),
    "baseline_engine": baseline_engine,
    "candidate_engine": candidate_engine,
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not all(
    (
        payload["metric_parity"],
        payload["continuous_batching_verified"],
        payload["paged_attention_verified"],
        payload["graph_policy_verified"],
        payload["cache_policy_compatible"],
    )
):
    raise SystemExit(f"fast-eval canary failed: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
