#!/usr/bin/env bash
set -euo pipefail

ROOT="${QVQ_FAST_EVAL_ROOT:-/root/QvQ-score-updates}"
PINNED_COMMIT="${QVQ_FAST_EVAL_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
CHECKPOINT="${QVQ_FAST_EVAL_CHECKPOINT:-/root/qvq-results/calibration-fisher-composition-v1/llama32-1b-f9_yaqa182_nm2048_yaqa1x-anchor-up4-l6-l8}"
GPU="${QVQ_FAST_EVAL_GPU:-0}"
OUTPUT="$CHECKPOINT/post_quant_eval_gsm8k_platinum_fast_v2.json"
SUMMARY="$CHECKPOINT/post_quant_eval_gsm8k_platinum_fast_v2_comparison.json"
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

if [ ! -f "$OUTPUT" ]; then
  env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_evaluate.py -- tasks \
      --checkpoint "$CHECKPOINT" --output "$OUTPUT" --task gsm8k_platinum_cot \
      --batch-size 64 --device cuda:0 --attn-implementation 'paged|flash_attention_2' \
      --use-cuda-graph
fi

python - "$OUTPUT" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
result = json.loads(result_path.read_text(encoding="utf-8"))
task = result["tasks"]["gsm8k_platinum_cot"]
accuracy = float(task["metrics"]["acc,num"])
correct = round(accuracy * 1209)
engine = task.get("engine", {})
execution = engine.get("execution", {})
expected_correct = 543
payload = {
    "fast_result": str(result_path),
    "fast_correct": correct,
    "fast_accuracy": accuracy,
    "fast_seconds": task["seconds"],
    "legacy_correct": expected_correct,
    "legacy_seconds": 1069.3343269173056,
    "speedup": 1069.3343269173056 / task["seconds"],
    "metric_parity": correct == expected_correct,
    "engine": engine,
    "continuous_batching_verified": execution.get("generation_backend") == "continuous_batching",
    "paged_attention_verified": execution.get("paged_attention") is True,
    "cuda_graph_requested": result.get("cuda_graph_requested") is True,
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not all(
    (
        payload["metric_parity"],
        payload["continuous_batching_verified"],
        payload["paged_attention_verified"],
        payload["cuda_graph_requested"],
    )
):
    raise SystemExit(f"fast-eval canary failed: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
