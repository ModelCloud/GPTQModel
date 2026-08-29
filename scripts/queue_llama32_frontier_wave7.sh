#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <physical-gpu> <arm-id> <output-name> <config>" >&2
  exit 2
fi

GPU="$1"
ARM="$2"
NAME="$3"
CONFIG="$4"
MODEL="/monster/data/model/Llama-3.2-1B-Instruct"
CAL="/monster/data/model/dataset/nm-calibration/llm.parquet"
YAQA="/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet"
MANIFEST="/root/QvQ/docs/experiments/disjointness-llama32-benchmark-v2.json"
OUT="/root/qvq-results/llama32-1b-w2-w7queued_${ARM}_llama32_1b_frontier_${NAME}"
LOG="/root/qvq-results/w7queued_${ARM}_llama32_1b_frontier_${NAME}.queue.log"
LOCK="/tmp/qvq-wave7-gpu-${GPU}.lock"

exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] queued arm=${ARM} gpu=${GPU} config=${CONFIG}"

# Two Wave-7 arms are assigned to each physical GPU.  The lock makes the
# second arm wait until the first arm's quantization and held-out evaluations
# have released the device, preventing accidental concurrent model loads.
while ! mkdir "$LOCK" 2>/dev/null; do
  if [ -f "$LOCK/pid" ]; then
    holder=$(cat "$LOCK/pid" 2>/dev/null || true)
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
      rm -f "$LOCK/pid"
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
  fi
  echo "[$(date -u +%FT%TZ)] waiting for Wave-7 GPU lock gpu=${GPU} arm=${ARM}"
  sleep 60
done
echo "$$" >"$LOCK/pid"
cleanup_lock() { rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup_lock EXIT

while true; do
  read -r UTIL MEM FREE < <(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits | tr ',' ' ')
  UTIL="${UTIL//[[:space:]]/}"
  MEM="${MEM//[[:space:]]/}"
  FREE="${FREE//[[:space:]]/}"
  if [ "${UTIL:-100}" -lt 5 ] && [ "${FREE:-0}" -gt 70000 ]; then
    break
  fi
  echo "[$(date -u +%FT%TZ)] waiting arm=${ARM} gpu=${GPU} util=${UTIL}% mem=${MEM}MiB free=${FREE}MiB"
  sleep 120
done

read -r YAQA_MIN YAQA_SEED < <(python - "$CONFIG" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    yaqa = json.load(handle).get("yaqa", {})
print(int(yaqa.get("minimum_sequences", 0)), int(yaqa.get("seed", 0)))
PY
)
if [ "${YAQA_MIN:-0}" -gt 182 ]; then
  echo "[$(date -u +%FT%TZ)] refusing arm=${ARM}: minimum_sequences=${YAQA_MIN} exceeds YAQA rows=182"
  exit 2
fi
echo "[$(date -u +%FT%TZ)] preflight arm=${ARM} yaqa_rows=182 minimum_sequences=${YAQA_MIN} yaqa_seed=${YAQA_SEED}"

if [ ! -f "$OUT/qvq_quantize_run.json" ]; then
  echo "[$(date -u +%FT%TZ)] starting quantization arm=${ARM} on physical gpu=${GPU}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python /root/QvQ/scripts/qvq_quantize.py \
      --model "$MODEL" \
      --output "$OUT" \
      --quant-config "/root/QvQ/$CONFIG" \
      --calibration-dataset "$CAL" --calibration-row-start 0 --calibration-rows 128 \
      --yaqa-dataset "$YAQA" --yaqa-row-start 0 --yaqa-rows 182 \
      --device cuda:0 \
      --disjointness-manifest "$MANIFEST" \
      --require-disjointness \
      --no-qvq-telemetry
fi

GSM="${OUT}-gsm8k-platinum-v1.json"
MICRO="${OUT}-micro-math-v1.json"
if [ ! -f "$GSM" ]; then
  echo "[$(date -u +%FT%TZ)] starting GSM8K Platinum arm=${ARM}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python /root/QvQ/scripts/qvq_evaluate.py tasks \
      --checkpoint "$OUT" --output "$GSM" \
      --task gsm8k_platinum_cot --batch-size 8 --device cuda:0 \
      --attn-implementation 'paged|sdpa'
fi
if [ ! -f "$MICRO" ]; then
  echo "[$(date -u +%FT%TZ)] starting Mini-GSM arm=${ARM}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python /root/QvQ/scripts/qvq_evaluate.py micro_math \
      --dense-model "$MODEL" --checkpoint "$OUT" \
      --dataset /root/QvQ/dataset/micro_math_llama3.2_1b.jsonl \
      --manifest /root/QvQ/docs/experiments/micro-math-disjointness.json \
      --rows 64 --rollout-tokens 48 --max-prompt-tokens 2048 \
      --device cuda:0 --dtype float16 --attn-implementation sdpa \
      --output "$MICRO"
fi
echo "[$(date -u +%FT%TZ)] evaluation complete arm=${ARM}"
