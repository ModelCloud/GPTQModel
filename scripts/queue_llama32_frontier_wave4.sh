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
OUT="/root/qvq-results/llama32-1b-w2-w4queued_${ARM}_llama32_1b_frontier_${NAME}"
LOG="/root/qvq-results/w4queued_${ARM}_llama32_1b_frontier_${NAME}.queue.log"

exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] queued arm=${ARM} gpu=${GPU} config=${CONFIG}"

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

if [ -e "$OUT/qvq_quantize_run.json" ]; then
  echo "[$(date -u +%FT%TZ)] completion marker already exists; not rerunning ${ARM}"
  exit 0
fi

read -r YAQA_MIN YAQA_SEED < <(python - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
yaqa = payload.get("yaqa", {})
print(int(yaqa.get("minimum_sequences", 0)), int(yaqa.get("seed", 0)))
PY
)
if [ "${YAQA_MIN:-0}" -gt 182 ]; then
  echo "[$(date -u +%FT%TZ)] refusing arm=${ARM}: minimum_sequences=${YAQA_MIN} exceeds YAQA rows=182"
  exit 2
fi
echo "[$(date -u +%FT%TZ)] preflight arm=${ARM} yaqa_rows=182 minimum_sequences=${YAQA_MIN} yaqa_seed=${YAQA_SEED}"

echo "[$(date -u +%FT%TZ)] starting arm=${ARM} on physical gpu=${GPU}"
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
echo "[$(date -u +%FT%TZ)] quantization finished arm=${ARM}; evaluation can be scheduled"
