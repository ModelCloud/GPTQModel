#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 <physical-gpu> <arm-id> <output-name> <config> <smooth|plain>" >&2
  exit 2
fi

GPU="$1"
ARM="$2"
NAME="$3"
CONFIG="$4"
MODE="$5"
MODEL="/monster/data/model/Llama-3.2-1B-Instruct"
OUT="/root/qvq-results/llama32-1b-w2-${NAME}"
CAL="/monster/data/model/dataset/nm-calibration/llm.parquet"
YAQA="/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet"
REPLAY="/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration_div300_sources_disjoint.parquet"
MANIFEST="/root/QvQ/docs/experiments/disjointness-div300-sources-disjoint.json"
LOG="/root/qvq-results/${NAME}.queue.log"
ATOMIC_OUT="/root/qvq-results/llama32-1b-w2-atomic-swiglu-tip-3f17c40b"

exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] queued arm=${ARM} gpu=${GPU} mode=${MODE} config=${CONFIG}"

# The combined arm must not start before the plain Atomic arm has produced a
# complete checkpoint.  This preserves the requested queue ordering.
if [ "$MODE" = smooth ]; then
  while [ ! -f "$ATOMIC_OUT/qvq_quantize_run.json" ]; do
    echo "[$(date -u +%FT%TZ)] waiting arm=${ARM} for ${ATOMIC_OUT}/qvq_quantize_run.json"
    sleep 120
  done
fi

while true; do
  read -r UTIL MEM < <(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' ')
  UTIL="${UTIL//[[:space:]]/}"
  MEM="${MEM//[[:space:]]/}"
  if [ "${UTIL:-100}" -lt 5 ] && [ "${MEM:-999999}" -lt 2000 ]; then
    break
  fi
  echo "[$(date -u +%FT%TZ)] waiting arm=${ARM} gpu=${GPU} util=${UTIL}% mem=${MEM}MiB"
  sleep 120
done

if [ -e "$OUT/qvq_quantize_run.json" ]; then
  echo "[$(date -u +%FT%TZ)] completion marker already exists; not rerunning ${ARM}"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] starting arm=${ARM} on physical gpu=${GPU}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
  python /root/QvQ/scripts/qvq_quantize.py \
    --model "$MODEL" \
    --output "$OUT" \
    --quant-config "/root/QvQ/$CONFIG" \
    --calibration-dataset "$CAL" --calibration-row-start 0 --calibration-rows 128 \
    --yaqa-dataset "$YAQA" --yaqa-row-start 0 --yaqa-rows 182 \
    --replay-search-dataset "$REPLAY" --replay-search-row-start 0 --replay-search-rows 32 \
    --replay-confirmation-dataset "$REPLAY" --replay-confirmation-row-start 32 --replay-confirmation-rows 32 \
    --device cuda:0 \
    --disjointness-manifest "$MANIFEST" \
    --no-qvq-telemetry
echo "[$(date -u +%FT%TZ)] quantization finished arm=${ARM}; monitor will schedule D300 and GSM8K"
