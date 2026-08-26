#!/usr/bin/env bash
set -euo pipefail

# Queue one matched reg=0.15 projection-family control on a verified physical
# GPU.  The wrapper waits without consuming GPU memory; qvq_eval_monitor.py
# will launch the canonical D300 and GSM8K Platinum evaluations once the
# qvq_quantize_run.json completion marker appears.
if [ "$#" -ne 4 ]; then
  echo "usage: $0 <physical-gpu> <arm-id> <output-name> <config>" >&2
  exit 2
fi

GPU="$1"
ARM="$2"
NAME="$3"
CONFIG="$4"
MODEL="/monster/data/model/Llama-3.2-1B-Instruct"
OUT="/root/qvq-results/llama32-1b-w2-reg015-${NAME}"
CAL="/monster/data/model/dataset/nm-calibration/llm.parquet"
YAQA="/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet"
MANIFEST="/root/QvQ/docs/experiments/disjointness-llama32-benchmark-v2.json"
LOG="/root/qvq-results/${NAME}.queue.log"

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
echo "[$(date -u +%FT%TZ)] quantization finished arm=${ARM}; monitor will schedule evaluations"
