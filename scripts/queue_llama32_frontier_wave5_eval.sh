#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <physical-gpu> <arm-id> <checkpoint-name> <checkpoint-dir>" >&2
  exit 2
fi

GPU="$1"
ARM="$2"
NAME="$3"
CHECKPOINT="$4"
ROOT="/root/QvQ"
LOG="/root/qvq-results/w5eval_${ARM}_${NAME}.eval.log"
GSM="${CHECKPOINT}-gsm8k-platinum-v1.json"
MICRO="${CHECKPOINT}-micro-math-v1.json"

exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] queued evaluation arm=${ARM} gpu=${GPU} checkpoint=${CHECKPOINT}"

if [ ! -f "$CHECKPOINT/qvq_quantize_run.json" ]; then
  echo "[$(date -u +%FT%TZ)] missing quantization completion marker for arm=${ARM}" >&2
  exit 2
fi

echo "[$(date -u +%FT%TZ)] starting GSM8K Platinum arm=${ARM}"
if [ ! -f "$GSM" ]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT/scripts/qvq_evaluate.py" tasks \
      --checkpoint "$CHECKPOINT" \
      --output "$GSM" \
      --task gsm8k_platinum_cot \
      --batch-size 8 \
      --device cuda:0 \
      --attn-implementation 'paged|sdpa'
else
  echo "[$(date -u +%FT%TZ)] GSM8K report already exists; reusing ${GSM}"
fi

echo "[$(date -u +%FT%TZ)] starting Mini-GSM arm=${ARM}"
if [ ! -f "$MICRO" ]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT/scripts/qvq_evaluate.py" micro_math \
      --dense-model /monster/data/model/Llama-3.2-1B-Instruct \
      --checkpoint "$CHECKPOINT" \
      --dataset "$ROOT/dataset/micro_math_llama3.2_1b.jsonl" \
      --manifest "$ROOT/docs/experiments/micro-math-disjointness.json" \
      --rows 64 \
      --rollout-tokens 48 \
      --max-prompt-tokens 2048 \
      --device cuda:0 \
      --dtype float16 \
      --attn-implementation sdpa \
      --output "$MICRO"
else
  echo "[$(date -u +%FT%TZ)] Mini-GSM report already exists; reusing ${MICRO}"
fi
echo "[$(date -u +%FT%TZ)] evaluation complete arm=${ARM}"
