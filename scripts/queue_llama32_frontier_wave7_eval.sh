#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <physical-gpu> <arm-id> <output-name> <checkpoint>" >&2
  exit 2
fi
GPU="$1"; ARM="$2"; NAME="$3"; OUT="$4"
MODEL="/monster/data/model/Llama-3.2-1B-Instruct"
LOG="/root/qvq-results/w7eval_${ARM}_${NAME}.eval.log"
LOCK="/tmp/qvq-wave7-v2-eval-gpu-${GPU}.lock"
GSM="${OUT}-gsm8k-platinum-v1.json"
MICRO="${OUT}-micro-math-v1.json"
exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] evaluation queued arm=${ARM} gpu=${GPU} checkpoint=${OUT}"

while [ ! -f "$OUT/qvq_quantize_run.json" ]; do
  echo "[$(date -u +%FT%TZ)] waiting for quantization arm=${ARM}"
  sleep 60
done

# Evaluation uses ~84 GB, so only one model may occupy a physical GPU while
# the two quantizers have finished.  The second Wave-7 arm waits here.
while ! mkdir "$LOCK" 2>/dev/null; do
  echo "[$(date -u +%FT%TZ)] waiting evaluation lock gpu=${GPU} arm=${ARM}"
  sleep 30
done
echo "$$" >"$LOCK/pid"
cleanup() { : >"$LOCK/pid"; rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT

while true; do
  read -r UTIL MEM FREE < <(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits | tr ',' ' ')
  UTIL="${UTIL//[[:space:]]/}"; MEM="${MEM//[[:space:]]/}"; FREE="${FREE//[[:space:]]/}"
  if [ "${UTIL:-100}" -lt 5 ] && [ "${FREE:-0}" -gt 70000 ]; then break; fi
  echo "[$(date -u +%FT%TZ)] waiting free GPU=${GPU} arm=${ARM} util=${UTIL}% mem=${MEM}MiB free=${FREE}MiB"
  sleep 60
done

if [ ! -f "$GSM" ]; then
  echo "[$(date -u +%FT%TZ)] starting GSM8K Platinum arm=${ARM}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python /root/QvQ/scripts/qvq_evaluate.py tasks --checkpoint "$OUT" --output "$GSM" \
      --task gsm8k_platinum_cot --batch-size 8 --device cuda:0 --attn-implementation 'paged|sdpa'
fi
if [ ! -f "$MICRO" ]; then
  echo "[$(date -u +%FT%TZ)] starting Mini-GSM arm=${ARM}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python /root/QvQ/scripts/qvq_evaluate.py micro_math --dense-model "$MODEL" --checkpoint "$OUT" \
      --dataset /root/QvQ/dataset/micro_math_llama3.2_1b.jsonl \
      --manifest /root/QvQ/docs/experiments/micro-math-disjointness.json \
      --rows 64 --rollout-tokens 48 --max-prompt-tokens 2048 --device cuda:0 --dtype float16 \
      --attn-implementation sdpa --output "$MICRO"
fi
echo "[$(date -u +%FT%TZ)] evaluation complete arm=${ARM}"
