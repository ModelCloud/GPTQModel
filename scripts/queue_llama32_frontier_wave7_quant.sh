#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 <physical-gpu> <arm-id> <output-name> <config> <slot>" >&2
  exit 2
fi
GPU="$1"; ARM="$2"; NAME="$3"; CONFIG="$4"; SLOT="$5"
MODEL="/monster/data/model/Llama-3.2-1B-Instruct"
CAL="/monster/data/model/dataset/nm-calibration/llm.parquet"
YAQA="/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet"
MANIFEST="/root/QvQ/docs/experiments/disjointness-llama32-benchmark-v2.json"
OUT="/root/qvq-results/llama32-1b-w2-w7queued_${ARM}_llama32_1b_frontier_${NAME}"
LOG="/root/qvq-results/w7queued_${ARM}_llama32_1b_frontier_${NAME}.queue.log"
LOCK="/tmp/qvq-wave7-v2-quant-gpu-${GPU}-slot-${SLOT}.lock"
exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] quant queue arm=${ARM} gpu=${GPU} slot=${SLOT} config=${CONFIG}"

while ! mkdir "$LOCK" 2>/dev/null; do
  echo "[$(date -u +%FT%TZ)] waiting quant slot gpu=${GPU} slot=${SLOT} arm=${ARM}"
  sleep 30
done
echo "$$" >"$LOCK/pid"
cleanup() { : >"$LOCK/pid"; rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT

read -r YAQA_MIN YAQA_SEED < <(python - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h:
    y = json.load(h).get("yaqa", {})
print(int(y.get("minimum_sequences", 0)), int(y.get("seed", 0)))
PY
)
if [ "${YAQA_MIN:-0}" -gt 182 ]; then
  echo "[$(date -u +%FT%TZ)] refusing arm=${ARM}: minimum_sequences=${YAQA_MIN} exceeds YAQA rows=182"
  exit 2
fi
if [ -f "$OUT/qvq_quantize_run.json" ]; then
  echo "[$(date -u +%FT%TZ)] quantization marker already exists arm=${ARM}"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] starting quantization arm=${ARM} gpu=${GPU} slot=${SLOT} yaqa_seed=${YAQA_SEED}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
  python /root/QvQ/scripts/qvq_quantize.py \
    --model "$MODEL" --output "$OUT" --quant-config "/root/QvQ/$CONFIG" \
    --calibration-dataset "$CAL" --calibration-row-start 0 --calibration-rows 128 \
    --yaqa-dataset "$YAQA" --yaqa-row-start 0 --yaqa-rows 182 --device cuda:0 \
    --disjointness-manifest "$MANIFEST" --require-disjointness --no-qvq-telemetry
echo "[$(date -u +%FT%TZ)] quantization complete arm=${ARM}"
