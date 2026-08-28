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
REPLAY="/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration_div300_sources.parquet"
MANIFEST="/root/QvQ/docs/experiments/disjointness-llama32-benchmark-replay-v2.json"
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

# Fail closed when the requested YAQA slice cannot satisfy the config's
# independent-sequence floor.  Earlier Wave-2 launches used stale configs
# with minimum_sequences=2000 while passing only 182 YAQA rows; that produced
# a late, avoidable failure during Sketch-B collection.  Keep this check in
# the launcher so a stale/generated config can never consume a GPU silently.
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
  echo "[$(date -u +%FT%TZ)] refusing arm=${ARM}: config minimum_sequences=${YAQA_MIN} exceeds YAQA rows=182"
  exit 2
fi
echo "[$(date -u +%FT%TZ)] preflight arm=${ARM} yaqa_rows=182 minimum_sequences=${YAQA_MIN} yaqa_seed=${YAQA_SEED}"

# Replay datasets are valid only for configs that explicitly enable the
# module-granular replay controller.  Passing them to an ordinary precision
# allocation arm is rejected by qvq_quantize.py, so derive the optional CLI
# fragment from the authoritative JSON config instead of guessing from the
# arm name.
REPLAY_ARGS=()
if python - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("module_granular_replay") else 1)
PY
then
  REPLAY_ARGS=(
    --replay-search-dataset "$REPLAY" --replay-search-row-start 0 --replay-search-rows 32
    --replay-confirmation-dataset "$REPLAY" --replay-confirmation-row-start 32 --replay-confirmation-rows 32
  )
fi

echo "[$(date -u +%FT%TZ)] starting arm=${ARM} on physical gpu=${GPU}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
  python /root/QvQ/scripts/qvq_quantize.py \
    --model "$MODEL" \
    --output "$OUT" \
    --quant-config "/root/QvQ/$CONFIG" \
    --calibration-dataset "$CAL" --calibration-row-start 0 --calibration-rows 128 \
    --yaqa-dataset "$YAQA" --yaqa-row-start 0 --yaqa-rows 182 \
    "${REPLAY_ARGS[@]}" \
    --device cuda:0 \
    --disjointness-manifest "$MANIFEST" \
    --require-disjointness \
    --no-qvq-telemetry
echo "[$(date -u +%FT%TZ)] quantization finished arm=${ARM}; monitor will schedule D300 and GSM8K"
