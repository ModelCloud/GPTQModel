#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/QvQ"
MODEL="/monster/data/model/Llama-3.2-1B-Instruct"
CAL="/monster/data/model/dataset/nm-calibration/llm.parquet"
YAQA="$ROOT/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet"
MANIFEST="$ROOT/docs/experiments/disjointness-llama32-benchmark-v2.json"

ids=(5b847c 611404 6fe5ca 228564 370c3a 0ea650 186871 37eb97)
names=(
  anchor_up4_l7_l8_l12
  anchor_up4_l6_l7_l8
  anchor_up4_l8_l12_down35_l12
  anchor_up4_l6_l8_down35_l8
  anchor_up4_l8_l11_l12
  anchor_up4_l6_l8_l11
  anchor_up4_l8_l12_down35_l8_l12
  anchor_up4_l6_l8_down35_l6_l8
)

run_arm() {
  local gpu="$1" arm="$2" name="$3"
  local config="$ROOT/scripts/configs/llama32_1b_frontier_w9_${name}.json"
  local out="/root/qvq-results/llama32-1b-w2-w9queued_${arm}_llama32_1b_frontier_${name}"
  local lock="/tmp/qvq-wave9-gpu-${gpu}.lock"
  local gsm="${out}-gsm8k-platinum-v1.json"
  local micro="${out}-micro-math-v1.json"

  echo "[$(date -u +%FT%TZ)] queued Wave-9 arm=${arm} gpu=${gpu} config=${config}"

  while ! mkdir "$lock" 2>/dev/null; do
    if [ -f "$lock/pid" ]; then
      holder=$(cat "$lock/pid" 2>/dev/null || true)
      if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
        rm -f "$lock/pid"
        rmdir "$lock" 2>/dev/null || true
      fi
    fi
    echo "[$(date -u +%FT%TZ)] waiting Wave-9 GPU lock gpu=${gpu} arm=${arm}"
    sleep 30
  done
  # BASHPID identifies this background arm, whereas $$ is shared by all
  # functions spawned from the orchestration shell and cannot detect a stale
  # per-GPU lock safely.
  echo "$BASHPID" >"$lock/pid"
  cleanup_lock() { rm -f "$lock/pid"; rmdir "$lock" 2>/dev/null || true; }
  trap cleanup_lock EXIT

  while true; do
    read -r util mem free < <(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits | tr ',' ' ')
    util="${util//[[:space:]]/}"; mem="${mem//[[:space:]]/}"; free="${free//[[:space:]]/}"
    if [ "${util:-100}" -lt 5 ] && [ "${free:-0}" -gt 70000 ]; then break; fi
    echo "[$(date -u +%FT%TZ)] waiting free gpu=${gpu} arm=${arm} util=${util}% mem=${mem}MiB free=${free}MiB"
    sleep 120
  done

  read -r yaqa_min yaqa_seed < <(python - "$config" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    y = json.load(handle).get("yaqa", {})
print(int(y.get("minimum_sequences", 0)), int(y.get("seed", 0)))
PY
  )
  if [ "${yaqa_min:-0}" -gt 182 ]; then
    echo "[$(date -u +%FT%TZ)] refusing arm=${arm}: minimum_sequences=${yaqa_min} exceeds YAQA rows=182"
    return 2
  fi
  echo "[$(date -u +%FT%TZ)] preflight arm=${arm} yaqa_rows=182 minimum_sequences=${yaqa_min} yaqa_seed=${yaqa_seed}"

  if [ ! -f "$out/qvq_quantize_run.json" ]; then
    echo "[$(date -u +%FT%TZ)] starting quantization arm=${arm}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$ROOT/scripts/qvq_quantize.py" \
        --model "$MODEL" --output "$out" --quant-config "$config" \
        --calibration-dataset "$CAL" --calibration-row-start 0 --calibration-rows 128 \
        --yaqa-dataset "$YAQA" --yaqa-row-start 0 --yaqa-rows 182 --device cuda:0 \
        --disjointness-manifest "$MANIFEST" --require-disjointness --no-qvq-telemetry
  fi

  if [ ! -f "$gsm" ]; then
    echo "[$(date -u +%FT%TZ)] starting GSM8K Platinum arm=${arm}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$ROOT/scripts/qvq_evaluate.py" tasks \
        --checkpoint "$out" --output "$gsm" --task gsm8k_platinum_cot \
        --batch-size 8 --device cuda:0 --attn-implementation 'paged|sdpa'
  fi

  if [ ! -f "$micro" ]; then
    echo "[$(date -u +%FT%TZ)] starting Mini-GSM arm=${arm}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$ROOT/scripts/qvq_evaluate.py" micro_math \
        --dense-model "$MODEL" --checkpoint "$out" \
        --dataset "$ROOT/dataset/micro_math_llama3.2_1b.jsonl" \
        --manifest "$ROOT/docs/experiments/micro-math-disjointness.json" \
        --rows 64 --rollout-tokens 48 --max-prompt-tokens 2048 \
        --device cuda:0 --dtype float16 --attn-implementation sdpa --output "$micro"
  fi
  echo "[$(date -u +%FT%TZ)] Wave-9 arm=${arm} evaluation complete"
}

pids=()
for i in "${!ids[@]}"; do
  run_arm "$i" "${ids[$i]}" "${names[$i]}" \
    >"/root/qvq-results/w9queued_${ids[$i]}_llama32_1b_frontier_${names[$i]}.queue.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "[$(date -u +%FT%TZ)] Wave-9 finished with one or more failed arms"
  exit 1
fi
echo "[$(date -u +%FT%TZ)] Wave-9 orchestration complete"
