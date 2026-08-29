#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/QvQ"
MODEL="/monster/data/model/Llama-3.2-1B-Instruct"
CAL="/monster/data/model/dataset/nm-calibration/llm.parquet"
YAQA="$ROOT/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet"
MANIFEST="$ROOT/docs/experiments/disjointness-llama32-benchmark-v2.json"
ids=(fb8247 735365 e5488a 6f37dc 9cd131 fdbd68 91a008 4e58f5 c4cc84 cef938 fe2ba7 446290 719a90 cecb41 1e3a24 864bb1)
names=(anchor_up4_l5_l12 anchor_up4_l5_l15 anchor_up4_l12_l15 anchor_up4_l9_l12 anchor_up4_l9_l15 anchor_up4_l6_l8 anchor_up4_l8_l12 anchor_up4_l5_l6 anchor_up4_l6_l12 anchor_up4_l6_l15 anchor_up4_l5_l7 anchor_up4_l7_l12 anchor_up4_l7_l15 anchor_up4_l4_l5 anchor_up4_l4_l12 anchor_up4_l4_l8)

run_arm() {
  local gpu="$1" arm="$2" name="$3" config="$4" slot="$5"
  local out="/root/qvq-results/llama32-1b-w2-w8queued_${arm}_llama32_1b_frontier_${name}"
  local log="/root/qvq-results/w8queued_${arm}_llama32_1b_frontier_${name}.queue.log"
  local lock="/tmp/qvq-wave8-gpu-${gpu}.lock"
  local gsm="${out}-gsm8k-platinum-v1.json" micro="${out}-micro-math-v1.json"
  echo "[$(date -u +%FT%TZ)] queued Wave-8 arm=${arm} gpu=${gpu} slot=${slot}"
  while ! mkdir "$lock" 2>/dev/null; do
    if [ -f "$lock/pid" ]; then
      holder=$(cat "$lock/pid" 2>/dev/null || true)
      if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then rm -f "$lock/pid"; rmdir "$lock" 2>/dev/null || true; fi
    fi
    sleep 60
  done
  echo "$BASHPID" >"$lock/pid"
  cleanup() { rm -f "$lock/pid"; rmdir "$lock" 2>/dev/null || true; }
  trap cleanup EXIT
  while true; do
    read -r util _ free < <(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits | tr ',' ' ')
    util="${util//[[:space:]]/}"; free="${free//[[:space:]]/}"
    if [ "${util:-100}" -lt 5 ] && [ "${free:-0}" -gt 70000 ]; then break; fi
    sleep 120
  done
  read -r yaqa_min yaqa_seed < <(python - "$config" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as h: y=json.load(h).get("yaqa", {})
print(int(y.get("minimum_sequences", 0)), int(y.get("seed", 0)))
PY
  )
  if [ "${yaqa_min:-0}" -gt 182 ]; then echo "refusing ${arm}: minimum_sequences=${yaqa_min}"; return 2; fi
  if [ ! -f "$out/qvq_quantize_run.json" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/qvq_quantize.py" \
      --model "$MODEL" --output "$out" --quant-config "$ROOT/$config" \
      --calibration-dataset "$CAL" --calibration-row-start 0 --calibration-rows 128 \
      --yaqa-dataset "$YAQA" --yaqa-row-start 0 --yaqa-rows 182 --device cuda:0 \
      --disjointness-manifest "$MANIFEST" --require-disjointness --no-qvq-telemetry
  fi
  if [ ! -f "$gsm" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/qvq_evaluate.py" tasks \
      --checkpoint "$out" --output "$gsm" --task gsm8k_platinum_cot --batch-size 8 --device cuda:0 --attn-implementation 'paged|sdpa'
  fi
  if [ ! -f "$micro" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/qvq_evaluate.py" micro_math \
      --dense-model "$MODEL" --checkpoint "$out" --dataset "$ROOT/dataset/micro_math_llama3.2_1b.jsonl" \
      --manifest "$ROOT/docs/experiments/micro-math-disjointness.json" --rows 64 --rollout-tokens 48 \
      --max-prompt-tokens 2048 --device cuda:0 --dtype float16 --attn-implementation sdpa --output "$micro"
  fi
  echo "[$(date -u +%FT%TZ)] Wave-8 arm=${arm} evaluation complete"
}

pids=()
for i in "${!ids[@]}"; do
  gpu=$((i % 8)); slot=$((i / 8))
  log="/root/qvq-results/w8queued_${ids[$i]}_llama32_1b_frontier_${names[$i]}.queue.log"
  run_arm "$gpu" "${ids[$i]}" "${names[$i]}" "scripts/configs/llama32_1b_frontier_w8_${names[$i]}.json" "$slot" >"$log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
