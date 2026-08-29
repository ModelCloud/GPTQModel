#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/QvQ"
ids=(e62172 4bab0e 6fba6c 3ff286 e1b702 788234 5f91ab 5ea33c b8fb5c d693ff d6fab2 bd7c59 bb10df adff86 254d84 515065)
names=(anchor_up4_l0 anchor_up4_l1 anchor_up4_l2 anchor_up4_l3 anchor_up4_l4 anchor_up4_l5 anchor_up4_l6 anchor_up4_l7 anchor_up4_l8 anchor_up4_l9 anchor_up4_l10 anchor_up4_l11 anchor_up4_l12 anchor_up4_l13 anchor_up4_l14 anchor_up4_l15)

echo "[$(date -u +%FT%TZ)] launching Wave-7 quantization (two sessions per GPU)"
pipelines=()
for i in "${!ids[@]}"; do
  gpu=$((i % 8)); slot=$((i / 8))
  arm="${ids[$i]}"; name="${names[$i]}"
  config="scripts/configs/llama32_1b_frontier_w7_${name}.json"
  out="/root/qvq-results/llama32-1b-w2-w7queued_${arm}_llama32_1b_frontier_${name}"

  # Keep each arm's evaluation attached to its quantizer.  The evaluation
  # wrapper waits for the checkpoint marker, an exclusive GPU lock, and the
  # required free memory, so it can overlap with quantization on other GPUs
  # without competing with a live quantizer on the same device.
  (
    if "$ROOT/scripts/queue_llama32_frontier_wave7_quant.sh" "$gpu" "$arm" "$name" \
      "$config" "$slot"; then
      "$ROOT/scripts/queue_llama32_frontier_wave7_eval.sh" "$gpu" "$arm" "$name" "$out"
    else
      rc=$?
      echo "[$(date -u +%FT%TZ)] quantization failed arm=${arm} gpu=${gpu} rc=${rc}; skipping evaluation"
      exit "$rc"
    fi
  ) &
  pipelines+=("$!")
done

failed=0
for pid in "${pipelines[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "[$(date -u +%FT%TZ)] Wave-7 orchestration finished with one or more failed arm pipelines"
  exit 1
fi
echo "[$(date -u +%FT%TZ)] Wave-7 orchestration complete"
