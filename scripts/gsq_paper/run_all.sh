#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Run the GSQ paper reproduction across bits widths on one process per GPU.
#
# Each arm is a separate quantize pass. ``gptq`` arms calibrate on the 512
# sequence split; ``gsq`` arms initialize from GPTQ on that same split and then
# run the staged reconstruction training on the 4096 sequence split.
#
# Usage:
#   ./run_all.sh --corpus /models/datasets/gsq-paper --model meta-llama/Llama-3.2-1B-Instruct \
#                --out /models/gsq-paper-out --gpus 0,1,2,3,4,5
#
# Adding --seeds 7,11,23 repeats every arm per seed. A single run cannot
# separate a quantization effect from run-to-run noise; see docs/experiments/
# gsq-paper-reproduction.md for the measured spread.
set -euo pipefail

CORPUS=""; MODEL=""; OUT=""; GPUS="0"
BITS="2 3 4"; SEEDS="7"; EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus) CORPUS="$2"; shift 2 ;;
    --model)  MODEL="$2";  shift 2 ;;
    --out)    OUT="$2";    shift 2 ;;
    --gpus)   GPUS="$2";   shift 2 ;;
    --bits)   BITS="$2";   shift 2 ;;
    --seeds)  SEEDS="$2";  shift 2 ;;
    --extra)  EXTRA="$2";  shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for required in CORPUS MODEL OUT; do
  if [[ -z "${!required}" ]]; then
    echo "missing --${required,,}" >&2; exit 2
  fi
done

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
IFS=' ' read -r -a BIT_LIST <<< "$BITS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
mkdir -p "$OUT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMS=()
for bits in "${BIT_LIST[@]}"; do
  for variant in gptq gsq; do
    for seed in "${SEED_LIST[@]}"; do
      ARMS+=("w${bits}-${variant}-s${seed}|${bits}|${variant}|${seed}")
    done
  done
done

echo "queued ${#ARMS[@]} arms across ${#GPU_LIST[@]} GPUs"

run_arm() {
  local name=$1 bits=$2 variant=$3 seed=$4 gpu=$5
  local dir="$OUT/$name"
  mkdir -p "$dir"
  if [[ -f "$dir/results.json" ]]; then
    echo "[$name] already complete, skipping"
    return 0
  fi
  echo "[$name] gpu=$gpu bits=$bits variant=$variant seed=$seed"
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/run_arm.py" \
    --variant "$variant" --model "$MODEL" --corpus "$CORPUS" --out "$dir" \
    --bits "$bits" --seed "$seed" $EXTRA 2>&1 | tee "$dir/run.log"
}

slot=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r name bits variant seed <<< "$arm"
  gpu_index=$(( slot % ${#GPU_LIST[@]} ))
  run_arm "$name" "$bits" "$variant" "$seed" "${GPU_LIST[$gpu_index]}" &
  slot=$(( slot + 1 ))
  # One arm per GPU: wait for a slot to free up before queuing the next one.
  if (( slot % ${#GPU_LIST[@]} == 0 )); then
    wait
  fi
done
wait

echo
echo "=== completion ==="
for arm in "${ARMS[@]}"; do
  IFS='|' read -r name _ _ _ <<< "$arm"
  if [[ -f "$OUT/$name/results.json" ]]; then
    echo "ok   $name"
  else
    echo "FAIL $name"
  fi
done
