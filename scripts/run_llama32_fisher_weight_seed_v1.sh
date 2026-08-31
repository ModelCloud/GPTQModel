#!/usr/bin/env bash
set -euo pipefail

# Four exclusive-GPU NM4096 Fisher weight/seed arms. Each arm waits for the
# preceding composition arm on its assigned GPU, then quantizes and evaluates
# without overlapping another workload on that physical device.

ROOT="${QVQ_FISHER_WEIGHT_SEED_ROOT:-/root/QvQ-score-updates}"
PINNED_COMMIT="${QVQ_FISHER_WEIGHT_SEED_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
MODEL="${QVQ_FISHER_WEIGHT_SEED_MODEL:-/monster/data/model/Llama-3.2-1B-Instruct}"
NM="${QVQ_FISHER_WEIGHT_SEED_NM:-/monster/data/model/dataset/nm-calibration/llm.parquet}"
DATA_ROOT="${QVQ_FISHER_WEIGHT_SEED_DATA:-/root/qvq-data/calibration-fisher-composition-v1}"
RESULTS="${QVQ_FISHER_WEIGHT_SEED_RESULTS:-/root/qvq-results/calibration-fisher-weight-seed-v1}"
REFERENCE_RESULTS="${QVQ_FISHER_COMPOSITION_RESULTS:-/root/qvq-results/calibration-fisher-composition-v1}"
DRIVER_MEMORY_ALLOWANCE_MIB="${QVQ_FISHER_WEIGHT_SEED_IDLE_MEMORY_MIB:-256}"
WORKBASE=""
WORKTREE=""

export CUDA_DEVICE_ORDER=PCI_BUS_ID

ids=(
  f13_yaqa182_nm4096_yaqa15x
  f14_yaqa182_nm4096_yaqa3x
  f15_yaqa182_nm4096_yaqa2x_seed1
  f16_yaqa182_nm4096_yaqa2x_seed2
)
physical_gpus=(0 1 2 3)
yaqa_weights=(1.5 3 2 2)
yaqa_seeds=(0 0 1 2)
effective_weighted_tokens=(1945483.5 2398773 2096580 2096580)
configs=(
  llama32_1b_fisher_composition_yaqa182_nm4096_yaqa15x.json
  llama32_1b_fisher_composition_yaqa182_nm4096_yaqa3x.json
  llama32_1b_fisher_composition_yaqa182_nm4096_yaqa2x_seed1.json
  llama32_1b_fisher_composition_yaqa182_nm4096_yaqa2x_seed2.json
)
config_shas=(
  357f6ef2173e7ce2077e7b43e7384929819d671e03529210a8f0fba4f5bb2bff
  832918a227ccb69c3b7d37ef8db5d298831ee9dde1686b1e2d1f0af449076cb7
  f5799c4fff008991b2ebb9e3d101e8dd44b1b0b9a835ac7f71dbd57abc010db0
  881f100ad9cb45efef5ef3e1518eac5845220bf79ca08062faa5069f5682183d
)
ARTIFACT_SHA=98ce63d2eb3eb7542fae55ffc849b29b6fb0a9dd92d4687181e905f73fb26d53
UNIQUE_ROWS=4277
RAW_VALID_TOKENS=1794387

gpu_uuid() {
  nvidia-smi --id="$1" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]'
}

gpu_idle() {
  local gpu="$1" uuid apps utilization memory
  uuid="$(gpu_uuid "$gpu")"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if grep -q "^${uuid}," <<<"$apps"; then
    return 1
  fi
  IFS=, read -r utilization memory < <(
    nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
  )
  utilization="${utilization//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  [ "$utilization" -eq 0 ] && [ "$memory" -le "$DRIVER_MEMORY_ALLOWANCE_MIB" ]
}

wait_for_exclusive_gpu() {
  local gpu="$1" stable=0
  while [ "$stable" -lt 3 ]; do
    if gpu_idle "$gpu"; then
      stable=$((stable + 1))
      echo "[$(date -u +%FT%TZ)] GPU $gpu exclusive preflight sample=$stable/3 passed"
      sleep 1
    else
      stable=0
      echo "[$(date -u +%FT%TZ)] arm assigned to GPU $gpu is waiting for exclusive idle"
      nvidia-smi --id="$gpu" --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
      sleep 60
    fi
  done
}

cleanup() {
  if [ -n "$WORKTREE" ] && [ -d "$WORKTREE" ]; then
    git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  fi
  if [ -n "$WORKBASE" ] && [ -d "$WORKBASE" ]; then
    rmdir "$WORKBASE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

mkdir -p "$RESULTS"
WORKBASE="$(mktemp -d /tmp/qvq-fisher-weight-seed-v1.XXXXXX)"
WORKTREE="$WORKBASE/source"
git -C "$ROOT" worktree add --detach "$WORKTREE" "$PINNED_COMMIT" >/dev/null
D300="/root/qvq-data/divergence300-v1/divergence300-development.jsonl"
ARTIFACT="$DATA_ROOT/yaqa182_nm4096.parquet"
DISJOINTNESS="$DATA_ROOT/yaqa182_nm4096.disjointness.json"

run_arm() {
  local index="$1" arm gpu weight seed effective_tokens config expected_config_sha out log gsm d300
  arm="${ids[$index]}"
  gpu="${physical_gpus[$index]}"
  weight="${yaqa_weights[$index]}"
  seed="${yaqa_seeds[$index]}"
  effective_tokens="${effective_weighted_tokens[$index]}"
  config="$WORKTREE/scripts/configs/${configs[$index]}"
  expected_config_sha="${config_shas[$index]}"
  out="$RESULTS/llama32-1b-${arm}-anchor-up4-l6-l8"
  log="$RESULTS/${arm}.log"
  gsm="$out/post_quant_eval_gsm8k_platinum.json"
  d300="$out/post_quant_eval_divergence300.json"

  wait_for_exclusive_gpu "$gpu"
  if [ "$(sha256sum "$ARTIFACT" | awk '{print $1}')" != "$ARTIFACT_SHA" ]; then
    echo "refusing arm $arm: Fisher artifact SHA-256 mismatch" >&2
    return 2
  fi
  if [ "$(sha256sum "$config" | awk '{print $1}')" != "$expected_config_sha" ]; then
    echo "refusing arm $arm: quant config SHA-256 mismatch" >&2
    return 2
  fi

  {
    echo "[$(date -u +%FT%TZ)] start arm=$arm gpu=$gpu uuid=$(gpu_uuid "$gpu")"
    echo "commit=$PINNED_COMMIT corpus=$ARTIFACT unique_sequences=$UNIQUE_ROWS raw_valid_tokens=$RAW_VALID_TOKENS effective_weighted_tokens=$effective_tokens YAQA_weight=$weight NM_weight=1 YAQA_seed=$seed"
    if [ ! -f "$out/qvq_quantize_run.json" ]; then
      env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
        python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_quantize.py -- \
          --model "$MODEL" --output "$out" --quant-config "$config" \
          --calibration-dataset "$NM" --calibration-dataset-split train \
          --calibration-row-start 0 --calibration-rows 128 \
          --yaqa-dataset "$ARTIFACT" --yaqa-dataset-split train \
          --yaqa-row-start 0 --yaqa-rows "$UNIQUE_ROWS" \
          --batch-size 1 --concat-size 0 --calibration-sort desc --device cuda:0 \
          --disjointness-manifest "$DISJOINTNESS" --require-disjointness --qvq-telemetry
    fi
    if [ ! -f "$gsm" ]; then
      env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
        python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_evaluate.py -- tasks \
          --checkpoint "$out" --output "$gsm" --task gsm8k_platinum_cot \
          --batch-size 8 --device cuda:0 --attn-implementation 'paged|sdpa'
    fi
    if [ ! -f "$d300" ]; then
      env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
        python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_evaluate.py -- divergence300 \
          --dense-model "$MODEL" --checkpoint "$out" --dataset "$D300" \
          --device cuda:0 --output "$d300" --max-prompt-tokens 16384 \
          --dtype float16 --attn-implementation sdpa
    fi
    python "$WORKTREE/scripts/hash_qvq_checkpoint.py" "$out" --output "$out/checkpoint_hashes.json"
    echo "[$(date -u +%FT%TZ)] complete arm=$arm"
  } >"$log" 2>&1
}

pids=()
for index in "${!ids[@]}"; do
  run_arm "$index" &
  pids+=("$!")
done

while true; do
  live=0
  echo "[$(date -u +%FT%TZ)] Fisher weight/seed live status"
  printf '%-5s %-44s %-8s %-6s %-12s\n' GPU ARM WEIGHT SEED STATE
  for index in "${!ids[@]}"; do
    pid="${pids[$index]}"
    out="$RESULTS/llama32-1b-${ids[$index]}-anchor-up4-l6-l8"
    if [ -f "$out/post_quant_eval_divergence300.json" ]; then
      state=complete
    elif [ -f "$out/post_quant_eval_gsm8k_platinum.json" ]; then
      state=d300
    elif [ -f "$out/qvq_quantize_run.json" ]; then
      state=gsm8k
    elif kill -0 "$pid" 2>/dev/null; then
      state=waiting_or_quantizing
    else
      state=failed
    fi
    kill -0 "$pid" 2>/dev/null && live=1
    printf '%-5s %-44s %-8s %-6s %-12s\n' \
      "${physical_gpus[$index]}" "${ids[$index]}" "${yaqa_weights[$index]}x" \
      "${yaqa_seeds[$index]}" "$state"
  done
  [ "$live" -eq 1 ] || break
  sleep 60
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[ "$failed" -eq 0 ] || exit "$failed"

# Record all pairwise seed distances for the fixed NM4096/YAQA2x recipe.
wait_for_exclusive_gpu 0
reference="$REFERENCE_RESULTS/llama32-1b-f12_yaqa182_nm4096_yaqa2x-anchor-up4-l6-l8"
seed1="$RESULTS/llama32-1b-f15_yaqa182_nm4096_yaqa2x_seed1-anchor-up4-l6-l8"
seed2="$RESULTS/llama32-1b-f16_yaqa182_nm4096_yaqa2x_seed2-anchor-up4-l6-l8"
for spec in "f12_vs_f15:$reference:$seed1" "f12_vs_f16:$reference:$seed2" "f15_vs_f16:$seed1:$seed2"; do
  IFS=: read -r label left right <<<"$spec"
  env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" \
      --script scripts/compare_qvq_seed_checkpoints.py -- \
      "$left" "$right" --device cuda:0 --output "$RESULTS/${label}_checkpoint_distance.json"
done
