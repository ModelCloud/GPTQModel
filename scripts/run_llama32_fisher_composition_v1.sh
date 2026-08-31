#!/usr/bin/env bash
set -euo pipefail

# Four exclusive-GPU Fisher-composition arms. Each arm waits independently for
# its assigned physical GPU, preserving quant/eval overlap across idle GPUs
# while still enforcing exactly one process per GPU.

ROOT="${QVQ_FISHER_COMPOSITION_ROOT:-/root/QvQ-score-updates}"
PINNED_COMMIT="${QVQ_FISHER_COMPOSITION_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
MODEL="${QVQ_FISHER_COMPOSITION_MODEL:-/monster/data/model/Llama-3.2-1B-Instruct}"
NM="${QVQ_FISHER_COMPOSITION_NM:-/monster/data/model/dataset/nm-calibration/llm.parquet}"
DATA_ROOT="${QVQ_FISHER_COMPOSITION_DATA:-/root/qvq-data/calibration-fisher-composition-v1}"
RESULTS="${QVQ_FISHER_COMPOSITION_RESULTS:-/root/qvq-results/calibration-fisher-composition-v1}"
DRIVER_MEMORY_ALLOWANCE_MIB="${QVQ_FISHER_COMPOSITION_IDLE_MEMORY_MIB:-256}"
WORKBASE=""
WORKTREE=""

export CUDA_DEVICE_ORDER=PCI_BUS_ID

ids=(
  f9_yaqa182_nm2048_yaqa1x
  f10_yaqa182_nm2048_yaqa2x
  f11_yaqa182_nm4096_yaqa1x
  f12_yaqa182_nm4096_yaqa2x
)
physical_gpus=(0 1 2 3)
fisher_keys=(yaqa182_nm2048 yaqa182_nm2048 yaqa182_nm4096 yaqa182_nm4096)
unique_rows=(2229 2229 4277 4277)
raw_valid_tokens=(1051958 1051958 1794387 1794387)
effective_weighted_tokens=(1051958 1354151 1794387 2096580)
yaqa_weights=(1 2 1 2)
artifact_shas=(
  7a960955ffb07c3f7fa33994177524af615306953f70290567ed44c5b7be0041
  7a960955ffb07c3f7fa33994177524af615306953f70290567ed44c5b7be0041
  98ce63d2eb3eb7542fae55ffc849b29b6fb0a9dd92d4687181e905f73fb26d53
  98ce63d2eb3eb7542fae55ffc849b29b6fb0a9dd92d4687181e905f73fb26d53
)
config_shas=(
  5ceee046ecab71ca8252f82d0e12c49a4a7afbcb2bf845ce6eea6231f1d6a488
  eed67bdd7da1a0b59de897bc2a3ac0e48b6914a9c9f498b02a3804f2525e5004
  f449e7718676db7bc625e804c39eb6a2b8d61e186cf26f53f011fe229fe40313
  f84503dc5a0011a25100981f00b2a2a72b5f13d3da0f58d98890130dbdc5d09d
)
configs=(
  llama32_1b_fisher_scaling_yaqa182_nm2048.json
  llama32_1b_fisher_composition_yaqa182_nm2048_yaqa2x.json
  llama32_1b_fisher_scaling_yaqa182_nm4096.json
  llama32_1b_fisher_composition_yaqa182_nm4096_yaqa2x.json
)

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
WORKBASE="$(mktemp -d /tmp/qvq-fisher-composition-v1.XXXXXX)"
WORKTREE="$WORKBASE/source"
git -C "$ROOT" worktree add --detach "$WORKTREE" "$PINNED_COMMIT" >/dev/null
D300="/root/qvq-data/divergence300-v1/divergence300-development.jsonl"

run_arm() {
  local index="$1" arm gpu key rows raw_tokens effective_tokens yaqa_weight
  local artifact expected_artifact_sha config expected_config_sha disjointness out log gsm d300
  arm="${ids[$index]}"
  gpu="${physical_gpus[$index]}"
  key="${fisher_keys[$index]}"
  rows="${unique_rows[$index]}"
  raw_tokens="${raw_valid_tokens[$index]}"
  effective_tokens="${effective_weighted_tokens[$index]}"
  yaqa_weight="${yaqa_weights[$index]}"
  artifact="$DATA_ROOT/${key}.parquet"
  expected_artifact_sha="${artifact_shas[$index]}"
  config="$WORKTREE/scripts/configs/${configs[$index]}"
  expected_config_sha="${config_shas[$index]}"
  disjointness="$DATA_ROOT/${key}.disjointness.json"
  out="$RESULTS/llama32-1b-${arm}-anchor-up4-l6-l8"
  log="$RESULTS/${arm}.log"
  gsm="$out/post_quant_eval_gsm8k_platinum.json"
  d300="$out/post_quant_eval_divergence300.json"

  wait_for_exclusive_gpu "$gpu"

  if [ "$(sha256sum "$artifact" | awk '{print $1}')" != "$expected_artifact_sha" ]; then
    echo "refusing arm $arm: Fisher artifact SHA-256 mismatch" >&2
    return 2
  fi
  if [ "$(sha256sum "$config" | awk '{print $1}')" != "$expected_config_sha" ]; then
    echo "refusing arm $arm: quant config SHA-256 mismatch" >&2
    return 2
  fi

  {
    echo "[$(date -u +%FT%TZ)] start arm=$arm gpu=$gpu uuid=$(gpu_uuid "$gpu")"
    echo "commit=$PINNED_COMMIT corpus=$artifact unique_sequences=$rows raw_valid_tokens=$raw_tokens effective_weighted_tokens=$effective_tokens YAQA_weight=$yaqa_weight NM_weight=1"
    if [ ! -f "$out/qvq_quantize_run.json" ]; then
      env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
        python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_quantize.py -- \
          --model "$MODEL" --output "$out" --quant-config "$config" \
          --calibration-dataset "$NM" --calibration-dataset-split train \
          --calibration-row-start 0 --calibration-rows 128 \
          --yaqa-dataset "$artifact" --yaqa-dataset-split train \
          --yaqa-row-start 0 --yaqa-rows "$rows" \
          --batch-size 1 --concat-size 0 --calibration-sort desc --device cuda:0 \
          --disjointness-manifest "$disjointness" --require-disjointness --qvq-telemetry
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
  echo "[$(date -u +%FT%TZ)] Fisher-composition live status"
  printf '%-5s %-42s %-12s %-12s\n' GPU ARM WEIGHT STATE
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
      state=quantizing
    else
      state=failed
    fi
    kill -0 "$pid" 2>/dev/null && live=1
    printf '%-5s %-42s %-12s %-12s\n' \
      "${physical_gpus[$index]}" "${ids[$index]}" "${yaqa_weights[$index]}x" "$state"
  done
  [ "$live" -eq 1 ] || break
  sleep 60
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
