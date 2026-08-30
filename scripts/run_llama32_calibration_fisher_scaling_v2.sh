#!/usr/bin/env bash
set -euo pipefail

# Eight Fisher-scaling arms on four 96 GiB physical GPUs. Llama 3.2 1B uses
# sufficiently little VRAM that two quantize/evaluate pipelines per GPU are
# intentional. The two NM-full replicas are placed on different GPUs.

ROOT="${QVQ_FISHER_SCALING_ROOT:-/root/QvQ-score-updates}"
PINNED_COMMIT="${QVQ_FISHER_SCALING_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
MODEL="${QVQ_FISHER_SCALING_MODEL:-/monster/data/model/Llama-3.2-1B-Instruct}"
NM="${QVQ_FISHER_SCALING_NM:-/monster/data/model/dataset/nm-calibration/llm.parquet}"
YAQA="${QVQ_FISHER_SCALING_YAQA:-/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet}"
DATA_ROOT="${QVQ_FISHER_SCALING_DATA:-/root/qvq-data/calibration-fisher-scaling-v2}"
RESULTS="${QVQ_FISHER_SCALING_RESULTS:-/root/qvq-results/calibration-fisher-scaling-v2}"
WORKBASE="$(mktemp -d /tmp/qvq-fisher-scaling-v2.XXXXXX)"
WORKTREE="$WORKBASE/source"
DRIVER_MEMORY_ALLOWANCE_MIB="${QVQ_FISHER_SCALING_IDLE_MEMORY_MIB:-256}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

ids=(
  f1_yaqa182
  f2_yaqa182_nm128
  f3_yaqa182_nm256
  f4_yaqa182_nm512
  f5_yaqa182_nm1024
  f6_yaqa182_nm10000
  f7_yaqa182_nmrandom_token188256_seed20260830
  f8_yaqa182_nm10000_seed1
)
physical_gpus=(0 1 2 3 2 0 3 1)
slots=(0 0 0 0 1 1 1 1)
fisher_keys=(
  yaqa182
  yaqa182_nm128
  yaqa182_nm256
  yaqa182_nm512
  yaqa182_nm1024
  yaqa182_nm10000
  yaqa182_nmrandom_token188256_seed20260830
  yaqa182_nm10000
)
fisher_rows=(182 310 438 694 1206 10178 704 10178)
fisher_tokens=(302193 351918 397132 490449 677908 3961260 490481 3961260)
artifact_shas=(
  2140541facb66112428212b3a36d51a7735393b28c79db59c2429f6e51ed57ef
  eddd5bd1fbfbc45861cf1c57f1226bffd122864c29c8be04d26e85b7f7a118c0
  3fd0e9dfef53a9ce6b27418a54dd6d8535f04519a2014d6e5b0b523cc287adb8
  05288e83a8ce4cc7a607d7d19bfd48eeb0b45055ca7cc4f4b20d5f037266f2dc
  a0ffad0d2dde12a1da7da50475fe9c585114c98fad43ca9c167a6c9bde4b60de
  5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39
  762bce57198bb1be12eb1ec86a184a5c38c55b7c7d2ae12c9ba264a3764904e4
  5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39
)
configs=(
  llama32_1b_frontier_w5_anchor_up4_l6_l8.json
  llama32_1b_fisher_scaling_yaqa182_nm128.json
  llama32_1b_fisher_scaling_yaqa182_nm256.json
  llama32_1b_fisher_scaling_yaqa182_nm512.json
  llama32_1b_fisher_scaling_yaqa182_nm1024.json
  llama32_1b_fisher_scaling_yaqa182_nm10000.json
  llama32_1b_fisher_scaling_yaqa182_nmrandom_token188256_seed20260830.json
  llama32_1b_fisher_scaling_yaqa182_nm10000_seed1.json
)

gpu_uuid() {
  nvidia-smi --id="$1" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]'
}

assert_idle_gpu() {
  local gpu="$1" uuid memory utilization apps sample
  uuid="$(gpu_uuid "$gpu")"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if grep -q "^${uuid}," <<<"$apps"; then
    echo "refusing launch: foreign compute process present on physical GPU $gpu ($uuid)" >&2
    return 1
  fi
  for sample in 1 2 3; do
    IFS=, read -r utilization memory < <(
      nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
    )
    utilization="${utilization//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"
    if [ "$utilization" -ne 0 ] || [ "$memory" -gt "$DRIVER_MEMORY_ALLOWANCE_MIB" ]; then
      echo "refusing launch: GPU $gpu sample $sample util=${utilization}% memory=${memory}MiB" >&2
      return 1
    fi
    echo "idle-preflight gpu=$gpu uuid=$uuid sample=$sample/3 util=${utilization}% memory=${memory}MiB"
    sleep 1
  done
}

for gpu in 0 1 2 3; do
  assert_idle_gpu "$gpu"
done

mkdir -p "$RESULTS"
git -C "$ROOT" worktree add --detach "$WORKTREE" "$PINNED_COMMIT" >/dev/null

cleanup() {
  git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  rmdir "$WORKBASE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

BASE_DISJOINTNESS="$WORKTREE/docs/experiments/disjointness-calibration-union-v1.json"
D300="/root/qvq-data/divergence300-v1/divergence300-development.jsonl"

run_arm() {
  local index="$1" arm gpu slot key rows tokens expected_sha config yaqa disjointness out log gsm d300
  index="$1"
  arm="${ids[$index]}"
  gpu="${physical_gpus[$index]}"
  slot="${slots[$index]}"
  key="${fisher_keys[$index]}"
  rows="${fisher_rows[$index]}"
  tokens="${fisher_tokens[$index]}"
  expected_sha="${artifact_shas[$index]}"
  config="$WORKTREE/scripts/configs/${configs[$index]}"
  if [ "$key" = "yaqa182" ]; then
    yaqa="$YAQA"
    disjointness="$BASE_DISJOINTNESS"
  else
    yaqa="$DATA_ROOT/${key}.parquet"
    disjointness="$DATA_ROOT/${key}.disjointness.json"
  fi
  if [ "$(sha256sum "$yaqa" | awk '{print $1}')" != "$expected_sha" ]; then
    echo "refusing arm $arm: Fisher artifact SHA-256 mismatch" >&2
    return 2
  fi
  out="$RESULTS/llama32-1b-${arm}-anchor-up4-l6-l8"
  log="$RESULTS/${arm}.log"
  gsm="$out/post_quant_eval_gsm8k_platinum.json"
  d300="$out/post_quant_eval_divergence300.json"
  {
    echo "[$(date -u +%FT%TZ)] start arm=$arm physical_gpu=$gpu slot=$slot uuid=$(gpu_uuid "$gpu")"
    echo "commit=$PINNED_COMMIT fisher=$yaqa rows=$rows valid_fisher_tokens=$tokens sha256=$expected_sha"
    if [ ! -f "$out/qvq_quantize_run.json" ]; then
      env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
        python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_quantize.py -- \
          --model "$MODEL" --output "$out" --quant-config "$config" \
          --calibration-dataset "$NM" --calibration-dataset-split train \
          --calibration-row-start 0 --calibration-rows 128 \
          --yaqa-dataset "$yaqa" --yaqa-dataset-split train \
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
  echo "[$(date -u +%FT%TZ)] Fisher-scaling live status"
  printf '%-5s %-4s %-48s %-12s %-10s\n' GPU SLOT ARM TOKENS STATE
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
    if kill -0 "$pid" 2>/dev/null; then
      live=1
    fi
    printf '%-5s %-4s %-48s %-12s %-10s\n' \
      "${physical_gpus[$index]}" "${slots[$index]}" "${ids[$index]}" "${fisher_tokens[$index]}" "$state"
  done
  [ "$live" -eq 1 ] || break
  sleep 60
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
