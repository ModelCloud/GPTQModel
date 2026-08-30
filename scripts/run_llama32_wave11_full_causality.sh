#!/usr/bin/env bash
set -euo pipefail

# Run the four end-to-end Wave-11 provenance controls. Each quantizer runs
# from a detached worktree at its recorded historical commit; evaluation is
# always performed from the pinned current evaluator commit. The frozen
# first-module canaries are intentionally not launched here: they require the
# exact H/Gram snapshot exporter described in the Wave-11 manifest.

ROOT="${QVQ_WAVE11_ROOT:-/root/QvQ-score-updates}"
MODEL="${QVQ_WAVE11_MODEL:-/monster/data/model/Llama-3.2-1B-Instruct}"
CAL="${QVQ_WAVE11_CALIBRATION:-/monster/data/model/dataset/nm-calibration/llm.parquet}"
YAQA="${QVQ_WAVE11_YAQA:-/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet}"
DISJOINTNESS="$ROOT/docs/experiments/disjointness-llama32-benchmark-v2.json"
CONFIG_REL="scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l8.json"
EVAL_COMMIT="e45e44f321d39baa55aee15e151aa54535b2530c"
RESULTS="${QVQ_WAVE11_RESULTS:-/root/qvq-results}"
WORKBASE="$(mktemp -d /tmp/qvq-wave11-full.XXXXXX)"
EVAL_WT="$WORKBASE/evaluator"

ids=(w11-old-full-a w11-old-full-b w11-new-full-a w11-new-full-b)
gpus=(0 1 2 3)
commits=(
  07517d85714a313ea0bf794c4d60783234c1a6ed
  07517d85714a313ea0bf794c4d60783234c1a6ed
  a1e0c50482715c57a7ae076906d78f81d1ec68f9
  a1e0c50482715c57a7ae076906d78f81d1ec68f9
)

cleanup() {
  for wt in "$WORKBASE"/*; do
    [ -d "$wt" ] || continue
    git -C "$ROOT" worktree remove --force "$wt" >/dev/null 2>&1 || true
  done
  rmdir "$WORKBASE" 2>/dev/null || true
}
trap cleanup EXIT

git -C "$ROOT" worktree add --detach "$EVAL_WT" "$EVAL_COMMIT" >/dev/null

run_one() {
  local arm="$1" gpu="$2" commit="$3"
  local wt="$WORKBASE/$arm"
  local out="$RESULTS/llama32-1b-${arm}-anchor_up4_l6_l8"
  local log="$RESULTS/${arm}.log"
  mkdir -p "$RESULTS"
  git -C "$ROOT" worktree add --detach "$wt" "$commit" >/dev/null
  {
    echo "[$(date -u +%FT%TZ)] start arm=$arm gpu=$gpu commit=$commit"
    env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$wt/scripts/qvq_quantize.py" \
        --model "$MODEL" --output "$out" --quant-config "$ROOT/$CONFIG_REL" \
        --calibration-dataset "$CAL" --calibration-row-start 0 --calibration-rows 128 \
        --yaqa-dataset "$YAQA" --yaqa-row-start 0 --yaqa-rows 182 --device cuda:0 \
        --disjointness-manifest "$DISJOINTNESS" --require-disjointness --qvq-telemetry
    env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$EVAL_WT/scripts/qvq_evaluate.py" tasks \
        --checkpoint "$out" --output "$out/post_quant_eval_result_wave11_gsm8k.json" \
        --task gsm8k_platinum_cot --batch-size 8 --device cuda:0 \
        --attn-implementation 'paged|sdpa'
    python "$ROOT/scripts/hash_qvq_checkpoint.py" "$out" \
      --output "$out/wave11_checkpoint_hashes.json"
    echo "[$(date -u +%FT%TZ)] complete arm=$arm"
  } >"$log" 2>&1
}

pids=()
for i in "${!ids[@]}"; do
  run_one "${ids[$i]}" "${gpus[$i]}" "${commits[$i]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
