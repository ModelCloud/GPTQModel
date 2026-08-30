#!/usr/bin/env bash
set -euo pipefail

# Eight-arm lifecycle x YAQA calibration ablation. The source revision is
# frozen once, installed through run_in_worktree.py, and shared read-only by
# all workers. Each physical GPU owns exactly one end-to-end arm and begins
# GSM8K/D300 evaluation immediately after its checkpoint is published.

ROOT="${QVQ_CAL_ABLATION_ROOT:-/root/QvQ-score-updates}"
PINNED_COMMIT="${QVQ_CAL_ABLATION_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
MODEL="${QVQ_CAL_ABLATION_MODEL:-/monster/data/model/Llama-3.2-1B-Instruct}"
NM="${QVQ_CAL_ABLATION_NM:-/monster/data/model/dataset/nm-calibration/llm.parquet}"
YAQA="${QVQ_CAL_ABLATION_YAQA:-/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet}"
UNION="${QVQ_CAL_ABLATION_UNION:-/root/qvq-data/calibration_union_v1/calibration_union_v1.parquet}"
RESULTS="${QVQ_CAL_ABLATION_RESULTS:-/root/qvq-results/calibration-ablation-v1}"
WORKBASE="$(mktemp -d /tmp/qvq-calibration-ablation-v1.XXXXXX)"
WORKTREE="$WORKBASE/source"

EXPECTED_UNION_SHA256="05288e83a8ce4cc7a607d7d19bfd48eeb0b45055ca7cc4f4b20d5f037266f2dc"
ACTUAL_UNION_SHA256="$(sha256sum "$UNION" | awk '{print $1}')"
if [ "$ACTUAL_UNION_SHA256" != "$EXPECTED_UNION_SHA256" ]; then
  echo "refusing calibration ablation: union SHA-256 mismatch" >&2
  exit 2
fi

mkdir -p "$RESULTS"
git -C "$ROOT" worktree add --detach "$WORKTREE" "$PINNED_COMMIT" >/dev/null

cleanup() {
  git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  rmdir "$WORKBASE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

BASE_CONFIG="scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l8.json"
UNION_CONFIG="scripts/configs/llama32_1b_calibration_ablation_anchor_up4_l6_l8_union694.json"
DISJOINTNESS="docs/experiments/disjointness-calibration-union-v1.json"
D300="/root/qvq-data/divergence300-v1/divergence300-development.jsonl"

ids=(
  c1_nm128__yaqa182
  c2_nm512__yaqa182
  c3_yaqa182__yaqa182
  c4_union694__yaqa182
  c5_nm128__union694
  c6_nm512__union694
  c7_yaqa182__union694
  c8_union694__union694
)
calibration_paths=("$NM" "$NM" "$YAQA" "$UNION" "$NM" "$NM" "$YAQA" "$UNION")
calibration_rows=(128 512 182 694 128 512 182 694)
yaqa_paths=("$YAQA" "$YAQA" "$YAQA" "$YAQA" "$UNION" "$UNION" "$UNION" "$UNION")
yaqa_rows=(182 182 182 182 694 694 694 694)
configs=(
  "$BASE_CONFIG" "$BASE_CONFIG" "$BASE_CONFIG" "$BASE_CONFIG"
  "$UNION_CONFIG" "$UNION_CONFIG" "$UNION_CONFIG" "$UNION_CONFIG"
)

run_arm() {
  local gpu="$1" arm="$2" calibration="$3" calibration_count="$4"
  local yaqa="$5" yaqa_count="$6" config="$7"
  local out="$RESULTS/llama32-1b-${arm}-anchor-up4-l6-l8"
  local log="$RESULTS/${arm}.log"
  local gsm="$out/post_quant_eval_gsm8k_platinum.json"
  local d300="$out/post_quant_eval_divergence300.json"

  {
    echo "[$(date -u +%FT%TZ)] start arm=$arm gpu=$gpu commit=$PINNED_COMMIT"
    echo "lifecycle=$calibration rows=$calibration_count yaqa=$yaqa rows=$yaqa_count"
    if [ ! -f "$out/qvq_quantize_run.json" ]; then
      env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
        python "$ROOT/scripts/run_in_worktree.py" --worktree "$WORKTREE" --script scripts/qvq_quantize.py -- \
          --model "$MODEL" --output "$out" --quant-config "$WORKTREE/$config" \
          --calibration-dataset "$calibration" --calibration-dataset-split train \
          --calibration-row-start 0 --calibration-rows "$calibration_count" \
          --yaqa-dataset "$yaqa" --yaqa-dataset-split train \
          --yaqa-row-start 0 --yaqa-rows "$yaqa_count" \
          --batch-size 1 --concat-size 0 --calibration-sort desc --device cuda:0 \
          --disjointness-manifest "$WORKTREE/$DISJOINTNESS" --require-disjointness --qvq-telemetry
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
    python "$WORKTREE/scripts/hash_qvq_checkpoint.py" "$out" \
      --output "$out/checkpoint_hashes.json"
    echo "[$(date -u +%FT%TZ)] complete arm=$arm"
  } >"$log" 2>&1
}

pids=()
for index in "${!ids[@]}"; do
  run_arm \
    "$index" "${ids[$index]}" \
    "${calibration_paths[$index]}" "${calibration_rows[$index]}" \
    "${yaqa_paths[$index]}" "${yaqa_rows[$index]}" \
    "${configs[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
