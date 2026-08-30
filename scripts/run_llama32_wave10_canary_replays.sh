#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/QvQ-score-updates"
RESULTS="/root/qvq-results"

checkpoints=(
  "/root/qvq-results/llama32-1b-w2-w5queued_8980aa_llama32_1b_frontier_anchor_up4_l6_l8"
  "/root/qvq-results/llama32-1b-w2-w5queued_8980aa_llama32_1b_frontier_anchor_up4_l6_l8"
  "/root/qvq-results/llama32-1b-w2-w5queued_8980aa_llama32_1b_frontier_anchor_up4_l6_l8"
  "/root/qvq-results/llama32-1b-w2-w5queued_8980aa_llama32_1b_frontier_anchor_up4_l6_l8"
  "/root/qvq-results/llama32-1b-w2-w8queued_fdbd68_llama32_1b_frontier_anchor_up4_l6_l8"
  "/root/qvq-results/llama32-1b-w2-w8queued_fdbd68_llama32_1b_frontier_anchor_up4_l6_l8"
  "/root/qvq-results/llama32-1b-w2-w8queued_fdbd68_llama32_1b_frontier_anchor_up4_l6_l8"
  "/root/qvq-results/llama32-1b-w2-w8queued_fdbd68_llama32_1b_frontier_anchor_up4_l6_l8"
)
ids=(w10-8980aa-r1 w10-8980aa-r2 w10-8980aa-r3 w10-8980aa-r4 w10-fdbd68-r1 w10-fdbd68-r2 w10-fdbd68-r3 w10-fdbd68-r4)
gpus=(0 1 2 3 4 5 6 7)

run_one() {
  local idx="$1" checkpoint="$2" arm="$3" gpu="$4"
  local out="$RESULTS/${arm}-gsm8k-platinum.json"
  local log="$RESULTS/${arm}.log"
  {
    echo "[$(date -u +%FT%TZ)] start arm=${arm} gpu=${gpu} checkpoint=${checkpoint}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      python "$ROOT/scripts/qvq_evaluate.py" tasks \
        --checkpoint "$checkpoint" --output "$out" \
        --task gsm8k_platinum_cot --batch-size 8 --device cuda:0 \
        --attn-implementation 'paged|sdpa'
    python - "$out" "$RESULTS/${arm}-digest.json" <<'PY'
import hashlib, json, sys
source, target = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
task = payload.get("tasks", {}).get("gsm8k_platinum_cot", {})
metrics = task.get("metrics", {})
canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
with open(target, "w", encoding="utf-8") as handle:
    json.dump({"metrics": metrics, "metrics_sha256": hashlib.sha256(canonical.encode()).hexdigest()}, handle, indent=2, sort_keys=True)
PY
    echo "[$(date -u +%FT%TZ)] complete arm=${arm}"
  } >"$log" 2>&1
}

pids=()
for i in "${!ids[@]}"; do
  run_one "$i" "${checkpoints[$i]}" "${ids[$i]}" "${gpus[$i]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "[$(date -u +%FT%TZ)] Wave-10 canary replay failed"
  exit 1
fi
echo "[$(date -u +%FT%TZ)] Wave-10 canary replay complete"
