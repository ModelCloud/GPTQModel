#!/usr/bin/env bash
set -euo pipefail

ROOT="${QVQ_TOP10_ROOT:-/root/QvQ-score-updates}"
RESULTS="${QVQ_TOP10_RESULTS:-/root/qvq-results/top10-cb-fa2-replay-v1}"
mkdir -p "$RESULTS"

names=(
  w2_flat35_seed1
  w3_flat35_up4_l12_15_seed1
  w2_flat35_up4_v4_l9_15
  w2_flat35_up4_l6_15
  w2_flat35_up4_l4_15
  w4_anchor_up4_l6_9
  w2_flat35_up4_o4_all
  w9_anchor_up4_l8_l11_l12
  w4_anchor_up4_l6
  w2_flat35_up4_l8_15
)
arm_ids=(9769b1 4e7424 049d0c e55f4d 428e4d 5a0dce f7f157 370c3a 7b6e2a 42c2fc)
checkpoints=(
  /root/qvq-results/llama32-1b-w2-w2fixed_9769b1_llama32_1b_frontier_w2_flat35_seed1
  /root/qvq-results/llama32-1b-w2-w3queued_4e7424_llama32_1b_frontier_w3_flat35_up4_l12_15_seed1
  /root/qvq-results/llama32-1b-w2-w2fixed_049d0c_llama32_1b_frontier_w2_flat35_up4_v4_l9_15
  /root/qvq-results/llama32-1b-w2-w2fixed_e55f4d_llama32_1b_frontier_w2_flat35_up4_l6_15
  /root/qvq-results/llama32-1b-w2-w2fixed_428e4d_llama32_1b_frontier_w2_flat35_up4_l4_15
  /root/qvq-results/llama32-1b-w2-w4queued_5a0dce_llama32_1b_frontier_w4_anchor_up4_l6_9
  /root/qvq-results/llama32-1b-w2-w2fixed_f7f157_llama32_1b_frontier_w2_flat35_up4_o4_all
  /root/qvq-results/llama32-1b-w2-w9queued_370c3a_llama32_1b_frontier_anchor_up4_l8_l11_l12
  /root/qvq-results/llama32-1b-w2-w4queued_7b6e2a_llama32_1b_frontier_w4_anchor_up4_l6
  /root/qvq-results/llama32-1b-w2-w2fixed_42c2fc_llama32_1b_frontier_w2_flat35_up4_l8_15
)
gpus=(0 1 2 3)

gpu_idle() {
  local gpu="$1" uuid apps utilization memory
  uuid="$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  grep -q "^${uuid}," <<<"$apps" && return 1
  IFS=, read -r utilization memory < <(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
  utilization="${utilization//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  [ "$utilization" -eq 0 ] && [ "$memory" -le 256 ]
}

wait_idle() {
  local gpu="$1" stable=0
  while [ "$stable" -lt 3 ]; do
    if gpu_idle "$gpu"; then stable=$((stable + 1)); sleep 1; else stable=0; echo "[$(date -u +%FT%TZ)] gpu=$gpu waiting for idle" >&2; sleep 60; fi
  done
}

run_arm() {
  local index="$1" gpu="$2" name="${names[$index]}" arm="${arm_ids[$index]}" checkpoint="${checkpoints[$index]}"
  local out="$RESULTS/$name" baseline="$out/post_quant_eval_gsm8k_platinum_fa2_graph_off_v4.json" fast="$out/post_quant_eval_gsm8k_platinum_fa2_decode_graph_v4.json" summary="$out/post_quant_eval_gsm8k_platinum_fa2_decode_graph_v4_comparison.json"
  mkdir -p "$out"
  [ -f "$checkpoint/model.safetensors.index.json" ] || { echo "missing checkpoint $checkpoint" >&2; return 1; }
  wait_idle "$gpu"
  if [ ! -f "$baseline" ]; then
    env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/qvq_evaluate.py" tasks \
      --checkpoint "$checkpoint" --output "$baseline" --task gsm8k_platinum_cot --batch-size 64 --device cuda:0 \
      --attn-implementation 'paged|flash_attention_2' --cuda-graph-mode off --no-use-async-batching \
      --max-blocks-per-request 32 --max-batch-tokens 8192
  fi
  if [ ! -f "$fast" ]; then
    env PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/qvq_evaluate.py" tasks \
      --checkpoint "$checkpoint" --output "$fast" --task gsm8k_platinum_cot --batch-size 64 --device cuda:0 \
      --attn-implementation 'paged|flash_attention_2' --cuda-graph-mode decode --no-use-async-batching \
      --max-blocks-per-request 32 --max-batch-tokens 8192
  fi
  python - "$baseline" "$fast" "$summary" "$name" "$arm" "$gpu" "$checkpoint" <<'PY'
import json, sys
from pathlib import Path
bpath, fpath, spath, name, arm, gpu, checkpoint = map(Path, sys.argv[1:])
b = json.loads(bpath.read_text()); f = json.loads(fpath.read_text())
bt = b['tasks']['gsm8k_platinum_cot']; ft = f['tasks']['gsm8k_platinum_cot']
be = bt.get('engine', {}); fe = ft.get('engine', {}); bx = be.get('execution', {}); fx = fe.get('execution', {})
bc = bx.get('continuous_batching_config', {}); fc = fx.get('continuous_batching_config', {})
bg = bc.get('cuda_graph_booleans', bc.get('use_cuda_graph')); fg = fc.get('cuda_graph_booleans', fc.get('use_cuda_graph'))
def accuracy(t): return float(t['metrics']['acc,num'])
def correct(t): return round(accuracy(t) * 1209)
payload = {
  'arm_name': str(name), 'arm_id': str(arm), 'gpu': int(gpu), 'checkpoint': str(checkpoint), 'task': 'gsm8k_platinum_cot', 'rows': 1209,
  'baseline_result': str(bpath), 'decode_graph_result': str(fpath), 'graph_off_correct': correct(bt), 'graph_off_accuracy': accuracy(bt),
  'decode_graph_correct': correct(ft), 'decode_graph_accuracy': accuracy(ft), 'graph_off_seconds': bt['seconds'], 'decode_graph_seconds': ft['seconds'],
  'speedup': bt['seconds'] / ft['seconds'], 'metric_parity': accuracy(bt) == accuracy(ft),
  'generation_backend': [bx.get('generation_backend'), fx.get('generation_backend')], 'paged_attention': [bx.get('paged_attention'), fx.get('paged_attention')],
  'graph_booleans': [bg, fg], 'continuous_batching_config': [bc, fc],
  'continuous_batching_verified': bx.get('generation_backend') == 'continuous_batching' and fx.get('generation_backend') == 'continuous_batching',
  'paged_attention_verified': bx.get('paged_attention') is True and fx.get('paged_attention') is True,
  'graph_policy_verified': bg == [False, False] and fg == [False, True],
  'cache_policy_compatible': bc.get('block_size') == fc.get('block_size') and bc.get('max_batch_tokens') == fc.get('max_batch_tokens') and bc.get('max_blocks_per_request') == fc.get('max_blocks_per_request') and bc.get('allow_block_sharing') == fc.get('allow_block_sharing') and bc.get('use_async_batching') is False and fc.get('use_async_batching') is False,
}
spath.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

worker() {
  local gpu="$1"; shift
  local index
  for index in "$@"; do run_arm "$index" "$gpu" >"$RESULTS/${names[$index]}_gpu${gpu}.log" 2>&1 || return 1; done
}

echo "[$(date -u +%FT%TZ)] top-10 CB/FA2 replay commit=$(git -C "$ROOT" rev-parse HEAD)"
pids=()
for gpu in "${gpus[@]}"; do
  indices=()
  for index in "${!names[@]}"; do [ $((index % ${#gpus[@]})) -eq "$gpu" ] && indices+=("$index"); done
  worker "$gpu" "${indices[@]}" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
