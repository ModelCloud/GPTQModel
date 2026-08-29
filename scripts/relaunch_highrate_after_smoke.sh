#!/usr/bin/env bash
set -euo pipefail
root=/root/QvQ
results=/root/qvq-results
smoke=highrate-rerun-llama32_1b_highrate_up4_l8_15
smoke_ckpt="$results/$smoke"
smoke_report="$results/$smoke-gsm8k-platinum-v2.json"
while [[ ! -f "$smoke_ckpt/model.safetensors.index.json" ]]; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 python "$root/scripts/qvq_evaluate.py" tasks --checkpoint "$smoke_ckpt" --output "$smoke_report" --task gsm8k_platinum_cot --batch-size 8 --device cuda:0 --attn-implementation 'paged|sdpa'
score=$(python - "$smoke_report" <<'PY'
import json,sys
try:
 d=json.load(open(sys.argv[1])); print(float(d["tasks"]["gsm8k_platinum_cot"]["metrics"].get("acc,num",0)))
except Exception: print(0.0)
PY
)
python - "$score" <<'PY'
import sys
assert float(sys.argv[1]) > 0, f"non-positive smoke score: {sys.argv[1]}"
PY
configs=(llama32_1b_highrate_up5_l12_15 llama32_1b_highrate_up7_l14_15 llama32_1b_highrate_up4_l12_13_up6_l14_15 llama32_1b_highrate_flat35_up4_l12_15 llama32_1b_highrate_flat35_up45_l14_15 llama32_1b_highrate_flat35_up55_l15 llama32_1b_highrate_flat35_up4_l14_up5_l15)
gpus=(1 2 3 4 5 6 7)
for i in "${!configs[@]}"; do
 cfg=${configs[$i]}; gpu=${gpus[$i]}; out="$results/highrate-rerun-$cfg"; log="$results/highrate-logs/rerun-$cfg.log"
 CUDA_VISIBLE_DEVICES=$gpu nohup python "$root/scripts/qvq_quantize.py" --model /monster/data/model/Llama-3.2-1B-Instruct --output "$out" --quant-config "$root/scripts/configs/$cfg.json" --calibration-dataset /monster/data/model/dataset/nm-calibration/llm.parquet --calibration-row-start 0 --calibration-rows 128 --yaqa-dataset "$root/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet" --yaqa-row-start 0 --yaqa-rows 182 --batch-size 1 --concat-size 0 --calibration-sort desc --device cuda:0 --no-qvq-telemetry >"$log" 2>&1 &
done
