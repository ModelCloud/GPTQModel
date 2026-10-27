#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Run the paper-model W2/W3 width database on two groups of four GPUs.
# Launch with setsid nohup; this script does not require a terminal or tmux.
set -euo pipefail

if [[ $# != 5 ]]; then
    echo "Usage: $0 MODEL CORPUS OUTPUT MIRROR ORIGINAL_ARMS" >&2
    echo "Set GSQ_PYTHON to the Python executable; ORIGINAL_ARMS contains w{2,3}gsq/ckpt and w{2,3}gptq/model." >&2
    exit 2
fi
model=$1
corpus=$2
output=$3
mirror=$4
original=$5
python=${GSQ_PYTHON:-python}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$mirror"
exec 9>"$output/driver.lock"
flock -n 9 || { echo "Another driver holds $output/driver.lock" >&2; exit 1; }

status() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" > "$output/status.tmp"
    mv "$output/status.tmp" "$output/status.txt"
}
mirror_checkpoints() {
    # Checkpoints are atomically published. Do not copy an incomplete model save.
    rsync -a --exclude='*.tmp' --exclude='driver.lock' --exclude='capture/' \
        --exclude='model/' "$output/" "$mirror/"
}
arm() {
    local bits=$1 visible=$2 result attempt
    mkdir -p "$output/w${bits}gsq"
    attempt=$(date -u +%Y%m%dT%H%M%SZ)
    if [[ -f "$output/w${bits}gsq/run.log" ]]; then
        mv "$output/w${bits}gsq/run.log" "$output/w${bits}gsq/run-$attempt.log"
    fi
    if [[ -f "$output/w${bits}gsq/exit-code.txt" ]]; then
        mv "$output/w${bits}gsq/exit-code.txt" "$output/w${bits}gsq/exit-code-$attempt.txt"
    fi
    result=0
    ROCR_VISIBLE_DEVICES="$visible" "$python" -m torch.distributed.run \
        --standalone --nproc-per-node=4 scripts/gsq_paper/run_arm.py \
        --distributed --variant gsq --model "$model" --corpus "$corpus" \
        --out "$output/w${bits}gsq" --bits "$bits" --layers 32 --group-size 128 \
        --train-seqs 4096 --gptq-seqs 512 --val-seqs 128 --epochs 20 --qk-steps 2000 \
        --batch-size 64 --microbatch 16 --seed 7 --capture gpu --target-cache device \
        --activation-checkpointing --reuse-capture --restore-best auto --resume \
        --resume-prefix-from "$original/w${bits}gsq/ckpt" --skip-eval \
        > "$output/w${bits}gsq/run.log" 2>&1 || result=$?
    printf '%s\n' "$result" > "$output/w${bits}gsq/exit-code.txt"
    return "$result"
}

status training
arm 2 0,1,2,3 &
w2=$!
arm 3 4,5,6,7 &
w3=$!
printf '%s\n' "$$" > "$output/driver.pid"
printf '%s %s\n' "$w2" "$w3" > "$output/arm-supervisors.pid"
(
    while kill -0 "$w2" 2>/dev/null || kill -0 "$w3" 2>/dev/null; do
        mirror_checkpoints || echo "Checkpoint mirror failed at $(date -u +%FT%TZ)" >&2
        sleep 30
    done
) &
mirror_pid=$!
finish() {
    local result=$?
    kill "$mirror_pid" 2>/dev/null || true
    if (( result != 0 )); then
        status "failed exit=$result; inspect arm/RCO logs"
        mirror_checkpoints || true
    fi
}
trap finish EXIT
w2_result=0
w3_result=0
wait "$w2" || w2_result=$?
wait "$w3" || w3_result=$?
kill "$mirror_pid" 2>/dev/null || true
wait "$mirror_pid" 2>/dev/null || true
if (( w2_result != 0 || w3_result != 0 )); then
    status "failed W2=$w2_result W3=$w3_result"
    mirror_checkpoints
    exit 1
fi
rsync -a --exclude='*.tmp' --exclude='driver.lock' --exclude='capture/' "$output/" "$mirror/"

status rco
all_devices=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
mkdir -p "$output/rco"
for spec in 2.375:2375 2.625:2625; do
    target=${spec%:*}
    tag=${spec#*:}
    "$python" scripts/gsq_paper/rco_allocate.py \
        --model "$model" --corpus "$corpus" --out "$output/rco/rco-$tag" \
        --target-bits "$target" --options 2,3 --arms "$output/w{bits}gsq/ckpt" \
        --layers 32 --group-size 128 --calib-seqs 64 --steps 200 --lr 0.1 \
        --device cuda:0 --batch-size 2 > "$output/rco/allocate-$tag.log" 2>&1
    "$python" scripts/gsq_paper/rco_materialize.py \
        --model "$model" --assignment "$output/rco/rco-$tag/assignment.json" \
        --arms "$output/w{bits}gsq/ckpt" --out "$output/rco/rco-$tag/model" \
        > "$output/rco/materialize-$tag.log" 2>&1
done
status nll
# This is a validation diagnostic: RCO search also uses the validation split.
# The paper's zero-shot benchmark remains a separate quality evaluation.
"$python" scripts/gsq_paper/compare_arms.py --corpus "$corpus" \
    --models "$output/rco/rco-2375/model" "$output/rco/rco-2625/model" \
             "$output/w2gsq/model" "$output/w3gsq/model" \
             "$original/w2gptq/model" "$original/w3gptq/model" \
    --labels RCO-2.375 RCO-2.625 W2-GSQ W3-GSQ W2-GPTQ W3-GPTQ \
    --dense "$model" --n 128 --devices "$all_devices" \
    --out "$output/rco/nll.json" --per-doc-out "$output/rco/nll-per-document.json" \
    > "$output/rco/nll.log" 2>&1
status complete
rsync -a --exclude='*.tmp' --exclude='driver.lock' --exclude='capture/' "$output/" "$mirror/"
