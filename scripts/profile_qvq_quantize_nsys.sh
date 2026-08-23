#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# nsys end-to-end capture of the real QVQ quantization harness.
#   scripts/profile_qvq_quantize_nsys.sh <run-name> [profile_qvq_quantize_nsys.py / qvq_quantize.py args...]
# Example (full Llama-3.2-1B, 128 calibration rows):
#   scripts/profile_qvq_quantize_nsys.sh llama32_1b_full \
#     --model /monster/data/model/Llama-3.2-1B-Instruct --output /tmp/qvq_prof/llama32_1b \
#     --calibration-dataset /monster/data/model/dataset/nm-calibration/llm.parquet --calibration-rows 128 ...
# Quick run: add --max-layers 2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:?run name}"; shift
VENV="${QVQ_PROFILE_VENV:-$REPO_ROOT/.venv-qvq-profile}"
NSYS="${NSYS:-/usr/local/bin/nsys}"
ART_DIR="${QVQ_NSYS_ARTIFACT_DIR:-$REPO_ROOT/artifacts/nsys}"
REP_DIR="${QVQ_NSYS_REP_DIR:-$ART_DIR/reps}"   # .nsys-rep / .sqlite are gitignored
mkdir -p "$ART_DIR" "$REP_DIR"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$VENV/cuda-shim-include${CPATH:+:$CPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

REP="$REP_DIR/$RUN_NAME"
"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=false \
  --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output="$REP" \
  "$VENV/bin/python" "$REPO_ROOT/scripts/profile_qvq_quantize_nsys.py" \
    --attribution-json "$ART_DIR/${RUN_NAME}_host_attribution.json" "$@"

# Stats used by docs/qvq_nsys_profile_llama32_1b.md
for report in cuda_gpu_kern_sum cuda_gpu_sum cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum nvtx_sum nvtx_pushpop_sum nvtx_gpu_proj_sum nvtx_kern_sum cuda_kern_exec_sum osrt_sum; do
  "$NSYS" stats --report "$report" --format csv --force-export=true \
    --output "$ART_DIR/${RUN_NAME}" "$REP.nsys-rep" >/dev/null 2>"$ART_DIR/${RUN_NAME}_${report}.err" \
    || echo "nsys stats --report $report failed (see ${RUN_NAME}_${report}.err)"
done
ls -la "$ART_DIR" | grep "$RUN_NAME"
echo "report: $REP.nsys-rep"
