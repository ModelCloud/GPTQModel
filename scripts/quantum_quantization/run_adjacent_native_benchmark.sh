#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
physical_gpu_index=6
gpu_uuid="$(
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    awk -F', ' -v target_index="${physical_gpu_index}" '$1 == target_index {print $2}'
)"

if [[ -z "${gpu_uuid}" ]]; then
  echo "Unable to resolve physical GPU index ${physical_gpu_index} to a UUID." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_uuid}"
cd "${repo_root}"
python scripts/quantum_quantization/benchmark_adjacent_native.py \
  --json-out scripts/quantum_quantization/results/adjacent_native_benchmark.json \
  --split-depth 8 \
  --max-nodes-per-worker 50000 \
  --timing-repeats 5 \
  "$@"
