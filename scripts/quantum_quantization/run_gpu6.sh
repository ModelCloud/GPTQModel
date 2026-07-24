#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
venv_path="${CUDAQ_VENV_PATH:-${repo_root}/venv/cudaq}"
physical_gpu_index="${CUDAQ_PHYSICAL_GPU_INDEX:-6}"
probe="${script_dir}/gptq_qubo_probe.py"

if [[ "${1:-}" == "--capacity" ]]; then
    probe="${script_dir}/statevector_capacity_probe.py"
    shift
elif [[ "${1:-}" == "--group32" ]]; then
    probe="${script_dir}/group32_cudaq_probe.py"
    shift
fi

if [[ "${1:-}" == "--" ]]; then
    shift
fi

if [[ ! -x "${venv_path}/bin/python" ]]; then
    echo "Missing CUDA-Q environment. Run scripts/quantum_quantization/setup_cudaq.sh first." >&2
    exit 1
fi

gpu_uuid="$(
    nvidia-smi -i "${physical_gpu_index}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]'
)"
if [[ -z "${gpu_uuid}" ]]; then
    echo "Could not resolve physical GPU index ${physical_gpu_index}." >&2
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_uuid}"
export CUDAQ_MAX_GPU_MEMORY_GB="${CUDAQ_MAX_GPU_MEMORY_GB:-94}"
export CUDAQ_MAX_CPU_MEMORY_GB="${CUDAQ_MAX_CPU_MEMORY_GB:-0}"

echo "CUDA-Q physical GPU ${physical_gpu_index}: ${gpu_uuid}" >&2
exec "${venv_path}/bin/python" "${probe}" "$@"
