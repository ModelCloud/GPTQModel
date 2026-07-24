#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cudaq_venv="${CUDAQ_VENV_PATH:-${repo_root}/venv/cudaq}"
gptq_python="${GPTQMODEL_PYTHON_BIN:-python}"
physical_gpu_index="${CUDAQ_PHYSICAL_GPU_INDEX:-6}"
problem_path="${GPTQ_ADJACENT_PROBLEM_PATH:-${cudaq_venv}/gptq_group32_problem.json}"

if [[ ! -x "${cudaq_venv}/bin/python" ]]; then
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

echo "GPTQ/CUDA-Q physical GPU ${physical_gpu_index}: ${gpu_uuid}" >&2
"${gptq_python}" "${script_dir}/export_gptq_group32.py" \
    --device cuda \
    --json-out "${problem_path}"

exec "${cudaq_venv}/bin/python" "${script_dir}/group32_cudaq_probe.py" \
    --problem-json "${problem_path}" \
    "$@"
