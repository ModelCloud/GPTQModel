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

compute_capability="$(
  nvidia-smi -i "${gpu_uuid}" --query-gpu=compute_cap --format=csv,noheader |
    awk 'NR == 1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print}'
)"
if [[ -z "${compute_capability}" ]]; then
  echo "Unable to resolve compute capability for ${gpu_uuid}." >&2
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_uuid}"
export TORCH_CUDA_ARCH_LIST="${compute_capability}"
export PYTHONHASHSEED=898
export TOKENIZERS_PARALLELISM=false

model="/monster/data/model/Qwen3-8B"
run_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
results_dir="${1:-${repo_root}/scripts/quantum_quantization/results/qwen3_8b_adjacent_ab/${run_timestamp}}"
checkpoint_root="$(mktemp -d /tmp/gptqmodel_qwen3_8b_adjacent_ab.XXXXXX)"
classic_checkpoint="${checkpoint_root}/classic_gptq"
adjacent_checkpoint="${checkpoint_root}/adjacent_hybrid"
script="${repo_root}/scripts/quantum_quantization/benchmark_llama32_adjacent_ab.py"

mkdir -p "${results_dir}"
cd "${repo_root}"

echo "GPU UUID: ${gpu_uuid}"
echo "Model: ${model}"
echo "Results: ${results_dir}"
echo "Checkpoints: ${checkpoint_root}"
echo "Evaluation convention: Qwen3 CI (chat template disabled, batch size 16)"

python "${script}" evaluate \
  --label dense_bf16 \
  --model "${model}" \
  --backend auto \
  --eval-batch-size 16 \
  --no-apply-chat-template \
  --result-json "${results_dir}/dense_eval.json"

python "${script}" quantize \
  --method classic \
  --model "${model}" \
  --save-dir "${classic_checkpoint}" \
  --result-json "${results_dir}/classic_quant.json"

python "${script}" quantize \
  --method adjacent \
  --model "${model}" \
  --save-dir "${adjacent_checkpoint}" \
  --result-json "${results_dir}/adjacent_quant.json"

python "${script}" evaluate \
  --label classic_gptq \
  --model "${classic_checkpoint}" \
  --backend marlin \
  --eval-batch-size 16 \
  --no-apply-chat-template \
  --result-json "${results_dir}/classic_eval.json"

python "${script}" evaluate \
  --label adjacent_hybrid \
  --model "${adjacent_checkpoint}" \
  --backend marlin \
  --eval-batch-size 16 \
  --no-apply-chat-template \
  --result-json "${results_dir}/adjacent_eval.json"

python "${script}" summarize \
  --model "${model}" \
  --dense-eval "${results_dir}/dense_eval.json" \
  --classic-quant "${results_dir}/classic_quant.json" \
  --adjacent-quant "${results_dir}/adjacent_quant.json" \
  --classic-eval "${results_dir}/classic_eval.json" \
  --adjacent-eval "${results_dir}/adjacent_eval.json" \
  --output "${results_dir}/summary.json"

echo "A/B complete: ${results_dir}/summary.json"
