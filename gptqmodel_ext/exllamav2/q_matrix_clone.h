// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <optional>

// Bind finalized native words without running the destructive shuffle again.
// QMatrix contains borrowed pointers only; its destructor does not own tensors.
int64_t clone_q_matrix(
    int64_t handle, torch::Tensor weight, torch::Tensor zeros, torch::Tensor scales,
    std::optional<torch::Tensor> perm, std::optional<torch::Tensor> invperm)
{
    TORCH_CHECK(handle != 0, "ExLlamaV2 requires an initialized matrix descriptor");
    QMatrix* source = reinterpret_cast<QMatrix*>(static_cast<uintptr_t>(handle));
    TORCH_CHECK(source->is_gptq, "Only finalized GPTQ/AWQ matrices can be combined");
    TORCH_CHECK(weight.dim() == 2 && zeros.dim() == 2 && scales.dim() == 2, "Packed operands must have rank two");
    TORCH_CHECK(weight.scalar_type() == at::kInt && zeros.scalar_type() == at::kInt && scales.scalar_type() == at::kHalf,
                "ExLlamaV2 requires int32 packed words and FP16 scales");
    TORCH_CHECK(weight.is_contiguous() && zeros.is_contiguous() && scales.is_contiguous(), "Packed operands must be contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.device() == zeros.device() && weight.device() == scales.device(), "Operand devices must match");
    TORCH_CHECK(weight.device().index() == source->device, "Descriptor and operands must use the same device");
    TORCH_CHECK(weight.size(0) * 8 == source->height && weight.size(1) % 32 == 0, "Native packed matrix dimensions do not match");
    TORCH_CHECK(zeros.size(0) == source->groups && scales.size(0) == source->groups &&
                zeros.size(1) * 8 == weight.size(1) && scales.size(1) == weight.size(1), "Native scale/zero dimensions do not match");
    for (auto item : {perm, invperm}) {
        TORCH_CHECK(!item.has_value() || (item->device() == weight.device() && item->scalar_type() == at::kShort &&
                    item->dim() == 1 && item->numel() == source->height && item->is_contiguous()),
                    "Input permutations must be contiguous int16 vectors on the matrix device");
    }
    TORCH_CHECK(perm.has_value() == invperm.has_value(), "Both input permutations must be provided together");
    QMatrix* result = new QMatrix(*source);
    result->width = weight.size(1);
    result->cuda_q_weight = reinterpret_cast<uint32_t*>(weight.data_ptr());
    result->cuda_gptq_qzeros = reinterpret_cast<uint32_t*>(zeros.data_ptr());
    result->cuda_gptq_scales = reinterpret_cast<half*>(scales.data_ptr());
    result->cuda_q_perm = perm.has_value() ? reinterpret_cast<uint16_t*>(perm->data_ptr()) : nullptr;
    result->cuda_q_invperm = invperm.has_value() ? reinterpret_cast<uint16_t*>(invperm->data_ptr()) : nullptr;
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(result));
}

void free_q_matrix(int64_t handle)
{
    delete reinterpret_cast<QMatrix*>(static_cast<uintptr_t>(handle));
}
