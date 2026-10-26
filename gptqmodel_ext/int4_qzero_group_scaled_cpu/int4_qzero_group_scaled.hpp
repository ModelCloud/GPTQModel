// Based on the Hugging Face kernels-community `quantization-gptq` CPU kernel:
// https://github.com/huggingface/kernels-community/tree/c467920211e144ea621d9e8193ff71ec160ce256/quantization-gptq
// Original author: jiqing-feng <jiqing.feng@intel.com> (Intel).
// Modified for GPT-QModel by ModelCloud.ai.

#pragma once

#include "cpu_features.hpp"
#include "int4_qzero_group_scaled_avx512.hpp"
#include <torch/all.h>
#include <optional>
#include <stdexcept>
#include <ATen/ATen.h>

namespace int4_qzero_group_scaled_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void gemm_int4(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &zeros,
        const torch::Tensor &absmax, const std::optional<torch::Tensor> &bias,
        torch::Tensor &out, int64_t blocksize);
} // namespace int4_qzero_group_scaled_cpu
