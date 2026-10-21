#pragma once

#include "cpu_features.hpp"
#include "gptq_avx512.hpp"
#include <torch/all.h>
#include <stdexcept>
#include <ATen/ATen.h>

namespace gptq_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void gemm_int4(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &zeros,
        const torch::Tensor &absmax, torch::Tensor &out, int64_t blocksize);
} // namespace gptq_cpu