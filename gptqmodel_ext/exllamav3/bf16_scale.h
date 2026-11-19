// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <tuple>

#include "../torch_stable_compat.h"

// Fused BF16 <-> FP16 activation adaptation for the FP16-only EXL3 kernels.
// See gptqmodel_ext/exllamav3/bf16_scale.cu for the arithmetic contract.
std::tuple<at::Tensor, at::Tensor> bf16_prescale(const at::Tensor& x, double bound);

at::Tensor bf16_rescale(
    const at::Tensor& result,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& bias,
    int64_t n_cols);
