// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <vector>


namespace int4_qzero_group_scaled_cpu
{

    std::vector<torch::Tensor> pack_gptq_int4(
        const torch::Tensor &qweight,
        const torch::Tensor &qzeros,
        const torch::Tensor &ret_idx,
        int64_t out_features,
        int64_t in_features);

    std::vector<torch::Tensor> pack_awq_int4(
        const torch::Tensor &qweight,
        const torch::Tensor &qzeros,
        int64_t out_features,
        int64_t in_features);

} // namespace int4_qzero_group_scaled_cpu
