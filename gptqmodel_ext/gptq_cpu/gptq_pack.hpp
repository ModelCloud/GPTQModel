// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <vector>


namespace gptq_cpu
{

    std::vector<torch::Tensor> pack_gptq_int4(
        const torch::Tensor &qweight,
        const torch::Tensor &qzeros,
        const torch::Tensor &ret_idx,
        int64_t out_features,
        int64_t in_features);

} // namespace gptq_cpu
