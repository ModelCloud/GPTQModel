// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 / NF12-AD kernel computation.
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original implementation: https://github.com/dphnAI/nf12
// Copyright (c) 2026 AlpinDale and the nf12 contributors.
// Licensed under the Apache License, Version 2.0.
//
// Vendored into GPT-QModel (https://github.com/ModelCloud/GPTQModel) and
// modified for the NF12-AD (AlpinDale) method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
// Select heuristically with config < 0, or force a table entry with config >= 0.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// Shared memory for the schedule is reserved at compile time.
inline constexpr int kMoeSm120MaxExperts = 256;

// Reports support only for SM120/SM121 with a loaded arch-specific body.
// num_experts must be in [1, kMoeSm120MaxExperts].
bool moe_large_sm120_supported(const DeviceInfo& dev, int64_t num_experts);
int moe_large_sm120_num_configs();
const char* moe_large_sm120_config_name(int config);
int moe_large_sm120_heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                              const DeviceInfo& dev);
size_t moe_linear_large_sm120_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                              const DeviceInfo& dev, int config = -1);
// Requires moe_linear() argument validation.
cudaError_t moe_linear_large_sm120(const MoeLinearArgs& args, int config = -1);

}  // namespace nf12::detail
