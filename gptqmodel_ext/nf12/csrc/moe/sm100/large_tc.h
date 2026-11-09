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
// Requires moe_linear() argument validation.
// Select heuristically with config < 0, or force a table entry with config >= 0.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// Reports support only with a loaded SM100/SM110-class sm_100a/sm_110a image.
bool moe_large_tc_supported(const DeviceInfo& dev);
int moe_large_tc_num_configs();
const char* moe_large_tc_config_name(int config);
size_t moe_large_tc_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                    const DeviceInfo& dev, int config = -1);
cudaError_t moe_linear_large_tc(const MoeLinearArgs& args, int config = -1);

// Fit bn * k BF16 activation panels in budget, capped at one expert's maximum
// m-tiles so band * n_tiles stays within the validated int32 tile bound.
inline int moe_large_tc_band(int64_t budget, int64_t max_assignments, int64_t k, int bn) {
  const int64_t x_panel = static_cast<int64_t>(bn) * k * 2;
  const int64_t expert_mtiles = std::max<int64_t>(1, (max_assignments + bn - 1) / bn);
  return static_cast<int>(std::clamp<int64_t>(budget / x_panel, 1, expert_mtiles));
}

}  // namespace nf12::detail
