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
// SM80+ with no split-K or workspace.
// Select heuristically with config < 0, or force a table entry with config >= 0.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

int moe_large_mma_num_configs();
const char* moe_large_mma_config_name(int config);
int moe_large_mma_heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                            const DeviceInfo& dev);
size_t moe_linear_large_mma_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                            const DeviceInfo& dev, int config = -1);
// Requires moe_linear() argument validation and SM80+.
cudaError_t moe_linear_large_mma(const MoeLinearArgs& args, int config = -1);

}  // namespace nf12::detail
