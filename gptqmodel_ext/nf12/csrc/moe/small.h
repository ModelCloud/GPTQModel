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
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

struct MoeSmallConfig {
  int mt;      // 8-row tiles per item, one of 1, 2, 4, 8
  int warps;   // 4 or 8
  int kc;      // cluster K split, 1..8 (must be 1 below SM90)
  int slice;   // super-atoms per x slice, a multiple of small_m_step(...) * kp
               // 0 uses the whole CTA K range
  int pdl;     // 1 enables PDL on SM90+ (0 disables it)
  int kp = 1;  // K parts per CTA (SM8x only), leaving warps / kp row blocks
};

bool moe_small_config_supported(const MoeSmallConfig& cfg, int64_t num_experts, int64_t n, int64_t k,
                                const DeviceInfo& dev);

// mt == 0 means no supported config. Uses host-known values only.
MoeSmallConfig moe_small_heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                   const DeviceInfo& dev);

// Requires moe_linear() argument validation, max_assignments >= 1 and a
// supported config. No workspace is needed.
cudaError_t moe_linear_small(const MoeLinearArgs& args, const MoeSmallConfig& cfg);

}  // namespace nf12::detail
