// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 kernel computation.
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
// modified for the NF12 method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
#pragma once

#include "internal.h"

namespace nf12::detail {

struct SmallMDirectConfig {
  int warps = 0;  // Zero disables this family.
  int kc = 1;
};

SmallMDirectConfig small_m_direct_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Requires validated arguments with 1 <= M <= 8 and a supported configuration.
// Reads activations directly and accumulates in FP32 without workspace.
// weights_ready permits only immutable NF12 weights to precede the PDL wait.
cudaError_t linear_small_m_direct(const LinearArgs& args, const SmallMDirectConfig& config);

}  // namespace nf12::detail
