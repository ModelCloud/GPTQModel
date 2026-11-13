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

struct SmallMAsyncConfig {
  int stages = 0;  // Zero disables this family; supported counts are 2, 3, 4, 8.
  bool stage_escapes = false;
};

SmallMAsyncConfig small_m_async_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Requires publicly validated arguments: 1 <= M <= 8, N divisible by 16,
// positive K divisible by 256, and 16-byte-aligned inputs/activation rows.
// Runtime ldx/ldy retain their 64-bit range. No workspace is used.
// Only immutable NF12 weights may precede the PDL dependency wait.
cudaError_t linear_small_m_async(const LinearArgs& args, const SmallMAsyncConfig& config);

}  // namespace nf12::detail
