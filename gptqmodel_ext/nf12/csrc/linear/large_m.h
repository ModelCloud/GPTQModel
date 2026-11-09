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
// Device-family launcher for compute-regime linear. dispatch.cu owns the crossover.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// Workspace bytes for (m, n, k) on `dev`, or 0 for an unsplit config.
size_t linear_large_m_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Requires args.m >= 1, n % 16 == 0, k % 64 == 0, 16-byte aligned x/y rows
// (ldx and ldy multiples of 8), and args.workspace_bytes >=
// linear_large_m_workspace_bytes(...).
cudaError_t linear_large_m(const LinearArgs& args);

}  // namespace nf12::detail
