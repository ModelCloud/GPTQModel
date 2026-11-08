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
#include "internal.h"

namespace nf12::detail {
// SM89 router tiles reduce all warp-K ranks in shared memory within one CTA,
// without inter-CTA synchronization or global workspace.
bool narrow_sm8x_supported(const DeviceInfo& dev);
// Config IDs: 6 = m16/k16, 8 = m8/k16, 9 = m8/k32, 10 = m32/k16.
// k16/k32 count warp-local K ranks. Config 10 uses one activation slot,
// the others two. The heuristic selects tile size, not dispatch eligibility.
int narrow_sm8x_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);
cudaError_t linear_large_m_sm8x_narrow(const LinearArgs& args, int config = -1);
}  // namespace nf12::detail
