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

#include <cuda_bf16.h>

#include "internal.h"

namespace nf12::detail {

// Write only valid sorted positions, with row stride a.k. Offsets use
// moe_linear's clamped, monotone interpretation.
// Requires a.gather != nullptr and moe_linear() argument validation.
// SM90+ launches use programmatic stream serialization.
cudaError_t moe_gather_rows(const MoeLinearArgs& a, __nv_bfloat16* out);

}  // namespace nf12::detail
