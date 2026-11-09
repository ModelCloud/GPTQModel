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
// Exposes codec stages for overflow tests without allocating ~100 GB inputs.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nf12::codec {

// In-place exclusive scan of v[0, n), with the total in v[n].
// Use 64-bit counts because N*K/8 can exceed 2^32 on supported devices.
cudaError_t escape_offsets_scan(uint64_t* v, int64_t n, cudaStream_t stream);

inline constexpr int64_t kMaxGridX = 0x7fffffff;

}
