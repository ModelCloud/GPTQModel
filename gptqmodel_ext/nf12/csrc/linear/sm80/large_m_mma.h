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
// mma.sync (SM80+) family. `config < 0` selects the modelled heuristic.
// Nonnegative configs force entry `config % 16` with split-K factor
// `config / 16 % 16 + 1` after the first `config / 256` tiles, which run
// whole K in a separate launch.
// Entry 14 (Staged_64x32_w2x2_s2) is selected by the outer SM120 dispatch,
// not the throughput model. It uses the same BF16T twin and split-K encoding.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

#if defined(__CUDACC__)
#define NF12_LARGE_M_HD __host__ __device__ __forceinline__
#else
#define NF12_LARGE_M_HD inline
#endif

// `base + offset < limit` for 0 <= base < limit and offset >= 0, without
// forming base + offset, which can exceed INT32_MAX in a padded last tile.
NF12_LARGE_M_HD bool tile_offset_in_range(int base, int offset, int limit) { return offset < limit - base; }

int large_m_mma_num_configs();
const char* large_m_mma_config_name(int config);
int large_m_mma_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);  // encoded as above
size_t linear_large_m_mma_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config);
cudaError_t linear_large_m_mma(const LinearArgs& args, int config);

}  // namespace nf12::detail
