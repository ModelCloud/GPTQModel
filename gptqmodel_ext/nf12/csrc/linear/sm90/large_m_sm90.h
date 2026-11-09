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
// SM90 TMA-fed wgmma with register-A decode or cooperative shared-A decode.
// `config < 0` selects the heuristic. Nonnegative configs force entry
// `config % 16` with schedule `config / 16 % 16` (0 = heuristic hybrid,
// 1 = data-parallel, 2 = stream-K, 3 = per-tile split-K, 4 = cluster split-K).
// Schedules 3 and 4 use `config / 256` splits. Only register-A configs
// support schedule 4.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// True when `dev` is an SM90 GPU and the loaded image carries the
// arch-specific (sm_90a) kernel body.
bool large_m_sm90_supported(const DeviceInfo& dev);
int large_m_sm90_num_configs();
int large_m_sm90_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);  // encoded as above
const char* large_m_sm90_config_name(int config);
size_t linear_large_m_sm90_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config);
// Requires linear_large_m() preconditions and args.workspace_bytes >=
// linear_large_m_sm90_workspace_bytes(...).
cudaError_t linear_large_m_sm90(const LinearArgs& args, int config);

}  // namespace nf12::detail
