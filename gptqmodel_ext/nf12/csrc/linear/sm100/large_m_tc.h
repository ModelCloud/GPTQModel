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
// tcgen05/UMMA (SM100/103/110) family. Negative configs select the heuristic.
// Nonnegative configs force an entry of the config table.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// Requires SM100/103/110 and the architecture-specific tcgen05 kernel body
// in the image loaded for this device.
bool large_m_tc_supported(const DeviceInfo& dev);
int large_m_tc_num_configs();
const char* large_m_tc_config_name(int config);
size_t linear_large_m_tc_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config);
cudaError_t linear_large_m_tc(const LinearArgs& args, int config);

}  // namespace nf12::detail
