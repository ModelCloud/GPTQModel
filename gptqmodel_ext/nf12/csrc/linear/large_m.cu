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
// Routing depends only on shape and device, so NF12 and bf16t share a kernel.
#include "linear/large_m.h"

#include "linear/sm80/large_m_mma.h"
#include "linear/sm120/large_m_sm120.h"
#include "linear/sm90/large_m_sm90.h"
#include "linear/sm100/large_m_tc.h"

namespace nf12::detail {
namespace {

bool use_sm90(const DeviceInfo& dev) { return large_m_sm90_supported(dev); }

bool use_sm120(int64_t m, int64_t n, const DeviceInfo& dev) {
  const bool fits = m <= 192 || (m <= 384 && n < 32768) || (m <= 512 && n < 16384) || (m <= 1024 && n <= 4096);
  return fits && large_m_sm120_supported(dev);
}

bool use_compact_mma(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (dev.sm_arch != 120 || m < 48 || k < 512 || n < 4096) return false;
  return (m <= 64 && k <= 2048 && n <= 8192) || (m <= 128 && k <= 896 && n <= 4864);
}

constexpr int kCompactMmaConfig = 14;

}  // namespace

size_t linear_large_m_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (large_m_tc_supported(dev)) return linear_large_m_tc_workspace_bytes(m, n, k, dev, -1);
  if (use_sm90(dev)) return linear_large_m_sm90_workspace_bytes(m, n, k, dev, -1);
  if (use_compact_mma(m, n, k, dev)) return linear_large_m_mma_workspace_bytes(m, n, k, dev, kCompactMmaConfig);
  if (use_sm120(m, n, dev)) return linear_large_m_sm120_workspace_bytes(m, n, k, dev, -1);
  return linear_large_m_mma_workspace_bytes(m, n, k, dev, -1);
}

cudaError_t linear_large_m(const LinearArgs& args) {
  if (large_m_tc_supported(*args.dev)) return linear_large_m_tc(args, -1);
  if (use_sm90(*args.dev)) return linear_large_m_sm90(args, -1);
  if (use_compact_mma(args.m, args.n, args.k, *args.dev)) return linear_large_m_mma(args, kCompactMmaConfig);
  if (use_sm120(args.m, args.n, *args.dev)) return linear_large_m_sm120(args, -1);
  return linear_large_m_mma(args, -1);
}

}  // namespace nf12::detail
