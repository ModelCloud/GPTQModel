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
// Per-expert counts are device-only, so dispatch uses host-known dimensions.
// NF12 and bf16t must select the same path and config.
// PDL kernels read routing metadata only after griddepcontrol.wait
// (linear/dispatch.cu).
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <nf12/format.cuh>

#include "internal.h"
#include "moe/sm80/large_mma.h"
#include "moe/sm120/large_sm120.h"
#include "moe/sm90/large_sm90.h"
#include "moe/sm100/large_tc.h"
#include "moe/schedule.cuh"
#include "moe/small.h"

namespace nf12::detail {
namespace {

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// Expected sorted rows per non-empty expert under uniform routing.
double expected_rows_per_expert(int64_t max_assignments, int64_t num_experts) {
  const double A = static_cast<double>(max_assignments);
  const double E = static_cast<double>(num_experts);
  return A / (E * (1.0 - std::pow(1.0 - 1.0 / E, A)));
}

// Measured decode/grouped crossovers.
double small_max_rows_per_expert(const DeviceInfo& dev) {
  switch (dev.sm_arch) {
    case 80: return 32.0;
    case 86: return 40.0;
    case 89: return 48.0;
    case 90: return 20.0;
    case 100:
    case 103: return 10.0;
    default: return 6.0;
  }
}

bool use_small(int64_t max_assignments, int64_t num_experts, const DeviceInfo& dev) {
  return expected_rows_per_expert(max_assignments, num_experts) <= small_max_rows_per_expert(dev);
}

bool use_sm90(int64_t num_experts, const DeviceInfo& dev) { return moe_large_sm90_supported(dev, num_experts); }

constexpr double kNever = 1e300;

// Below SM90, torch._grouped_mm falls back to a host-synchronizing per-group
// loop, violating the no-host-sync and graph-capture contracts.
struct MoeSliceRoute {
  int sm_arch;
  double min_rows_per_expert;
};

constexpr MoeSliceRoute kMoeSliceRoutes[] = {
    {80, kNever},
    {86, kNever},
    {89, kNever},
    {90, 384},
    {100, kNever},
    {103, kNever},
};

// Bound on one decoded slice of all experts (BF16 bytes), from the caller's allocator.
constexpr int64_t kMoeSliceBytes = int64_t(1) << 30;

}  // namespace

size_t moe_linear_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                  const DeviceInfo& dev) {
  if (max_assignments <= 0 || num_experts < 1 || use_small(max_assignments, num_experts, dev)) return 0;
  if (moe_large_tc_supported(dev)) return moe_large_tc_workspace_bytes(max_assignments, num_experts, n, k, dev);
  if (use_sm90(num_experts, dev))
    return moe_linear_large_sm90_workspace_bytes(max_assignments, num_experts, n, k, dev);
  if (moe_large_sm120_supported(dev, num_experts))
    return moe_linear_large_sm120_workspace_bytes(max_assignments, num_experts, n, k, dev);
  return moe_linear_large_mma_workspace_bytes(max_assignments, num_experts, n, k, dev);
}

int64_t moe_linear_decode_gemm_rows(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                    const DeviceInfo& dev) {
  if (max_assignments <= 0 || num_experts < 1 || num_experts > kMoeMaxExperts) return 0;
  double min_rows = kNever;
  for (const MoeSliceRoute& r : kMoeSliceRoutes) {
    if (r.sm_arch == dev.sm_arch) min_rows = r.min_rows_per_expert;
  }
  if (expected_rows_per_expert(max_assignments, num_experts) < min_rows) return 0;
  // Equal slices of at most kMoeSliceBytes over all experts, never below 16 rows.
  const int64_t cap = std::max<int64_t>(16, kMoeSliceBytes / (2 * k * num_experts) / 16 * 16);
  const int64_t slices = (n + cap - 1) / cap;
  return (n / 16 + slices - 1) / slices * 16;
}

cudaError_t moe_linear(const MoeLinearArgs& a) {
  if (a.dev == nullptr) return cudaErrorInvalidValue;
  if (a.dev->sm_arch < 80) return cudaErrorNotSupported;
  if (a.num_experts < 1 || a.n <= 0 || a.k <= 0 || a.n % 16 != 0 || a.k % 64 != 0) return cudaErrorInvalidValue;
  if (a.n > INT32_MAX || a.k > INT32_MAX || a.max_assignments < 0 || a.max_assignments > INT32_MAX)
    return cudaErrorInvalidValue;
  if (a.gather_div < 1) return cudaErrorInvalidValue;
  if (a.x == nullptr || a.y == nullptr || a.weight == nullptr || a.expert_offsets == nullptr)
    return cudaErrorInvalidValue;
  // Rows of x and y are moved in 16-byte vectors.
  if (a.ldx < a.k || a.ldx % 8 != 0 || a.ldy < a.n || a.ldy % 8 != 0) return cudaErrorInvalidValue;
  if (!aligned16(a.x) || !aligned16(a.y) || !aligned16(a.weight)) return cudaErrorInvalidValue;
  if (a.source == WeightSource::kNF12 &&
      (a.exp_base == nullptr || a.escape_offsets == nullptr || (a.escapes != nullptr && !aligned16(a.escapes))))
    return cudaErrorInvalidValue;
  if (a.num_experts > kMoeMaxExperts) return cudaErrorNotSupported;
  const DeviceInfo& dev = *a.dev;
  const size_t ws = moe_linear_workspace_bytes(a.max_assignments, a.num_experts, a.n, a.k, dev);
  if (a.workspace_bytes < ws || (ws > 0 && a.workspace == nullptr)) return cudaErrorInvalidValue;
  if (a.max_assignments == 0) return cudaSuccess;

  if (use_small(a.max_assignments, a.num_experts, dev)) {
    const MoeSmallConfig cfg = moe_small_heuristic(a.max_assignments, a.num_experts, a.n, a.k, dev);
    return cfg.mt == 0 ? cudaErrorNotSupported : moe_linear_small(a, cfg);
  }
  if (moe_large_tc_supported(dev)) return moe_linear_large_tc(a);
  if (use_sm90(a.num_experts, dev)) return moe_linear_large_sm90(a);
  if (moe_large_sm120_supported(dev, a.num_experts)) return moe_linear_large_sm120(a);
  return moe_linear_large_mma(a);
}

}  // namespace nf12::detail
