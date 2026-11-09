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
// Streaming GEMV/skinny-GEMM decodes super-atoms directly into MMA A fragments.
// Configs are exposed for tests and tuning.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

inline constexpr int kSmallMMaxM = 64;

struct SmallMConfig {
  int mt;      // n8 token tiles per mma column block: 1, 2, 4 or 8 (covers m <= 8 * mt)
  int warps;   // warps per CTA: 4 or 8 (16 below SM90)
  int kc;      // K split across CTAs, 1..8: a thread-block cluster from SM90, below SM90
               // independent CTAs whose FP32 partials a second kernel sums (workspace)
  int slice;   // super-atoms of x per double-buffered shared-memory slice, a multiple of
               // small_m_step(small_m_stages(...)) * kp; 0 = whole CTA K range
  int kp = 1;  // K parts per CTA (small_m_core.cuh), below SM90 only: warps / kp row blocks per CTA
};

// Shared-memory ring stages available on `dev`, or 0 for register-ring-only devices.
int small_m_bulk_stages(const DeviceInfo& dev);

// Weight ring the dense kernel runs `cfg` with on an n x k weight: 0 =
// register ring, else the stages of the per-warp shared-memory ring
// (small_m_core.cuh).
int small_m_stages(const SmallMConfig& cfg, int64_t n, int64_t k, const DeviceInfo& dev);

// Whether `cfg` can run (m, n, k) on `dev` (template exists, smem fits, cluster legal).
bool small_m_config_supported(const SmallMConfig& cfg, int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Resident CTAs per SM for `cfg` at (m, n, k) on `dev` (0 if it cannot run).
int small_m_resident_ctas(const SmallMConfig& cfg, int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Tuned choice for m <= kSmallMMaxM (mt == 0 if no config can run on `dev`).
SmallMConfig small_m_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// The x slice small_m_heuristic() pairs with cfg's (mt, warps, kc, kp)
// (cfg.slice is ignored), -1 if that config cannot run (tests, tuning).
int small_m_heuristic_slice(const SmallMConfig& cfg, int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Workspace of `cfg` at (m, n): below SM90 with kc > 1 the FP32 partials
// (kc * m * n floats), else 0 (split-K partials are reduced through cluster
// shared memory).
size_t small_m_workspace_bytes(const SmallMConfig& cfg, int64_t m, int64_t n, const DeviceInfo& dev);

// Requires args validated per SPEC, small_m_config_supported(cfg, ...) and
// small_m_workspace_bytes(cfg, ...) bytes of 16-byte aligned workspace.
cudaError_t linear_small_m(const LinearArgs& args, const SmallMConfig& cfg);

// Shared host kernel state is memoized per (kernel, device). `device` must
// be current. Raises the dynamic shared-memory cap once when smem > 48 KB.
cudaError_t small_m_prepare_kernel(const void* fn, const DeviceInfo& dev, size_t smem);
// Resident CTAs per SM for `fn` (0 if it cannot run), cached per smem size.
int small_m_kernel_occupancy(const void* fn, const DeviceInfo& dev, int threads, size_t smem);

}  // namespace nf12::detail
