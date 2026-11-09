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
// Persistent CTAs traverse expert-major tiles using the dense grouped raster.
// Each tile is owned by one CTA with no split-K, preserving accumulation order.
// Gathered rows past an expert's end are zero-filled without global reads and
// never stored. Apply row_scale in FP32 before BF16 rounding.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>

#include "linear/mma.cuh"
#include "moe/common.cuh"
#include "moe/sm80/large_mma.h"
#include "moe/schedule.cuh"

namespace nf12::detail {
namespace {

using mma::Loop;
using mma::Tile;

struct MoeMmaParams {
  const __nv_bfloat16* x;
  int64_t ldx;
  const int32_t* gather;
  int64_t gather_div;
  const int32_t* scatter;
  const float* row_scale;
  const int64_t* expert_offsets;
  int64_t max_assignments;
  int num_experts;
  const uint8_t* w;
  int64_t expert_bytes;
  const uint4* escapes;  // null for bf16t, or when no expert has escapes
  const int64_t* escape_offsets;
  const uint8_t* exp_base;
  int n, kblocks;
  __nv_bfloat16* y;
  int64_t ldy;
  int n_tiles;
  int group;  // n-tiles per rasterization group
};

// in == -1 marks rows past the expert's end.
struct RowMap {
  int32_t in, out;
  float scale;
};

// Shared row maps avoid register spills for 64x64 warp tiles.
// Zero-fill padding rather than rereading the last row, which creates an L2 hot spot.
template <class C>
struct GatherActLoader {
  static constexpr int kChunks = C::BM * 8;
  static constexpr int kIters = (kChunks + C::kThreads - 1) / C::kThreads;
  const __nv_bfloat16* x;
  int64_t ldx;
  const RowMap* rows;  // shared, [BM]

  __device__ __forceinline__ void load(uint32_t dst, int kb_rel) const {
    const __nv_bfloat16* src = x + static_cast<int64_t>(kb_rel) * 64;
#pragma unroll
    for (int i = 0; i < kIters; ++i) {
      const int c = threadIdx.x + i * C::kThreads;
      if (kChunks % C::kThreads != 0 && c >= kChunks) continue;
      const int row = c / 8, ch = c % 8;
      const int32_t in = rows[row].in;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(mma::act_chunk_dst(dst, row, ch)),
                   "l"(src + (in >= 0 ? in : 0) * ldx + ch * 8), "r"(in >= 0 ? 16u : 0u));
    }
  }
};

class MoeProblem {
 public:
  using Params = MoeMmaParams;
  struct Work {
    int n0, m0, kb_begin, nkb;
    int rows;  // valid rows of the tile, from sorted position m0
    const uint8_t* w;
    const uint4* escapes;
    uint32_t num_escapes, base_x4;
  };
  struct RowTerms {};

  // Shared memory past the pipeline: RowMap[BM], then the schedule.
  static size_t extra_smem_bytes(int bm, int num_experts) {
    return static_cast<size_t>(bm) * sizeof(RowMap) + sizeof(int) * moe_schedule_smem_ints(num_experts);
  }

  __device__ __forceinline__ MoeProblem(const Params& p, uint8_t* extra) : p_(p) {
    rows_ = reinterpret_cast<RowMap*>(extra);
  }

  template <class C>
  __device__ __forceinline__ bool next(Work& w) {
    if (tile_ < 0) {
      sched_ = reinterpret_cast<int*>(rows_ + C::BM);
      const int mtiles = moe_mtile_prefix(p_.expert_offsets, p_.num_experts, C::BM, p_.max_assignments, sched_);
      tiles_ = mtiles * p_.n_tiles;  // < 2^31: the host bounds the launch's tile space
      tile_ = static_cast<int>(blockIdx.x);
    } else {
      tile_ += static_cast<int>(gridDim.x);
      // The previous tile's epilogue still reads rows_ and the staging tile,
      // which this tile's row map and prologue overwrite.
      __syncthreads();
    }
    if (tile_ >= tiles_) return false;
    const int E = p_.num_experts;
    const int e = moe_mtile_expert(sched_, E, tile_ / p_.n_tiles);
    const int first = sched_[e];
    int nt, mt;
    tile_raster(tile_ - first * p_.n_tiles, p_.n_tiles, sched_[e + 1] - first, p_.group, nt, mt);
    int row0, rows;
    moe_mtile_rows(sched_, E, e, mt, C::BM, row0, rows);
    w.n0 = nt * C::BN;
    w.m0 = row0;
    w.kb_begin = 0;
    w.nkb = p_.kblocks;
    w.rows = rows;
    w.w = p_.w + e * p_.expert_bytes;
    w.escapes = nullptr;
    w.num_escapes = 0;
    w.base_x4 = 0;
    if (p_.exp_base != nullptr) w.base_x4 = exp_base_x4(__ldg(p_.exp_base + e));
    if (p_.escapes != nullptr) {
      const int64_t e0 = __ldg(p_.escape_offsets + e);
      w.escapes = p_.escapes + e0;
      w.num_escapes = static_cast<uint32_t>(__ldg(p_.escape_offsets + e + 1) - e0);
    }
    for (int r = threadIdx.x; r < C::BM; r += C::kThreads) {
      RowMap m;
      m.in = r < rows ? static_cast<int32_t>(moe_in_row(p_.gather, p_.gather_div, row0 + r)) : -1;
      m.out = r < rows ? static_cast<int32_t>(moe_out_row(p_.scatter, row0 + r)) : 0;
      m.scale = r < rows && p_.row_scale != nullptr ? __ldg(p_.row_scale + m.out) : 1.f;
      rows_[r] = m;
    }
    __syncthreads();  // the activation loads of every thread read the whole map
    return true;
  }

  __device__ __forceinline__ const uint8_t* weights(const Work& w) const { return w.w; }
  __device__ __forceinline__ const uint4* escapes(const Work& w) const { return w.escapes; }
  __device__ __forceinline__ uint32_t num_escapes(const Work& w) const { return w.num_escapes; }
  __device__ __forceinline__ uint32_t base_x4(const Work& w) const { return w.base_x4; }
  __device__ __forceinline__ int n() const { return p_.n; }
  __device__ __forceinline__ int kblocks() const { return p_.kblocks; }

  template <class C>
  __device__ __forceinline__ GatherActLoader<C> act(const Work&) const {
    return GatherActLoader<C>{p_.x, p_.ldx, rows_};
  }
  template <class C>
  __device__ __forceinline__ RowTerms row_terms(const Work&, int) const {
    return {};
  }
  template <class C>
  __device__ __forceinline__ void fragment(const Work&, const RowTerms&, int tok, float (&v)[4]) const {
    if (p_.row_scale != nullptr) {
      const float s0 = rows_[tok].scale, s1 = rows_[tok + 1].scale;
      v[0] *= s0;
      v[1] *= s1;
      v[2] *= s0;
      v[3] *= s1;
    }
  }
  __device__ __forceinline__ __nv_bfloat16* out_row(const Work& w, int row) const {
    return row < w.rows ? p_.y + static_cast<int64_t>(rows_[row].out) * p_.ldy + w.n0 : nullptr;
  }

 private:
  const Params& p_;
  RowMap* rows_;
  int* sched_ = nullptr;
  int tile_ = -1, tiles_ = 0;
};

struct MoeConfig {
  const char* name;
  cudaError_t (*launch)(const MoeLinearArgs&);
  size_t (*smem)(int64_t num_experts);
};

template <class C, Loop L>
size_t config_smem(int64_t num_experts) {
  // Both sources must fit, so a config never runs NF12 without its bf16t twin.
  return std::max(C::template smem_bytes<WeightSource::kNF12>(L, false),
                  C::template smem_bytes<WeightSource::kBF16T>(L, false)) +
         MoeProblem::extra_smem_bytes(C::BM, static_cast<int>(num_experts));
}

template <class C, Loop L, WeightSource S>
constexpr auto kernel_of() {
  if constexpr (L == Loop::kStream)
    return &mma::large_m_stream_kernel<C, S, MoeProblem>;
  else if constexpr (L == Loop::kRing)
    return &mma::large_m_ring_kernel<C, S, MoeProblem>;
  else if constexpr (L == Loop::kBudget)
    return &mma::large_m_budget_kernel<C, S, MoeProblem>;
  else
    return &mma::large_m_staged_kernel<C, S, MoeProblem>;
}

template <class C, Loop L, WeightSource S>
cudaError_t launch_as(const MoeLinearArgs& a) {
  if constexpr (L == Loop::kBudget) {
    if (a.dev->sm_arch >= 90) return cudaErrorNotSupported;
  }
  constexpr auto kernel = kernel_of<C, L, S>();
  const int64_t n_tiles = (a.n + C::BN - 1) / C::BN;
  const int64_t bound = moe_max_mtiles(a.max_assignments, a.num_experts, C::BM) * n_tiles;
  if (bound == 0) return cudaSuccess;
  // Tile indices, rows and k-blocks are 32-bit. Byte offsets are 64-bit.
  if (bound > INT32_MAX || a.n > INT32_MAX || a.k / 64 > INT32_MAX) return cudaErrorNotSupported;
  const size_t smem_total = C::template smem_bytes<S>(L, false) +
                            MoeProblem::extra_smem_bytes(C::BM, static_cast<int>(a.num_experts));
  if (smem_total > a.dev->smem_optin_bytes) return cudaErrorNotSupported;
  const int smem = static_cast<int>(smem_total);
  static std::atomic<int> granted[64];
  cudaError_t err = mma::ensure_smem(reinterpret_cast<const void*>(kernel), granted, a.dev->device, smem);
  if (err != cudaSuccess) return err;
  // Cache (smem << 8 | blocks) because expert count changes shared-memory use.
  static std::atomic<int64_t> occupancy[64];
  const int dev = a.dev->device;
  int blocks = 0;
  if (dev >= 0 && dev < 64) {
    const int64_t c = occupancy[dev].load(std::memory_order_relaxed);
    if ((c >> 8) == smem) blocks = static_cast<int>(c & 0xff);
  }
  if (blocks == 0) {
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, C::kThreads, smem);
    if (err != cudaSuccess) return err;
    blocks = std::clamp(blocks, 1, 255);
    if (dev >= 0 && dev < 64) occupancy[dev].store((static_cast<int64_t>(smem) << 8) | blocks, std::memory_order_relaxed);
  }

  MoeMmaParams p;
  p.x = a.x;
  p.ldx = a.ldx;
  p.gather = a.gather;
  p.gather_div = a.gather_div;
  p.scatter = a.scatter;
  p.row_scale = a.row_scale;
  p.expert_offsets = a.expert_offsets;
  p.max_assignments = a.max_assignments;
  p.num_experts = static_cast<int>(a.num_experts);
  p.w = a.weight;
  p.expert_bytes = moe_expert_bytes(S, a.n, a.k);
  const bool nf12 = S == WeightSource::kNF12;
  p.escapes = nf12 ? a.escapes : nullptr;
  p.escape_offsets = nf12 ? a.escape_offsets : nullptr;
  p.exp_base = nf12 ? a.exp_base : nullptr;
  p.n = static_cast<int>(a.n);
  p.kblocks = static_cast<int>(a.k / 64);
  p.y = a.y;
  p.ldy = a.ldy;
  p.n_tiles = static_cast<int>(n_tiles);
  p.group = static_cast<int>(std::min<int64_t>(8, n_tiles));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(std::min<int64_t>(bound, static_cast<int64_t>(a.dev->num_sms) * blocks)));
  cfg.blockDim = dim3(C::kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = a.stream;
  return cudaLaunchKernelEx(&cfg, kernel, p);
}

template <class C, Loop L>
cudaError_t config_launch(const MoeLinearArgs& a) {
  return a.source == WeightSource::kNF12 ? launch_as<C, L, WeightSource::kNF12>(a)
                                         : launch_as<C, L, WeightSource::kBF16T>(a);
}

#define NF12_MOE_CONFIG(BN, BM, WN, WM, ST, LOOP)                                                              \
  MoeConfig {                                                                                                  \
    #LOOP "_" #BN "x" #BM "_w" #WN "x" #WM "_s" #ST, &config_launch<Tile<BN, BM, WN, WM, ST>, Loop::k##LOOP>,  \
        &config_smem<Tile<BN, BM, WN, WM, ST>, Loop::k##LOOP>                                                  \
  }

const MoeConfig kConfigs[] = {
    NF12_MOE_CONFIG(128, 128, 4, 2, 2, Staged),
    NF12_MOE_CONFIG(128, 64, 4, 2, 2, Staged),
    NF12_MOE_CONFIG(256, 128, 4, 2, 2, Staged),
    NF12_MOE_CONFIG(128, 64, 4, 2, 4, Stream),
    NF12_MOE_CONFIG(256, 64, 4, 2, 2, Staged),
    NF12_MOE_CONFIG(256, 128, 4, 2, 3, Ring),
    NF12_MOE_CONFIG(256, 128, 4, 2, 2, Budget),
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

}  // namespace

int moe_large_mma_num_configs() { return kNumConfigs; }

const char* moe_large_mma_config_name(int config) {
  return config >= 0 && config < kNumConfigs ? kConfigs[config].name : "invalid";
}

int moe_large_mma_heuristic(int64_t max_assignments, int64_t num_experts, int64_t, int64_t k, const DeviceInfo& dev) {
  const double A = static_cast<double>(max_assignments), E = static_cast<double>(num_experts);
  const double rows = A / (E * (1.0 - std::pow(1.0 - 1.0 / E, A)));
  enum {
    k128x128 = 0, k128x64 = 1, k256x128 = 2, kStream128x64 = 3,
    k256x64 = 4, kRing256x128 = 5, kBudget256x128 = 6
  };
  int preferred;
  switch (dev.sm_arch) {
    case 80: preferred = rows <= 48 ? k256x64 : kRing256x128; break;
    case 86:
      preferred = rows > 256 ? kBudget256x128 : rows > 48 ? k128x128 : k >= 2048 ? kStream128x64 : k128x64;
      break;
    case 89: preferred = rows <= 48 ? k128x64 : kBudget256x128; break;
    default: preferred = rows <= 48 ? k128x64 : rows <= 160 ? k128x128 : k256x128; break;
  }
  // Larger expert counts leave less shared memory for tile staging.
  const int order[] = {preferred, k256x128, k128x128, k128x64};
  for (const int c : order)
    if (kConfigs[c].smem(num_experts) <= dev.smem_optin_bytes) return c;
  return k128x64;
}

size_t moe_linear_large_mma_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int) { return 0; }

cudaError_t moe_linear_large_mma(const MoeLinearArgs& a, int config) {
  if (a.dev == nullptr || a.dev->sm_arch < 80) return cudaErrorNotSupported;
  if (config >= kNumConfigs) return cudaErrorInvalidValue;
  const int c = config >= 0 ? config : moe_large_mma_heuristic(a.max_assignments, a.num_experts, a.n, a.k, *a.dev);
  if (kConfigs[c].smem(a.num_experts) > a.dev->smem_optin_bytes) return cudaErrorNotSupported;
  return kConfigs[c].launch(a);
}

}  // namespace nf12::detail
