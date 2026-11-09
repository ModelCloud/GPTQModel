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
// Dense MMA tiles use grouped rasterization and optional split-K.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <string_view>

#include "linear/sm80/large_m_mma.h"
#include "linear/mma.cuh"

namespace nf12::detail {
namespace {

using mma::Loop;
using mma::Tile;

struct MmaParams {
  const __nv_bfloat16* x;
  int64_t ldx;
  int m;
  const uint8_t* w;
  const uint4* escapes;
  uint32_t num_escapes;
  uint32_t base_x4;
  int n;
  int kblocks;  // K / 64
  const __nv_bfloat16* bias;
  __nv_bfloat16* y;
  int64_t ldy;
  int n_tiles, m_tiles;
  int group;       // n-tiles per rasterization group
  int tile_begin;  // first tile (in rasterization order) of this launch
  // Split-K below SM90: `splits` ranks of each tile of the launch store their
  // partials at partials + (blockIdx.x * splits + rank) * kPartialsPerRank.
  float4* partials;
  int splits;
};

// Zero-fill tail tokens without global reads. Clamping to the last token
// creates an L2 hot spot across CTAs. Tail rows are never stored.
template <class C>
struct ActLoader {
  static constexpr int kChunks = C::BM * 8;
  static constexpr int kIters = (kChunks + C::kThreads - 1) / C::kThreads;
  const __nv_bfloat16* base;  // token m0, column kb_begin * 64
  int64_t ldx;
  int last_row;  // last token inside the matrix, relative to m0

  __device__ __forceinline__ void load(uint32_t dst, int kb_rel) const {
    const __nv_bfloat16* src = base + static_cast<int64_t>(kb_rel) * 64;
#pragma unroll
    for (int i = 0; i < kIters; ++i) {
      const int c = threadIdx.x + i * C::kThreads;
      if (kChunks % C::kThreads != 0 && c >= kChunks) continue;
      const int row = c / 8, ch = c % 8;
      const bool in = row <= last_row;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(mma::act_chunk_dst(dst, row, ch)),
                   "l"(src + (in ? row : 0) * ldx + ch * 8), "r"(in ? 16u : 0u));
    }
  }
};

// One matrix, one tile per CTA (blockIdx.x in rasterization order, offset by
// tile_begin), K split over blockIdx.z.
class DenseProblem {
 public:
  using Params = MmaParams;
  struct Work {
    int n0, m0, kb_begin, nkb;
  };
  struct RowTerms {
    float lo, hi;  // bias of rows g and g + 8
  };

  __device__ __forceinline__ DenseProblem(const Params& p, uint8_t*) : p_(p) {}

  template <class C>
  __device__ __forceinline__ bool next(Work& w) {
    if (done_) return false;
    done_ = true;
    int nt, mt;
    tile_raster(p_.tile_begin + static_cast<int>(blockIdx.x), p_.n_tiles, p_.m_tiles, p_.group, nt, mt);
    w.n0 = nt * C::BN;
    w.m0 = mt * C::BM;
    const int splits = gridDim.z;
    w.kb_begin = static_cast<int>(static_cast<int64_t>(p_.kblocks) * blockIdx.z / splits);
    w.nkb = static_cast<int>(static_cast<int64_t>(p_.kblocks) * (blockIdx.z + 1) / splits) - w.kb_begin;
    return true;
  }
  __device__ __forceinline__ const uint8_t* weights(const Work&) const { return p_.w; }
  __device__ __forceinline__ const uint4* escapes(const Work&) const { return p_.escapes; }
  __device__ __forceinline__ uint32_t num_escapes(const Work&) const { return p_.num_escapes; }
  __device__ __forceinline__ uint32_t base_x4(const Work&) const { return p_.base_x4; }
  __device__ __forceinline__ int n() const { return p_.n; }
  __device__ __forceinline__ int kblocks() const { return p_.kblocks; }

  template <class C>
  __device__ __forceinline__ ActLoader<C> act(const Work& w) const {
    return ActLoader<C>{p_.x + w.m0 * p_.ldx + static_cast<int64_t>(w.kb_begin) * 64, p_.ldx, p_.m - 1 - w.m0};
  }
  template <class C>
  __device__ __forceinline__ RowTerms row_terms(const Work& w, int nl) const {
    RowTerms r{0.f, 0.f};
    if (p_.bias != nullptr) {
      // Padded rows read the last bias and are never stored.
      const int last = p_.n - 1 - w.n0;
      r.lo = __bfloat162float(p_.bias[w.n0 + min(nl, last)]);
      r.hi = __bfloat162float(p_.bias[w.n0 + min(nl + 8, last)]);
    }
    return r;
  }
  template <class C>
  __device__ __forceinline__ void fragment(const Work&, const RowTerms& r, int, float (&v)[4]) const {
    v[0] += r.lo;
    v[1] += r.lo;
    v[2] += r.hi;
    v[3] += r.hi;
  }
  __device__ __forceinline__ __nv_bfloat16* out_row(const Work& w, int row) const {
    return tile_offset_in_range(w.m0, row, p_.m) ? p_.y + (static_cast<int64_t>(w.m0) + row) * p_.ldy + w.n0
                                                 : nullptr;
  }
  template <class C>
  __device__ __forceinline__ float4* partials(const Work&) const {
    return p_.partials + static_cast<size_t>(blockIdx.x) * p_.splits * mma::kPartialsPerRank<C>;
  }

 private:
  const Params& p_;
  bool done_ = false;
};


// The first `dp_tiles` rasterized tiles run whole K in one launch.
// The rest split K `splits` ways in a second launch, balancing the last
// partial wave without reducing every tile.
struct Plan {
  int splits, group;
  int64_t dp_tiles;
};

struct ConfigEntry {
  const char* name;
  int bn, bm;
  // Sustained compute-bound fraction of mma.sync peak, overridden by kArchModels.
  float efficiency;
  cudaError_t (*launch)(const LinearArgs&, const Plan&);
  int (*smem)(WeightSource, bool split);
};

template <class C, Loop L>
int tile_smem(WeightSource s, bool split) {
  return s == WeightSource::kNF12 ? C::template smem_bytes<WeightSource::kNF12>(L, split)
                                  : C::template smem_bytes<WeightSource::kBF16T>(L, split);
}

template <class C, Loop L, WeightSource S>
constexpr auto kernel_of() {
  if constexpr (L == Loop::kStream)
    return &mma::large_m_stream_kernel<C, S, DenseProblem>;
  else if constexpr (L == Loop::kBudget)
    return &mma::large_m_budget_kernel<C, S, DenseProblem>;
  else
    return &mma::large_m_staged_kernel<C, S, DenseProblem>;
}

// Split-K reduces through clusters on SM90+, or workspace and a second
// reduce launch on older architectures.
bool cluster_splits(const DeviceInfo& dev) { return dev.sm_arch >= 90; }

template <class C, Loop L, WeightSource S>
cudaError_t launch_as(const MmaParams& p, const LinearArgs& a, int splits, int tiles) {
  constexpr auto kernel = kernel_of<C, L, S>();
  static std::atomic<int> granted[64];
  const bool cluster = splits > 1 && cluster_splits(*a.dev);
  const int smem = C::template smem_bytes<S>(L, cluster);
  cudaError_t err = mma::ensure_smem(reinterpret_cast<const void*>(kernel), granted, a.dev->device, smem);
  if (err != cudaSuccess) return err;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(tiles), 1, static_cast<unsigned>(splits));
  cfg.blockDim = dim3(C::kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = a.stream;
  cudaLaunchAttribute attr[1];
  if (cluster) {
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = splits;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  err = cudaLaunchKernelEx(&cfg, kernel, p);
  if (err != cudaSuccess || splits == 1 || cluster) return err;

  constexpr auto reduce = &mma::large_m_split_reduce_kernel<C, DenseProblem>;
  static std::atomic<int> granted_reduce[64];
  const int reduce_smem = C::BM * C::kStagingStride * 2;
  err = mma::ensure_smem(reinterpret_cast<const void*>(reduce), granted_reduce, a.dev->device, reduce_smem);
  if (err != cudaSuccess) return err;
  cfg.gridDim = dim3(static_cast<unsigned>(tiles), 1, 1);
  cfg.dynamicSmemBytes = reduce_smem;
  return cudaLaunchKernelEx(&cfg, reduce, p, splits);
}

// Workspace of a plan's split launch below SM90 (FP32 partials of every
// rank of every split tile); 0 otherwise.
size_t plan_workspace_bytes(int bn, int bm, const Plan& plan, int64_t m, int64_t n, const DeviceInfo& dev) {
  if (plan.splits <= 1 || cluster_splits(dev)) return 0;
  const int64_t tiles = ((n + bn - 1) / bn) * ((m + bm - 1) / bm);
  return static_cast<size_t>(tiles - std::min(plan.dp_tiles, tiles)) * plan.splits * bn * bm * sizeof(float);
}

template <class C, Loop L>
cudaError_t tile_launch(const LinearArgs& a, const Plan& plan) {
  // Kernel dimensions and tile counts are 32-bit, while byte offsets are
  // 64-bit. Reject oversized grids rather than wrapping their indices.
  const int64_t n_tiles = (a.n + C::BN - 1) / C::BN, m_tiles = (a.m + C::BM - 1) / C::BM;
  const int64_t tiles = n_tiles * m_tiles;
  if (a.m > INT32_MAX || a.n > INT32_MAX || a.k / 64 > INT32_MAX || tiles > INT32_MAX) return cudaErrorNotSupported;
  if (a.workspace_bytes < plan_workspace_bytes(C::BN, C::BM, plan, a.m, a.n, *a.dev)) return cudaErrorInvalidValue;
  MmaParams p;
  p.x = a.x;
  p.ldx = a.ldx;
  p.m = static_cast<int>(a.m);
  p.w = a.weight;
  p.escapes = a.escapes;
  p.num_escapes = a.source == WeightSource::kNF12 ? a.num_escapes : 0;
  p.base_x4 = exp_base_x4(a.exp_base);
  p.n = static_cast<int>(a.n);
  p.kblocks = static_cast<int>(a.k / 64);
  p.bias = a.bias;
  p.y = a.y;
  p.ldy = a.ldy;
  p.n_tiles = static_cast<int>(n_tiles);
  p.m_tiles = static_cast<int>(m_tiles);
  p.group = std::max(1, std::min(plan.group, p.n_tiles));
  p.partials = static_cast<float4*>(a.workspace);
  const bool nf12 = a.source == WeightSource::kNF12;
  auto launch = [&](int64_t begin, int64_t count, int splits) {
    p.tile_begin = static_cast<int>(begin);
    p.splits = splits;
    return nf12 ? launch_as<C, L, WeightSource::kNF12>(p, a, splits, static_cast<int>(count))
                : launch_as<C, L, WeightSource::kBF16T>(p, a, splits, static_cast<int>(count));
  };
  const int64_t dp = std::min(plan.dp_tiles, tiles);
  if (dp > 0) {
    const cudaError_t err = launch(0, dp, 1);
    if (err != cudaSuccess) return err;
  }
  return dp < tiles ? launch(dp, tiles - dp, plan.splits) : cudaSuccess;
}

#define NF12_CONFIG(BN, BM, WN, WM, ST, LOOP, EFF)                                                   \
  ConfigEntry {                                                                                    \
    #LOOP "_" #BN "x" #BM "_w" #WN "x" #WM "_s" #ST, BN, BM, EFF,                                  \
        &tile_launch<Tile<BN, BM, WN, WM, ST>, Loop::k##LOOP>, &tile_smem<Tile<BN, BM, WN, WM, ST>, Loop::k##LOOP> \
  }

const ConfigEntry kConfigs[] = {
    NF12_CONFIG(128, 128, 4, 2, 2, Staged, 0.82f),  // 0: 8 warps of 32x64
    NF12_CONFIG(128, 128, 8, 2, 2, Staged, 0.78f),  // 1: 16 warps of 16x64
    NF12_CONFIG(128, 64, 4, 2, 2, Staged, 0.62f),   // 2: 8 warps of 32x32
    NF12_CONFIG(64, 128, 2, 4, 2, Staged, 0.59f),   // 3: 8 warps of 32x32
    NF12_CONFIG(128, 256, 2, 4, 2, Staged, 0.88f),  // 4: 8 warps of 64x64
    NF12_CONFIG(256, 128, 4, 2, 2, Staged, 0.86f),  // 5: 8 warps of 64x64
    NF12_CONFIG(128, 192, 2, 4, 2, Staged, 0.84f),  // 6: 8 warps of 64x48
    NF12_CONFIG(128, 128, 4, 2, 3, Stream, 0.72f),  // 7: 8 warps of 32x64
    NF12_CONFIG(128, 64, 4, 2, 4, Stream, 0.57f),   // 8: 8 warps of 32x32
    NF12_CONFIG(128, 64, 4, 1, 4, Stream, 0.53f),   // 9: 4 warps of 32x64
    NF12_CONFIG(64, 128, 2, 2, 4, Stream, 0.53f),   // 10: 4 warps of 32x64
    NF12_CONFIG(64, 64, 2, 2, 6, Stream, 0.41f),    // 11: 4 warps of 32x32
    NF12_CONFIG(128, 32, 4, 2, 4, Stream, 0.30f),   // 12: 8 warps of 32x16
    NF12_CONFIG(256, 128, 4, 2, 2, Budget, 0.0f),  // 13: SM86/SM89 models supply its efficiency
    NF12_CONFIG(64, 32, 2, 2, 2, Staged, 0.0f),    // 14: compact SM120 route, 24 KiB shared memory
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
constexpr int kNumHeuristicConfigs = 14;  // The compact tile is selected by the outer dispatch.
static_assert(kNumConfigs <= 16, "forced config encoding reserves four bits for the tile");
constexpr int kBudget256x128 = 13;

// Per-architecture model fits. The staged 128x256 and 256x128 tiles spill
// below SM90, while the register-budgeted 256x128 variant avoids those spills.
// Efficiencies follow kConfigs order and use machine_of()'s mma.sync peak.
struct ArchModel {
  int sm_arch;
  float efficiency[kNumConfigs];
  double wave_us, launch_us, reduce_us;
  double partial_bytes;  // DRAM bytes per FP32 split partial (write + read, plus contention)
};
constexpr ArchModel kArchModels[] = {
    {80, {0.535f, 0.44f, 0.469f, 0.39f, 0.596f, 0.58f, 0.556f, 0.39f, 0.28f, 0.43f, 0.30f, 0.20f, 0.24f, 0.0f},
     2.0, 25.0, 8.0, 8.0},  // CMP 170HX
    {86, {0.55f, 0.653f, 0.294f, 0.428f, 0.50f, 0.51f, 0.599f, 0.52f, 0.42f, 0.40f, 0.41f, 0.217f, 0.252f, 0.685f},
     0.0, 5.0, 2.0, 16.0},  // RTX A6000
    {89, {0.364f, 0.31f, 0.175f, 0.325f, 0.29f, 0.32f, 0.34f, 0.27f, 0.246f, 0.23f, 0.24f, 0.133f, 0.23f, 0.72f},
     8.0, 15.0, 4.0, 8.0},  // RTX 6000 Ada
};

// Every other arch: kConfigs' efficiencies and the SM120 fit (see choose()).
constexpr ArchModel kDefaultModel{0, {}, 4.0, 15.0, 8.0, 8.0};

const ArchModel& arch_model(const DeviceInfo& dev) {
  for (const ArchModel& a : kArchModels)
    if (a.sm_arch == dev.sm_arch) return a;
  return kDefaultModel;
}

float efficiency(int config, const ArchModel& am) {
  return am.sm_arch != 0 ? am.efficiency[config] : kConfigs[config].efficiency;
}

struct Choice {
  int config;
  Plan plan;
};

// Per-SM mma.sync BF16 throughput (FLOP/us) and DRAM bandwidth share
// (bytes/us). Only their ratio and the efficiencies above matter, so
// untuned architectures use rough public figures.
struct Machine {
  double flops_per_us, bytes_per_us;
};
Machine machine_of(const DeviceInfo& dev) {
  switch (dev.sm_arch) {
    case 120: return {2.32e6, 8.2e3};    // RTX PRO 6000: 437 TF/s, 1.54 TB/s over 188 SMs (measured)
    case 110: return {3.2e6, 13.5e3};    // Thor: 64 TF/s, 270 GB/s over 20 SMs (measured)
    case 100: return {7.5e6, 54.0e3};
    case 90: return {5.0e6, 25.0e3};
    case 89: return {2.72e6, 6.3e3};     // RTX 6000 Ada: 386 TF/s, 891 GB/s over 142 SMs (measured)
    case 86: return {1.96e6, 8.5e3};     // RTX A6000: 165 TF/s, 712 GB/s over 84 SMs (measured)
    case 80: return {2.98e6, 23.1e3};    // CMP 170HX: 221 TF/s, 1.71 TB/s over 74 SMs (measured)
    default: return {2.0e6, 12.0e3};
  }
}

// Both sources of a config must fit, so that the heuristic never picks a
// config whose bf16t twin cannot run.
bool fits(const ConfigEntry& e, bool split, const DeviceInfo& dev) {
  const bool cluster = split && cluster_splits(dev);
  return static_cast<size_t>(std::max(e.smem(WeightSource::kNF12, cluster), e.smem(WeightSource::kBF16T, cluster))) <=
         dev.smem_optin_bytes;
}

// Model each launch as the slower of tensor work over whole CTA waves and
// operand streaming, plus split-K and extra-launch costs. Power-of-two
// splits pack clusters into GPCs better than odd sizes. The choice is
// source-independent so NF12 and bf16t use the same arithmetic.
Choice choose(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  // Keep the SM86 outer route in sync: staged 128x64 with two K splits
  // beats small-M here for both real weights and escape-heavy weights.
  if (dev.sm_arch == 86 && m > 32 && m <= 64 && k > 24576 && k <= 25600 && n >= 4352 && n <= 4608)
    return Choice{2, Plan{2, 8, 0}};
  const Machine mc = machine_of(dev);
  const int64_t sms = std::max(dev.num_sms, 1);
  const int64_t kblocks = k / 64;
  const double bytes = 1.5 * n * k + 2.0 * m * k + 2.0 * m * n;
  const bool cluster = cluster_splits(dev);
  // Stream loops can saturate bandwidth with fewer active SMs. Other loops
  // scale bandwidth by SM activity. Cluster splits pay co-scheduling and
  // DSMEM costs, while SM8x splits pay a workspace round trip and reduce launch.
  const ArchModel& am = arch_model(dev);
  Choice best{0, Plan{1, 8, 0}};
  double best_t = 1e30;
  for (int c = 0; c < kNumHeuristicConfigs; ++c) {
    const ConfigEntry& e = kConfigs[c];
    // A 192-token tile avoids kBudget's second M tile at 129..192 rows.
    const bool budget = c == kBudget256x128;
    if (budget && ((dev.sm_arch != 86 && dev.sm_arch != 89) || m <= 64 || (m > 128 && m <= 192))) continue;
    if (!fits(e, false, dev)) continue;
    // 256-token tiles measured far below the model below M = 512.
    if (e.bm >= 256 && m < 512) continue;
    const bool stream = std::string_view(e.name).rfind("Stream", 0) == 0;
    const int64_t tiles = ((n + e.bn - 1) / e.bn) * ((m + e.bm - 1) / e.bm);
    const double stage = 2.0 * e.bn * e.bm * 64 / (mc.flops_per_us * efficiency(c, am));
    // One launch over `count` tiles split s ways.
    auto launch_us = [&](int64_t count, int s) {
      const int64_t ctas = count * s, waves = (ctas + sms - 1) / sms;
      const double t_mma = waves * (((kblocks + s - 1) / s) * stage + am.wave_us);
      // Express the wider budget tile's bandwidth activity in 128-row tiles.
      const double active = std::min<double>(1.0, static_cast<double>(ctas) * (budget ? 2.0 : 1.0) / sms);
      const double t_mem = bytes * count / tiles / (mc.bytes_per_us * sms * (stream ? 1.0 : 0.55 + 0.45 * active));
      double split_us = 0.0;
      if (s > 1 && cluster) split_us = waves * (4.0 + 4.0 * (s - 1)) * (e.bn * e.bm) / (128.0 * 256.0);
      if (s > 1 && !cluster) split_us = am.partial_bytes * ctas * e.bn * e.bm / (mc.bytes_per_us * sms) + am.reduce_us;
      return std::max(t_mma, t_mem) + split_us;
    };
    auto consider = [&](const Plan& plan, double t) {
      if (t < best_t) {
        best_t = t;
        best = Choice{c, plan};
      }
    };
    for (int s = 1; s <= 8; s *= 2) {
      if (s > 1 && (kblocks / s < 4 || !fits(e, true, dev))) break;
      // Avoid splitting every tile when clusters require a two-pass reduction.
      if (s == 1 || !cluster || e.bn * e.bm * 4 <= 96 * 1024) consider(Plan{s, 8, 0}, launch_us(tiles, s));
      // Whole waves data-parallel, the remainder (plus one more wave) split.
      if (s > 1) {
        for (int64_t f = tiles / sms; f >= std::max<int64_t>(1, tiles / sms - 1); --f) {
          const int64_t rest = tiles - f * sms;
          if (rest > 0) consider(Plan{s, 8, f * sms}, launch_us(f * sms, 1) + launch_us(rest, s) + am.launch_us);
        }
      }
    }
  }
  return best;
}

// Resolve heuristic or forced configs, clamping splits to K / 64.
// A result config >= kNumConfigs marks an invalid forced config.
Choice resolve(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config) {
  Choice c = choose(m, n, k, dev);
  if (config >= 0) {
    c.config = config % 16;
    c.plan.splits = (config / 16) % 16 + 1;
    c.plan.dp_tiles = config / 256;
  }
  c.plan.splits = static_cast<int>(std::min<int64_t>(c.plan.splits, k / 64));
  return c;
}

// Tests/tuning encoding: table index + 16 * (splits - 1) + 256 * dp_tiles.
int encode(const Choice& c) {
  return c.config + 16 * (c.plan.splits - 1) + 256 * static_cast<int>(std::min<int64_t>(c.plan.dp_tiles, 1 << 22));
}

}  // namespace

int large_m_mma_num_configs() { return kNumConfigs; }

const char* large_m_mma_config_name(int config) {
  return config >= 0 && config < kNumConfigs ? kConfigs[config].name : "invalid";
}

int large_m_mma_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  return encode(choose(m, n, k, dev));
}

size_t linear_large_m_mma_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config) {
  const Choice c = resolve(m, n, k, dev, config);
  if (c.config >= kNumConfigs) return 0;
  return plan_workspace_bytes(kConfigs[c.config].bn, kConfigs[c.config].bm, c.plan, m, n, dev);
}

cudaError_t linear_large_m_mma(const LinearArgs& args, int config) {
  if (args.dev->sm_arch < 80) return cudaErrorNotSupported;
  const Choice c = resolve(args.m, args.n, args.k, *args.dev, config);
  if (c.config >= kNumConfigs) return cudaErrorInvalidValue;
  if (c.config == kBudget256x128 && args.dev->sm_arch >= 90) return cudaErrorNotSupported;
  if (c.plan.splits > 8) return cudaErrorNotSupported;
  const ConfigEntry& e = kConfigs[c.config];
  if (static_cast<size_t>(e.smem(args.source, c.plan.splits > 1 && cluster_splits(*args.dev))) >
      args.dev->smem_optin_bytes)
    return cudaErrorNotSupported;
  return e.launch(args, c.plan);
}

}  // namespace nf12::detail
