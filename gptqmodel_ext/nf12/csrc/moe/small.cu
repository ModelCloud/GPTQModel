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
// Work items are expert-major, then row group, then m-tile, so consecutive
// m-tiles reuse weights in L2. Counts remain device-side. The host grid bounds
// the work, and CTAs beyond the scanned item count exit before cluster barriers.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>

#include "linear/small_m.h"
#include "linear/small_m_core.cuh"
#include "moe/common.cuh"
#include "moe/schedule.cuh"
#include "moe/small.h"

namespace nf12::detail {
namespace {

struct MoeSmallParams {
  const __nv_bfloat16* x;
  int64_t ldx;
  const int32_t* gather;
  int64_t gather_div;
  const int32_t* scatter;
  const int64_t* expert_offsets;
  int64_t max_assignments;
  const uint8_t* w;
  int64_t expert_bytes;
  const uint4* escapes;
  const int64_t* escape_offsets;
  const uint8_t* exp_base;
  const float* row_scale;
  __nv_bfloat16* y;
  int64_t ldy;
  int num_experts;
  int nb;      // 16-row blocks per expert
  int kb;      // super-atoms per row block
  int groups;  // row groups (WARPS row blocks) per expert
  int slice;
  int xstride;      // bytes per staged x row
  int kc;           // cluster K split, with blockIdx.x = item * kc + rank
  int table_bytes;  // offset of the per-row table past the weight ring
};

// The row table follows streaming storage so x staging cannot overwrite it.
constexpr int kRowTableBytes = 8 + 8 + 4;

// Bound register use to retain the target CTA residency.
constexpr int moe_small_threads_per_sm(int mt) { return mt == 1 ? 1024 : mt == 2 ? 768 : mt == 4 ? 512 : 256; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
constexpr bool kDeviceHasClusters = true;
#else
constexpr bool kDeviceHasClusters = false;
#endif

template <WeightSource S, int MT, int WARPS, int kStages, int KP>
__global__ void __launch_bounds__(WARPS * 32, moe_small_threads_per_sm(MT) / (WARPS * 32))
    moe_small_kernel(const MoeSmallParams p) {
  // K parts substitute for cluster K splits below SM90.
  if constexpr (KP > 1 && kDeviceHasClusters) {
    __trap();
  } else {
  using T = WeightTraits<S>;
  constexpr int kBm = 8 * MT;
  constexpr int RW = WARPS / KP;  // row blocks per CTA
  // The weight ring precedes overlapping schedule, x and partial-result storage.
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* const region = smem + small_m_ring_bytes(kStages, WARPS, T::kSuperAtomBytes);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int kc = p.kc;
  // Routing-independent arithmetic may precede the wait. The launch bounds
  // blockIdx.x and rank * kb to int32.
  const int rank = static_cast<int>(blockIdx.x % static_cast<unsigned>(kc));
  const int item = static_cast<int>(blockIdx.x / static_cast<unsigned>(kc));
  const int mtile = item / p.groups;
  const int kb0 = rank * p.kb / kc;
  const int cnt = (rank + 1) * p.kb / kc - kb0;
  // Repeat the last row block without storing it to keep the pipeline uniform.
  const int kpart = KP == 1 ? 0 : warp / RW;

  // No global reads, including expert_offsets, may precede the PDL wait.
  // Decode prefetch returns no data and may overlap the wait and scan.
  moe_prefetch_codec<S>(p.exp_base, p.escapes, p.escape_offsets, p.num_experts, threadIdx.x, WARPS * 32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  int* sched = reinterpret_cast<int*>(region);
  const int total = moe_mtile_prefix(p.expert_offsets, p.num_experts, kBm, p.max_assignments, sched);
  // All CTAs of a cluster share `item`, so they leave together, before any cluster barrier.
  if (mtile >= total) return;
  const int e = moe_mtile_expert_block(sched, p.num_experts, mtile);
  const int local = item - p.groups * sched[e];
  const int ntiles = sched[e + 1] - sched[e];
  const int group = local / ntiles;
  int row0, m;
  moe_mtile_rows(sched, p.num_experts, e, local - group * ntiles, kBm, row0, m);
  // Defer row-map arithmetic until prep() to overlap the first weight loads.
  int32_t in_idx = 0, out_idx = 0;
  if (static_cast<int>(threadIdx.x) < m) {
    const int pos = row0 + static_cast<int>(threadIdx.x);
    in_idx = p.gather != nullptr ? __ldg(p.gather + pos) : pos;
    out_idx = p.scatter != nullptr ? __ldg(p.scatter + pos) : pos;
  }

  const int nb = min(group * RW + (KP == 1 ? warp : warp % RW), p.nb - 1);
  const uint8_t* wp = p.w + static_cast<size_t>(e) * p.expert_bytes +
                      (static_cast<size_t>(nb) * p.kb + kb0 + kpart) * T::kSuperAtomBytes + lane * 16;
  const MoeExpertCodec codec = moe_expert_codec<S>(p.exp_base, p.escapes, p.escape_offsets, e);

  const auto** xrows = reinterpret_cast<const __nv_bfloat16**>(region + p.table_bytes);
  auto** yrows = reinterpret_cast<__nv_bfloat16**>(region + p.table_bytes + kBm * 8);
  auto* scales = reinterpret_cast<float*>(region + p.table_bytes + kBm * 16);
  const int n0 = group * RW * 16;
  small_m_core<S, MT, WARPS, kStages, KP>(
      region, smem, wp, cnt, codec.base_x4, codec.escapes, m, p.slice, p.xstride, kc, rank,
      [&](int r) { return xrows[r]; },
      [&] {
        // The table lives past the region x staging overwrites, and the
        // barrier also retires every read of the schedule scratch.
        const int r = threadIdx.x;
        if (r < m) {
          const int64_t in = p.gather != nullptr ? in_idx / p.gather_div : in_idx;
          xrows[r] = p.x + in * p.ldx + static_cast<size_t>(kb0) * 64;
          yrows[r] = p.y + int64_t{out_idx} * p.ldy;
          if (p.row_scale != nullptr) scales[r] = __ldg(p.row_scale + out_idx);
        }
        __syncthreads();
      },
      [&](int tok, int rp, float2 s) {
        const int n = n0 + 2 * rp;
        if (n >= p.nb * 16) return;
        if (p.row_scale != nullptr) {
          s.x *= scales[tok];
          s.y *= scales[tok];
        }
        *reinterpret_cast<__nv_bfloat162*>(yrows[tok] + n) = __floats2bfloat162_rn(s.x, s.y);
      });
  }
}

struct Plan {
  int slice;
  int xstride;
  int table_bytes;
  size_t smem;  // dynamic shared memory, the kernel's whole footprint
};

// SM90 uses the register ring rather than the bulk shared-memory ring.
int moe_small_stages(const DeviceInfo& dev) { return dev.sm_arch == 90 ? 0 : small_m_bulk_stages(dev); }

// x slices hold whole steps of every K part.
int slice_step(const MoeSmallConfig& cfg, int stages) { return small_m_step(stages) * cfg.kp; }

Plan make_plan(const MoeSmallConfig& cfg, int64_t num_experts, int64_t k, int stages, WeightSource source) {
  const int bm = 8 * cfg.mt;
  const int kb = static_cast<int>(k / 64);
  const int cnt_max = (kb + cfg.kc - 1) / cfg.kc;
  const auto round_up = [](int64_t v, int64_t to) { return (v + to - 1) / to * to; };
  Plan pl{};
  pl.slice = static_cast<int>((cfg.slice <= 0 || cfg.slice >= cnt_max) ? round_up(cnt_max, slice_step(cfg, stages))
                                                                         : cfg.slice);
  const bool two_slots = pl.slice < cnt_max;
  const int64_t xstride = int64_t{pl.slice} * 128 + 16;
  pl.xstride = static_cast<int>(std::min<int64_t>(xstride, INT32_MAX));
  const int64_t xbytes = (two_slots ? 2 : 1) * bm * xstride;
  const int64_t pbytes = int64_t{cfg.kp} * bm * small_m_partial_stride(cfg.warps / cfg.kp) * sizeof(float);
  const int64_t sbytes = int64_t{moe_schedule_smem_ints(static_cast<int>(num_experts))} * sizeof(int);
  const int64_t table = round_up(std::max({xbytes, pbytes, sbytes}), 16);
  pl.table_bytes = static_cast<int>(std::min<int64_t>(table, INT32_MAX));
  const int super_atom = source == WeightSource::kNF12 ? kSuperAtomBytes : kSuperAtomBytesBf16t;
  pl.smem = static_cast<size_t>(small_m_ring_bytes(stages, cfg.warps, super_atom) + table + bm * kRowTableBytes);
  return pl;
}

using KernelFn = void (*)(MoeSmallParams);

// K parts require the register ring.
template <WeightSource S, int MT, int kStages>
KernelFn kernel_warps(int warps, int kp) {
  if (kp == 1) {
    switch (warps) {
      case 4: return moe_small_kernel<S, MT, 4, kStages, 1>;
      case 8: return moe_small_kernel<S, MT, 8, kStages, 1>;
      default: return nullptr;
    }
  }
  if constexpr (kStages == 0) {
    switch (warps * 16 + kp) {
      case 4 * 16 + 2: return moe_small_kernel<S, MT, 4, 0, 2>;
      case 4 * 16 + 4: return moe_small_kernel<S, MT, 4, 0, 4>;
      case 8 * 16 + 2: return moe_small_kernel<S, MT, 8, 0, 2>;
      case 8 * 16 + 4: return moe_small_kernel<S, MT, 8, 0, 4>;
      case 8 * 16 + 8: return moe_small_kernel<S, MT, 8, 0, 8>;
      default: return nullptr;
    }
  }
  return nullptr;
}

template <WeightSource S, int kStages>
KernelFn kernel_mt(const MoeSmallConfig& cfg) {
  switch (cfg.mt) {
    case 1: return kernel_warps<S, 1, kStages>(cfg.warps, cfg.kp);
    case 2: return kernel_warps<S, 2, kStages>(cfg.warps, cfg.kp);
    case 4: return kernel_warps<S, 4, kStages>(cfg.warps, cfg.kp);
    case 8: return kernel_warps<S, 8, kStages>(cfg.warps, cfg.kp);
    default: return nullptr;
  }
}

template <WeightSource S>
KernelFn kernel_stages(const MoeSmallConfig& cfg, int stages) {
  switch (stages) {
    case 0: return kernel_mt<S, 0>(cfg);
    case kSmallMBulkStages: return kernel_mt<S, kSmallMBulkStages>(cfg);
    default: return nullptr;
  }
}

KernelFn kernel_for(WeightSource s, const MoeSmallConfig& cfg, int stages) {
  return s == WeightSource::kNF12 ? kernel_stages<WeightSource::kNF12>(cfg, stages)
                                  : kernel_stages<WeightSource::kBF16T>(cfg, stages);
}

int resident_ctas(const MoeSmallConfig& cfg, int64_t num_experts, int64_t k, const DeviceInfo& dev) {
  const int stages = moe_small_stages(dev);
  const KernelFn fn = kernel_for(WeightSource::kNF12, cfg, stages);
  if (fn == nullptr) return 0;
  const Plan pl = make_plan(cfg, num_experts, k, stages, WeightSource::kNF12);
  return pl.smem > dev.smem_optin_bytes
             ? 0
             : small_m_kernel_occupancy(reinterpret_cast<const void*>(fn), dev, cfg.warps * 32, pl.smem);
}

// Clusters must fit wholly inside a GPC, so occupancy * SMs / kc can overcount.
int64_t resident_clusters(const MoeSmallConfig& cfg, int64_t num_experts, int64_t k, const DeviceInfo& dev) {
  const int occ = resident_ctas(cfg, num_experts, k, dev);
  if (occ <= 0) return 0;
  if (cfg.kc == 1) return int64_t{occ} * dev.num_sms;
  const int stages = moe_small_stages(dev);
  const KernelFn fn = kernel_for(WeightSource::kNF12, cfg, stages);
  const Plan pl = make_plan(cfg, num_experts, k, stages, WeightSource::kNF12);
  static std::mutex mu;
  static std::map<std::tuple<KernelFn, int, size_t, int>, int64_t> cache;
  const auto key = std::make_tuple(fn, dev.device, pl.smem, cfg.kc);
  std::lock_guard<std::mutex> lock(mu);
  if (const auto it = cache.find(key); it != cache.end()) return it->second;
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>(cfg.kc), 1, 1);
  lc.blockDim = dim3(cfg.warps * 32, 1, 1);
  lc.dynamicSmemBytes = pl.smem;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cfg.kc;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  lc.attrs = &attr;
  lc.numAttrs = 1;
  int clusters = 0;
  if (cudaOccupancyMaxActiveClusters(&clusters, fn, &lc) != cudaSuccess) {
    cudaGetLastError();  // clear the non-sticky occupancy error
    clusters = occ * dev.num_sms / cfg.kc;
  }
  cache.emplace(key, clusters);
  return clusters;
}

}  // namespace

bool moe_small_config_supported(const MoeSmallConfig& cfg, int64_t num_experts, int64_t n, int64_t k,
                                const DeviceInfo& dev) {
  const int stages = moe_small_stages(dev);
  if (kernel_for(WeightSource::kNF12, cfg, stages) == nullptr) return false;
  if (num_experts < 1 || num_experts > kMoeMaxExperts || n < 16 || k < 64) return false;
  if (cfg.kc < 1 || cfg.kc > 8 || (cfg.kc > 1 && dev.sm_arch < 90) || cfg.kc > k / 64) return false;
  if (cfg.kp > 1 && dev.sm_arch >= 90) return false;
  if (cfg.slice < 0 || cfg.slice % slice_step(cfg, stages) != 0) return false;
  if (cfg.pdl != 0 && (cfg.pdl != 1 || dev.sm_arch < 90)) return false;
  // The bf16t twin runs the same config with larger super-atoms.
  return make_plan(cfg, num_experts, k, stages, WeightSource::kBF16T).smem <= dev.smem_optin_bytes;
}

cudaError_t moe_linear_small(const MoeLinearArgs& a, const MoeSmallConfig& cfg) {
  const int stages = moe_small_stages(*a.dev);
  const KernelFn fn = kernel_for(a.source, cfg, stages);
  if (fn == nullptr) return cudaErrorInvalidValue;
  const Plan pl = make_plan(cfg, a.num_experts, a.k, stages, a.source);
  if (pl.smem > a.dev->smem_optin_bytes) return cudaErrorInvalidValue;
  const int nb = static_cast<int>(a.n / 16);
  const int groups = (nb + cfg.warps / cfg.kp - 1) / (cfg.warps / cfg.kp);
  const int64_t clusters = moe_max_mtiles(a.max_assignments, a.num_experts, 8 * cfg.mt) * groups;
  if (clusters * cfg.kc > INT32_MAX) return cudaErrorNotSupported;
  const cudaError_t err = small_m_prepare_kernel(reinterpret_cast<const void*>(fn), *a.dev, pl.smem);
  if (err != cudaSuccess) return err;

  MoeSmallParams p{};
  p.x = a.x;
  p.ldx = a.ldx;
  p.gather = a.gather;
  p.gather_div = a.gather_div;
  p.scatter = a.scatter;
  p.expert_offsets = a.expert_offsets;
  p.max_assignments = a.max_assignments;
  p.w = a.weight;
  p.expert_bytes = moe_expert_bytes(a.source, a.n, a.k);
  p.escapes = a.escapes;
  p.escape_offsets = a.escape_offsets;
  p.exp_base = a.exp_base;
  p.row_scale = a.row_scale;
  p.y = a.y;
  p.ldy = a.ldy;
  p.num_experts = static_cast<int>(a.num_experts);
  p.nb = nb;
  p.kb = static_cast<int>(a.k / 64);
  p.groups = groups;
  p.slice = pl.slice;
  p.xstride = pl.xstride;
  p.kc = cfg.kc;
  p.table_bytes = pl.table_bytes;

  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>(clusters * cfg.kc), 1, 1);
  lc.blockDim = dim3(cfg.warps * 32, 1, 1);
  lc.dynamicSmemBytes = pl.smem;
  lc.stream = a.stream;
  cudaLaunchAttribute attr[2];
  lc.attrs = attr;
  if (cfg.kc > 1) {
    attr[lc.numAttrs].id = cudaLaunchAttributeClusterDimension;
    attr[lc.numAttrs].val.clusterDim.x = cfg.kc;
    attr[lc.numAttrs].val.clusterDim.y = 1;
    attr[lc.numAttrs].val.clusterDim.z = 1;
    ++lc.numAttrs;
  }
  if (cfg.pdl) {
    attr[lc.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[lc.numAttrs].val.programmaticStreamSerializationAllowed = 1;
    ++lc.numAttrs;
  }
  return cudaLaunchKernelEx(&lc, fn, p);
}

// Uniform routing predicts E * (1 - (1 - 1/E)^A) active experts. The model
// charges a fixed cost per CTA for the scan, gather and cluster reduction.
// SM90 and SM100 instead fit K splits within one resident wave, with at least
// four super-atoms per CTA. SM8x splits K inside a CTA because it has no clusters.
namespace {

MoeSmallConfig moe_small_heuristic_sm8x(double active, double rows, int64_t num_experts, int64_t n, int64_t k,
                                        const DeviceInfo& dev) {
  const int64_t kb = k / 64;
  constexpr int kWarps = 8;
  const bool sm80 = dev.sm_arch == 80;
  MoeSmallConfig cfg{};
  cfg.mt = sm80 && rows <= 4 ? 1 : rows <= 12 ? 2 : rows <= (dev.sm_arch == 89 ? 28 : 40) ? 4 : 8;
  cfg.warps = kWarps;
  cfg.kc = 1;
  int max_kp = 1;
  if (sm80) {
    const int64_t ctas = static_cast<int64_t>(std::ceil(active)) * ((n / 16 + kWarps - 1) / kWarps);
    max_kp = ctas < 2 * int64_t{dev.num_sms} ? 2 : 1;
  } else {
    max_kp = rows <= 12 ? 4 : rows <= 24 ? 2 : 1;
  }
  int kp = 1;
  while (2 * kp <= max_kp && kb >= 16 * kp) kp *= 2;  // >= 8 super-atoms per part
  const int min_slice = sm80 || dev.sm_arch == 89 ? 8 : 4;
  // Fewer K parts, then shorter slices, where shared memory does not fit.
  for (; kp >= 1; kp /= 2) {
    cfg.kp = kp;
    for (int slice = std::max(min_slice, kp * small_m_step(0)); slice >= kp * small_m_step(0); slice /= 2) {
      cfg.slice = slice;
      if (moe_small_config_supported(cfg, num_experts, n, k, dev)) return cfg;
    }
  }
  return MoeSmallConfig{};
}

MoeSmallConfig moe_small_heuristic_sm90(double active, double rows, int64_t num_experts, int64_t n, int64_t k,
                                        const DeviceInfo& dev) {
  const int64_t kb = k / 64;
  const int mt = rows <= 6 ? 1 : rows <= 12 ? 2 : rows <= 24 ? 4 : 8;
  constexpr int kWarps = 8;
  const int64_t ctas_per_split = static_cast<int64_t>(std::ceil(active)) * ((n / 16 + kWarps - 1) / kWarps);
  MoeSmallConfig best{};
  for (int kc = 1; kc <= std::min<int64_t>(8, kb); ++kc) {
    const int64_t cnt = (kb + kc - 1) / kc;
    if (kc > 1 && cnt < 4) break;
    MoeSmallConfig cfg{};
    cfg.mt = mt;
    cfg.warps = kWarps;
    cfg.kc = kc;
    cfg.pdl = mt <= 2 ? 1 : 0;
    cfg.slice = 8 * mt * (cnt * 128 + 16) <= 16 * 1024 ? 0 : 2 * small_m_step(moe_small_stages(dev));
    if (!moe_small_config_supported(cfg, num_experts, n, k, dev)) continue;
    const int64_t capacity = resident_clusters(cfg, num_experts, k, dev);
    if (capacity <= 0) continue;
    if (best.mt == 0 || ctas_per_split <= capacity) best = cfg;
  }
  return best;
}

MoeSmallConfig moe_small_heuristic_sm100(double active, int64_t num_experts, int64_t n, int64_t k,
                                         const DeviceInfo& dev) {
  const int64_t kb = k / 64;
  const int64_t ctas_per_split = static_cast<int64_t>(std::ceil(active)) * ((n / 16 + 7) / 8);
  MoeSmallConfig best{};
  int64_t best_capacity = 0;
  for (int kc = 1; kc <= std::min<int64_t>(8, kb); ++kc) {
    if (kc > 1 && (kb + kc - 1) / kc < 4) break;
    MoeSmallConfig cfg{};
    cfg.mt = 2;
    cfg.warps = 8;
    cfg.kc = kc;
    cfg.slice = 2 * small_m_step(moe_small_stages(dev));
    if (!moe_small_config_supported(cfg, num_experts, n, k, dev)) continue;
    const int occ = resident_ctas(cfg, num_experts, k, dev);
    if (occ <= 0) continue;
    if (best.mt == 0 || ctas_per_split * kc <= int64_t{occ} * dev.num_sms) {
      best = cfg;
      best_capacity = int64_t{occ} * dev.num_sms;
    }
  }
  best.pdl = best.mt != 0 && !(best.kc == 1 && ctas_per_split <= best_capacity) ? 1 : 0;
  return best;
}

}  // namespace

MoeSmallConfig moe_small_heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                   const DeviceInfo& dev) {
  constexpr double kSaturation = 0.75;
  constexpr double kWaveCost = 4.0;
  constexpr double kTail = 0.1;
  constexpr double kSplitCost = 0.4;
  constexpr double kCtaCost = 8.0;  // super-atoms per CTA
  const double A = static_cast<double>(std::max<int64_t>(max_assignments, 1));
  const double E = static_cast<double>(num_experts);
  const double active = E * (1.0 - std::pow(1.0 - 1.0 / E, A));
  if (dev.sm_arch == 100) return moe_small_heuristic_sm100(active, num_experts, n, k, dev);
  if (dev.sm_arch == 90) return moe_small_heuristic_sm90(active, A / active, num_experts, n, k, dev);
  if (dev.sm_arch < 90) return moe_small_heuristic_sm8x(active, A / active, num_experts, n, k, dev);
  const double rows = A / active;
  const int64_t kb = k / 64;
  const int64_t nb = n / 16;
  MoeSmallConfig best{};
  double best_t = 0;
  const int max_kc = dev.sm_arch >= 90 ? static_cast<int>(std::min<int64_t>(8, kb)) : 1;
  const int mt = rows <= 6 ? 1 : rows <= 12 ? 2 : rows <= 24 ? 4 : 8;
  for (const int warps : {8, 4}) {
    for (int kc = 1; kc <= max_kc; ++kc) {
      MoeSmallConfig cfg{};
      cfg.mt = mt;
      cfg.warps = warps;
      cfg.kc = kc;
      cfg.pdl = dev.sm_arch >= 90 && mt <= 2 ? 1 : 0;
      const int64_t cnt = (kb + kc - 1) / kc;
      const double m = std::min(rows, 8.0 * mt);
      cfg.slice = 8 * mt * (cnt * 128 + 16) <= 8 * 1024 ? 0 : small_m_step(moe_small_stages(dev));
      if (!moe_small_config_supported(cfg, num_experts, n, k, dev)) continue;
      const int occ = resident_ctas(cfg, num_experts, k, dev);
      if (occ <= 0) continue;
      const int64_t capacity = int64_t{occ} * dev.num_sms;
      const int64_t total = static_cast<int64_t>(std::ceil(active)) * ((nb + warps - 1) / warps) * kc;
      const double per_cta = active * nb * kb / total;
      double t = 0;
      for (int64_t rem = total; rem > 0;) {
        const int64_t w = std::min(rem, capacity);
        const double rate = std::min({1.0, w * warps / 8.0 / (kSaturation * dev.num_sms),
                                      std::min<int64_t>(w, dev.num_sms) / static_cast<double>(dev.num_sms)});
        t += w * per_cta / rate + kWaveCost * (1.0 + m / 8.0) * dev.num_sms;
        t += kSplitCost * m * (kc - 1) * dev.num_sms;
        rem -= w;
      }
      if (total > capacity) t += kTail * per_cta * capacity;
      t += kCtaCost * total;
      if (best.mt == 0 || t < best_t) {
        best = cfg;
        best_t = t;
      }
    }
  }
  return best;
}

}  // namespace nf12::detail
