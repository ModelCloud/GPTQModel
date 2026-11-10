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
// Small-M linear (m <= 64): memory-bound streaming kernel.
//
// A CTA owns `WARPS / KP` consecutive 16-row blocks over a contiguous K
// range. Its warps share one staged x copy. The `kc` CTAs of each row group
// split K, reducing through cluster shared memory on SM90+. On SM8x they
// publish FP32 workspace partials for a second kernel to sum in rank order.
#include <cuda_bf16.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>

#include "linear/small_m.h"
#include "linear/small_m_core.cuh"

namespace nf12::detail {
namespace {

struct SmallMParams {
  const __nv_bfloat16* x;
  int64_t ldx;
  const uint8_t* w;
  const uint4* escapes;
  uint32_t base_x4;
  int m;
  int nb;  // 16-row blocks
  int kb;  // super-atoms per row block
  int slice;
  int xstride;  // bytes per staged x row
  int kc;       // K split; blockIdx.x = row group * kc + rank (SM90+: cluster size)
  const __nv_bfloat16* bias;
  __nv_bfloat16* y;
  int64_t ldy;
  float* partials;  // below SM90 with kc > 1: FP32 partials [kc][m][nb * 16]
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
constexpr bool kDeviceHasClusters = true;
#else
constexpr bool kDeviceHasClusters = false;
#endif

template <WeightSource S, int MT, int WARPS, int kStages, int KP, bool Retain = false>
__global__ void __launch_bounds__(WARPS * 32) small_m_kernel(const SmallMParams p) {
  // SM8x-only K parts and 16-warp CTAs have no code in SM90+ images and are
  // rejected by small_m_config_supported().
  if constexpr ((KP > 1 || WARPS > 8) && kDeviceHasClusters) {
    __trap();
  } else {
  using T = WeightTraits<S>;
  constexpr int RW = WARPS / KP;
  // The weight ring (kStages > 0) first, then x and the partials.
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int64_t kRingBytes = small_m_ring_bytes(kStages, WARPS, T::kSuperAtomBytes);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // One flat grid dimension: gridDim.y would cap N at 65535 row groups.
  const int kc = p.kc;
  const int rank = static_cast<int>(blockIdx.x) % kc;
  const int group = static_cast<int>(blockIdx.x) / kc;
  const int kb0 = static_cast<int>(static_cast<int64_t>(rank) * p.kb / kc);
  const int cnt = static_cast<int>(static_cast<int64_t>(rank + 1) * p.kb / kc) - kb0;
  // Tail warps re-read the last row block to keep the pipeline uniform.
  // Their results are not stored.
  const int kpart = KP == 1 ? 0 : warp / RW;
  const int nb = min(group * RW + (KP == 1 ? warp : warp % RW), p.nb - 1);
  const uint8_t* wp = p.w + (static_cast<size_t>(nb) * p.kb + kb0 + kpart) * T::kSuperAtomBytes + lane * 16;

  // PDL may overlap prologues, but all global loads must wait for the prior
  // kernel, which may still be writing the weights (dispatch.cu contract).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const __nv_bfloat16* x0 = p.x + static_cast<size_t>(kb0) * 64;
  const int n0 = group * RW * 16;
  // Below SM90 the kc ranks of a row group are independent CTAs: each
  // finalizes all of its outputs (as a 1-rank cluster) into the workspace.
  const int core_kc = kDeviceHasClusters ? kc : 1;
  const int core_rank = kDeviceHasClusters ? rank : 0;
  small_m_core<S, MT, WARPS, kStages, KP, Retain>(
      smem + kRingBytes, smem, wp, cnt, p.base_x4, p.escapes, p.m, p.slice, p.xstride, core_kc, core_rank,
      [&](int r) { return x0 + r * p.ldx; }, [] {},
      [&](int tok, int rp, float2 s) {
        const int n = n0 + 2 * rp;
        if (n >= p.nb * 16) return;
        if constexpr (!kDeviceHasClusters) {
          if (kc > 1) {
            *reinterpret_cast<float2*>(p.partials + (static_cast<size_t>(rank) * p.m + tok) * (p.nb * 16) + n) = s;
            return;
          }
        }
        if (p.bias != nullptr) {
          const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p.bias + n));
          s.x += b.x;
          s.y += b.y;
        }
        *reinterpret_cast<__nv_bfloat162*>(p.y + tok * p.ldy + n) = __floats2bfloat162_rn(s.x, s.y);
      });
  }
}

// Below SM90, after small_m_kernel with kc > 1: y = bf16(partials of ranks
// 0..kc-1 summed in rank order + bias), four columns per thread. SM90+
// images carry no code for it.
__global__ void small_m_split_reduce_kernel(const float* __restrict__ partials, int kc, int m, int n,
                                            const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* y, int64_t ldy) {
  if constexpr (kDeviceHasClusters) {
    __trap();
  } else {
    const int quads = n / 4;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t{m} * quads) return;
    const int tok = static_cast<int>(i / quads);
    const int c = static_cast<int>(i % quads) * 4;
    const size_t rank_stride = static_cast<size_t>(m) * n;
    const float* src = partials + static_cast<size_t>(tok) * n + c;
    float4 s = *reinterpret_cast<const float4*>(src);
    for (int r = 1; r < kc; ++r) {
      const float4 v = *reinterpret_cast<const float4*>(src + r * rank_stride);
      s.x += v.x;
      s.y += v.y;
      s.z += v.z;
      s.w += v.w;
    }
    if (bias != nullptr) {
      const float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + c));
      const float2 b1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + c + 2));
      s.x += b0.x;
      s.y += b0.y;
      s.z += b1.x;
      s.w += b1.y;
    }
    __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(y + tok * ldy + c);
    dst[0] = __floats2bfloat162_rn(s.x, s.y);
    dst[1] = __floats2bfloat162_rn(s.z, s.w);
  }
}

struct Plan {
  int slice;
  int xstride;
  size_t smem;  // dynamic shared memory, the kernel's whole footprint
};

// x slices hold whole steps of every K part.
int slice_step(const SmallMConfig& cfg, int stages) { return small_m_step(stages) * cfg.kp; }

Plan make_plan(const SmallMConfig& cfg, int64_t m, int64_t k, int stages, WeightSource source) {
  const int kb = static_cast<int>(k / 64);
  const int cnt_max = (kb + cfg.kc - 1) / cfg.kc;
  const auto round_up = [](int64_t v, int64_t to) { return (v + to - 1) / to * to; };
  Plan pl{};
  pl.slice = (cfg.slice <= 0 || cfg.slice >= cnt_max) ? static_cast<int>(round_up(cnt_max, slice_step(cfg, stages)))
                                                      : cfg.slice;
  const bool two_slots = pl.slice < cnt_max;
  const int64_t xstride = int64_t{pl.slice} * 128 + 16;
  pl.xstride = static_cast<int>(std::min<int64_t>(xstride, INT32_MAX));
  const int64_t xbytes = (two_slots ? 2 : 1) * m * xstride;
  const int64_t pbytes = cfg.kp * m * small_m_partial_stride(cfg.warps / cfg.kp) * int64_t{sizeof(float)};
  const int64_t region = round_up(std::max(xbytes, pbytes), 16);
  const int super_atom = source == WeightSource::kNF12 ? kSuperAtomBytes : kSuperAtomBytesBf16t;
  pl.smem = static_cast<size_t>(small_m_ring_bytes(stages, cfg.warps, super_atom) + region);
  return pl;
}

using KernelFn = void (*)(SmallMParams);

// 4- and 8-warp CTAs; below SM90 also K parts (kp > 1, register ring only:
// 16-warp CTAs of 8 / 4 / 2 row blocks, 8-warp of 4 / 2 / 1, 4-warp of 2 / 1)
// and 16-warp CTAs of 16 row blocks.
template <WeightSource S, int MT, int kStages, bool Retain = false>
KernelFn kernel_warps(int warps, int kp) {
  if (kp == 1) {
    switch (warps) {
      case 4: return small_m_kernel<S, MT, 4, kStages, 1, Retain>;
      case 8: return small_m_kernel<S, MT, 8, kStages, 1, Retain>;
      default: break;
    }
  }
  if constexpr (kStages == 0) {
    switch (warps * 16 + kp) {
      case 4 * 16 + 2: return small_m_kernel<S, MT, 4, 0, 2, Retain>;
      case 4 * 16 + 4: return small_m_kernel<S, MT, 4, 0, 4, Retain>;
      case 8 * 16 + 2: return small_m_kernel<S, MT, 8, 0, 2, Retain>;
      case 8 * 16 + 4: return small_m_kernel<S, MT, 8, 0, 4, Retain>;
      case 8 * 16 + 8: return small_m_kernel<S, MT, 8, 0, 8, Retain>;
      case 16 * 16 + 1: return small_m_kernel<S, MT, 16, 0, 1, Retain>;
      case 16 * 16 + 2: return small_m_kernel<S, MT, 16, 0, 2, Retain>;
      case 16 * 16 + 4: return small_m_kernel<S, MT, 16, 0, 4, Retain>;
      case 16 * 16 + 8: return small_m_kernel<S, MT, 16, 0, 8, Retain>;
      default: return nullptr;
    }
  }
  return nullptr;
}

template <WeightSource S, int kStages, bool Retain = false>
KernelFn kernel_mt(const SmallMConfig& cfg) {
  switch (cfg.mt) {
    case 1: return kernel_warps<S, 1, kStages, Retain>(cfg.warps, cfg.kp);
    case 2: return kernel_warps<S, 2, kStages, Retain>(cfg.warps, cfg.kp);
    case 4: return kernel_warps<S, 4, kStages, Retain>(cfg.warps, cfg.kp);
    case 8: return kernel_warps<S, 8, kStages, Retain>(cfg.warps, cfg.kp);
    default: return nullptr;
  }
}

template <WeightSource S>
KernelFn kernel_stages(const SmallMConfig& cfg, int stages) {
  switch (stages) {
    case 0: return kernel_mt<S, 0>(cfg);
    case kSmallMBulkStages: return kernel_mt<S, kSmallMBulkStages>(cfg);
    default: return nullptr;
  }
}

KernelFn kernel_for(WeightSource s, const SmallMConfig& cfg, int stages, bool retain = false) {
  if (retain && stages == 0)
    return s == WeightSource::kNF12 ? kernel_mt<WeightSource::kNF12, 0, true>(cfg)
                                    : kernel_mt<WeightSource::kBF16T, 0, true>(cfg);
  return s == WeightSource::kNF12 ? kernel_stages<WeightSource::kNF12>(cfg, stages)
                                  : kernel_stages<WeightSource::kBF16T>(cfg, stages);
}

bool retain_weights(int64_t n, int64_t k, const DeviceInfo& dev) {
  // Leave headroom for activations and neighboring allocations in Ada's L2.
  // Use the BF16 footprint for both twins, independent of escape contents.
  return dev.sm_arch == 89 && n > 0 && k > 0 &&
         static_cast<uint64_t>(n) <= dev.l2_bytes / 4 / static_cast<uint64_t>(k);
}

}  // namespace

// Host-side kernel state, memoized per (kernel, device): the dynamic shared
// memory cap is raised once (idempotent, not a stream operation, so legal
// during graph capture) and occupancy is cached per shared-memory size.
// Assumes `device` is the current device, as for the launch itself.
namespace {
std::mutex g_mu;
std::set<std::pair<const void*, int>> g_raised;
std::map<std::tuple<const void*, int, size_t>, int> g_occupancy;
}  // namespace

cudaError_t small_m_prepare_kernel(const void* fn, const DeviceInfo& dev, size_t smem) {
  if (smem <= 48 * 1024) return cudaSuccess;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_raised.count({fn, dev.device})) return cudaSuccess;
  const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(dev.smem_optin_bytes));
  if (e == cudaSuccess) g_raised.insert({fn, dev.device});
  return e;
}

int small_m_kernel_occupancy(const void* fn, const DeviceInfo& dev, int threads, size_t smem) {
  if (small_m_prepare_kernel(fn, dev, smem) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  const auto key = std::make_tuple(fn, dev.device, smem);
  const auto it = g_occupancy.find(key);
  if (it != g_occupancy.end()) return it->second;
  int n = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, threads, smem) != cudaSuccess) {
    cudaGetLastError();
    n = 0;
  }
  g_occupancy.emplace(key, n);
  return n;
}

// SM90 and SM100 support the shared-memory ring. SM90 chooses per shape.
int small_m_bulk_stages(const DeviceInfo& dev) {
  return dev.sm_arch == 100 || dev.sm_arch == 90 ? kSmallMBulkStages : 0;
}

// With enough row groups, MT = 1 can fill SM90 without the bulk ring.
int small_m_stages(const SmallMConfig& cfg, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (dev.sm_arch == 90 && cfg.mt == 1 && n >= std::max<int64_t>(k, 8192)) return 0;
  return small_m_bulk_stages(dev);
}

bool small_m_config_supported(const SmallMConfig& cfg, int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (m < 1 || m > 8 * cfg.mt || m > kSmallMMaxM) return false;
  const int stages = small_m_stages(cfg, n, k, dev);
  if (kernel_for(WeightSource::kNF12, cfg, stages) == nullptr) return false;
  if (cfg.kc < 1 || cfg.kc > 8 || cfg.kc > k / 64) return false;
  if ((cfg.kp > 1 || cfg.warps > 8) && dev.sm_arch >= 90) return false;
  if (cfg.slice < 0 || cfg.slice % slice_step(cfg, stages) != 0) return false;
  // The bf16t twin runs the same config with larger super-atoms.
  return make_plan(cfg, m, k, stages, WeightSource::kBF16T).smem <= dev.smem_optin_bytes;
}

size_t small_m_workspace_bytes(const SmallMConfig& cfg, int64_t m, int64_t n, const DeviceInfo& dev) {
  return cfg.kc > 1 && dev.sm_arch < 90 ? static_cast<size_t>(cfg.kc) * m * n * sizeof(float) : 0;
}

int small_m_resident_ctas(const SmallMConfig& cfg, int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  const int stages = small_m_stages(cfg, n, k, dev);
  const KernelFn fn = kernel_for(WeightSource::kNF12, cfg, stages, retain_weights(n, k, dev));
  if (fn == nullptr) return 0;
  const Plan pl = make_plan(cfg, m, k, stages, WeightSource::kNF12);
  return pl.smem > dev.smem_optin_bytes
             ? 0
             : small_m_kernel_occupancy(reinterpret_cast<const void*>(fn), dev, cfg.warps * 32, pl.smem);
}

cudaError_t linear_small_m(const LinearArgs& a, const SmallMConfig& cfg) {
  const int stages = small_m_stages(cfg, a.n, a.k, *a.dev);
  const KernelFn fn = kernel_for(a.source, cfg, stages, retain_weights(a.n, a.k, *a.dev));
  if (fn == nullptr) return cudaErrorInvalidValue;
  const Plan pl = make_plan(cfg, a.m, a.k, stages, a.source);
  if (pl.smem > a.dev->smem_optin_bytes) return cudaErrorInvalidValue;
  const size_t ws = small_m_workspace_bytes(cfg, a.m, a.n, *a.dev);
  if (ws > 0 && (a.workspace == nullptr || a.workspace_bytes < ws)) return cudaErrorInvalidValue;
  const cudaError_t e = small_m_prepare_kernel(reinterpret_cast<const void*>(fn), *a.dev, pl.smem);
  if (e != cudaSuccess) return e;

  SmallMParams p{};
  p.x = a.x;
  p.ldx = a.ldx;
  p.w = a.weight;
  p.escapes = a.escapes;
  p.base_x4 = exp_base_x4(a.exp_base);
  p.m = static_cast<int>(a.m);
  p.nb = static_cast<int>(a.n / 16);
  p.kb = static_cast<int>(a.k / 64);
  p.slice = pl.slice;
  p.xstride = pl.xstride;
  p.kc = cfg.kc;
  p.bias = a.bias;
  p.y = a.y;
  p.ldy = a.ldy;
  p.partials = ws > 0 ? static_cast<float*>(a.workspace) : nullptr;

  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>(cfg.kc * ((p.nb + cfg.warps / cfg.kp - 1) / (cfg.warps / cfg.kp))), 1, 1);
  lc.blockDim = dim3(cfg.warps * 32, 1, 1);
  lc.dynamicSmemBytes = pl.smem;
  lc.stream = a.stream;
  cudaLaunchAttribute attr[2];
  lc.attrs = attr;
  if (cfg.kc > 1 && a.dev->sm_arch >= 90) {
    attr[lc.numAttrs].id = cudaLaunchAttributeClusterDimension;
    attr[lc.numAttrs].val.clusterDim.x = cfg.kc;
    attr[lc.numAttrs].val.clusterDim.y = 1;
    attr[lc.numAttrs].val.clusterDim.z = 1;
    ++lc.numAttrs;
  }
  // Larger token tiles lose more to PDL overhead than they save in overlap.
  if (a.dev->sm_arch >= 90 && cfg.mt <= 2) {
    attr[lc.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[lc.numAttrs].val.programmaticStreamSerializationAllowed = 1;
    ++lc.numAttrs;
  }
  const cudaError_t err = cudaLaunchKernelEx(&lc, fn, p);
  if (err != cudaSuccess || ws == 0) return err;
  cudaLaunchConfig_t rc{};
  rc.gridDim = dim3(static_cast<unsigned>((a.m * (a.n / 4) + 255) / 256), 1, 1);
  rc.blockDim = dim3(256, 1, 1);
  rc.stream = a.stream;
  return cudaLaunchKernelEx(&rc, small_m_split_reduce_kernel, static_cast<const float*>(p.partials), cfg.kc, p.m,
                            static_cast<int>(a.n), a.bias, a.y, a.ldy);
}

// Model time in super-atoms streamed at the saturated device rate. Each wave
// pays pipeline ramp/drain and ragged-tail costs. K splitting adds restaging
// and reduction costs. SM8x also accounts for x traffic, slice barriers,
// workspace reduction and the register ring's dependent-load latency.
namespace {

struct SmallMModel {
  double saturation;
  double wave_cost;   // super-atoms per SM, scaled by (1 + m / 8)
  double tail;
  double coverage;
  double split_cost;  // per token and extra cluster rank or K part: x restaging + reduction
  double cta_cost;    // super-atoms per CTA
  double locality;    // DRAM page locality: a CTA's kp parts read kp adjacent super-atoms per step
  double x_cost = 0.0;       // per staged x byte, relative to a weight byte (below SM90)
  double slice_cost = 0.0;   // super-atoms per SM per x slice of a wave (below SM90)
  double reduce_cost = 0.0;  // super-atoms per SM for the reduce launch of kc > 1 (below SM90)
  double warp_cost = 0.0;    // super-atoms per CTA and warp: start-up, x staging and partials sums (below SM90)
  // Below SM90: a wave lasts at least chain_cost super-atoms per SM for each
  // super-atom a warp streams in sequence (DRAM latency of the register ring).
  double chain_cost = 0.0;
};

// Per-architecture fits use exhaustive config sweeps with each config's
// heuristic slice and selected weight ring. Replay with
// tests/kernels/linear/tune_small_m.py and tests/kernels/linear/bench_small_m.py --sweep.
constexpr SmallMModel kDefaultModel{0.75, 4.0, 0.1, 1.0, 0.4, 0.0, 0.0};
constexpr SmallMModel kSm80Model{4.0, 16.0, 0.0, 0.5, 0.0, 16.0, 0.0, 0.5, 8.0, 80.0, 0.0, 0.0};
constexpr SmallMModel kSm86Model{2.0, 4.0, 0.1, 1.0, 0.0, 4.0, 0.8, 1.5, 1.0, 0.0, 0.0, 16.0};
constexpr SmallMModel kSm89Model{6.0, 0.0, 0.0, 0.85, 0.0, 16.0, 0.2, 2.0, 0.25, 50.0, 4.0, 32.0};
constexpr SmallMModel kSm90Model{2.25, 0.5, 0.0, 0.75, 0.1, 6.0, 0.0};
constexpr SmallMModel kSm90RegisterModel{3.0, 0.0, 0.0, 0.5, 0.05, 0.0, 0.0};
constexpr SmallMModel kSm100Model{3.0, 16.0, 0.0, 1.0, 0.1, 16.0, 0.0};

const SmallMModel& small_m_model(const DeviceInfo& dev, int stages) {
  switch (dev.sm_arch) {
    case 80: return kSm80Model;
    case 86: return kSm86Model;
    case 89: return kSm89Model;
    case 90: return stages > 0 ? kSm90Model : kSm90RegisterModel;
    case 100: return kSm100Model;
    default: return kDefaultModel;
  }
}

// Ignores cfg.slice and returns -1 for an unsupported config. SM90+ uses
// whole-range staging when small, otherwise minimum double-buffered slices
// to preserve residency. SM8x chooses the longest slice that preserves the
// minimum slice's residency to amortize barriers and restaging latency.
int heuristic_slice(SmallMConfig cfg, int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  const int stages = small_m_stages(cfg, n, k, dev);
  const int step = slice_step(cfg, stages);
  const int64_t cnt = (k / 64 + cfg.kc - 1) / cfg.kc;
  if (dev.sm_arch >= 90) {
    cfg.slice = m * (cnt * 128 + 16) <= 8 * 1024 ? 0 : step;
    return small_m_resident_ctas(cfg, m, n, k, dev) > 0 ? cfg.slice : -1;
  }
  cfg.slice = step;
  const int occ = small_m_resident_ctas(cfg, m, n, k, dev);
  if (occ <= 0) return -1;
  cfg.slice = 0;
  if (small_m_resident_ctas(cfg, m, n, k, dev) >= occ) return 0;
  int best = step;
  for (int64_t s = 2 * int64_t{step}; s < cnt; s *= 2) {
    cfg.slice = static_cast<int>(s);
    if (small_m_resident_ctas(cfg, m, n, k, dev) < occ) break;
    best = cfg.slice;
  }
  return best;
}

SmallMConfig choose_small_m(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  SmallMConfig best{};
  double best_t = 0;
  const int64_t kb = k / 64;
  const int64_t nb = n / 16;
  const bool sm8x = dev.sm_arch < 90;
  // K splits: across CTAs (kc; clusters from SM90, the workspace below) and,
  // below SM90, inside the CTA (kp).
  const int max_kc = static_cast<int>(std::min<int64_t>(8, kb));
  const int max_kp = sm8x ? 8 : 1;
  for (const int warps : {16, 8, 4}) {
    // Avoid 16-warp CTAs when there are fewer than 16 row blocks.
    if (warps == 16 && (!sm8x || nb < 16)) continue;
    for (int kp = 1; kp <= std::min(max_kp, warps); kp *= 2) {
    for (int kc = 1; kc <= max_kc; ++kc) {
      SmallMConfig cfg{};
      cfg.mt = m <= 8 ? 1 : m <= 16 ? 2 : m <= 32 ? 4 : 8;
      cfg.warps = warps;
      cfg.kc = kc;
      cfg.kp = kp;
      const int stages = small_m_stages(cfg, n, k, dev);
      const SmallMModel& mod = small_m_model(dev, stages);
      const int64_t cnt = (kb + kc - 1) / kc;
      const int rows = warps / kp;  // row blocks per CTA
      // At 64 tokens, four-warp CTAs stage as many x bytes as weight bytes.
      // Restrict them to short, latency-bound K ranges.
      if (warps == 4 && cfg.mt == 8 && cnt / kp > 16) continue;
      cfg.slice = heuristic_slice(cfg, m, n, k, dev);
      if (cfg.slice < 0) continue;
      const int occ = small_m_resident_ctas(cfg, m, n, k, dev);
      if (occ <= 0) continue;
      const int64_t capacity = int64_t{occ} * dev.num_sms;
      const int64_t total = (nb + rows - 1) / rows * kc;
      const double per_cta = static_cast<double>(nb * kb) / total;
      const int64_t slices = cfg.slice == 0 ? 1 : (cnt + cfg.slice - 1) / cfg.slice;
      // Warps that stream: K parts past a CTA's cnt super-atoms idle.
      const double warps_busy = warps * std::min(1.0, static_cast<double>(cnt) / kp);
      double t = 0;
      for (int64_t rem = total; rem > 0;) {
        const int64_t w = std::min(rem, capacity);
        // Bandwidth requires both enough active warps and SM coverage.
        // Long-K 64-token tiles issue 32 MMAs per super-atom per warp, so
        // they can saturate with lower SM coverage.
        const double coverage = (cfg.mt == 8 && cnt > 16) ? 0.85 * mod.coverage : mod.coverage;
        const double rate = std::min({1.0, w * warps_busy / 8.0 / (mod.saturation * dev.num_sms),
                                      std::min<int64_t>(w, dev.num_sms) / (coverage * dev.num_sms)});
        const double stream = w * per_cta / rate * (1.0 + mod.locality / kp);
        t += std::max(stream, mod.chain_cost * ((cnt + kp - 1) / kp) * dev.num_sms) +
             mod.wave_cost * (1.0 + m / 8.0) * dev.num_sms;
        t += mod.split_cost * m * (kc * kp - 1) * dev.num_sms;
        t += mod.slice_cost * slices * dev.num_sms;
        rem -= w;
      }
      if (total > capacity) t += mod.tail * per_cta * capacity;
      t += (mod.cta_cost + mod.warp_cost * warps) * total;
      t += mod.x_cost * total * m * cnt * 128.0 / kSuperAtomBytes;
      // Workspace split: the reduce launch, partials written and read back.
      if (sm8x && kc > 1) t += mod.reduce_cost * dev.num_sms + 8.0 * kc * m * n / kSuperAtomBytes;
      if (best.mt == 0 || t < best_t) {
        best = cfg;
        best_t = t;
      }
    }
    }
  }
  return best;
}

std::mutex g_choice_mu;
std::map<std::tuple<int64_t, int64_t, int64_t, int>, SmallMConfig> g_choices;

}  // namespace

// Memoized per (m, n, k, device): the SM8x search spans ~90 configs and
// their slice choices.
SmallMConfig small_m_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  // The 124-SM Ampere compressor decode benefits from five K ranks and eight
  // in-CTA K parts; the generic SM80 model was fitted on a 74-SM card.
  if (dev.sm_arch == 80 && dev.num_sms >= 120 && m == 6 && n == 512 && k == 5120) {
    SmallMConfig tuned{1, 16, 5, 0, 8};
    tuned.slice = heuristic_slice(tuned, m, n, k, dev);
    if (tuned.slice >= 0 && small_m_config_supported(tuned, m, n, k, dev)) return tuned;
  }
  const auto key = std::make_tuple(m, n, k, dev.device);
  {
    std::lock_guard<std::mutex> lock(g_choice_mu);
    const auto it = g_choices.find(key);
    if (it != g_choices.end()) return it->second;
  }
  const SmallMConfig c = choose_small_m(m, n, k, dev);
  std::lock_guard<std::mutex> lock(g_choice_mu);
  g_choices.emplace(key, c);
  return c;
}

int small_m_heuristic_slice(const SmallMConfig& cfg, int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  return heuristic_slice(cfg, m, n, k, dev);
}

}  // namespace nf12::detail
