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
// Cluster-reduced narrow-weight MMA for SM90, SM100/103 and SM120/121.
// The grid tiles tokens and splits K over `splits` CTAs, each holding at
// most kChunk k-blocks. Each warp owns one 16-row weight block and issues
// all its super-atom loads before decoding. The CTA stages x with cp.async.
// Each rank publishes FP32 partials in distributed shared memory, then sums
// an equal output slice in ascending rank order before bias and BF16 stores.
// Both sources use the same arithmetic order for a given config.
// No workspace is needed.
// PDL ordering is documented in linear/dispatch.cu.
#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>

#include "linear/common.cuh"
#include "linear/sm90/large_m_narrow_cluster.h"

namespace nf12::detail {
namespace {

constexpr int kMaxDevices = 64;
constexpr int kMaxSplits = 8;  // portable cluster size
// SM90 also runs non-portable clusters of up to 16 CTAs (K up to 8192).
constexpr int kMaxSplitsSm90 = 16;
constexpr int kChunk = 8;      // k-blocks per CTA, all in flight at once
constexpr size_t kMaxSmem = 8 * 8 * 32 * 16 + kChunk * 64 * 128;  // partials + x tile at W = 8, 64 tokens

struct NarrowParams {
  const __nv_bfloat16* x;
  int64_t ldx;
  const uint8_t* w;
  const uint4* escapes;
  uint32_t base_x4;
  const __nv_bfloat16* bias;
  __nv_bfloat16* y;
  int64_t ldy;
  int m, nb, kb, splits;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float4 ld_cluster(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(remote)
               : "memory");
  return v;
}
__device__ __forceinline__ void cp_async_16_zfill(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t saddr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(saddr));
}

// Stores one C fragment's (row, row + 8) x (tok, tok + 1) values + bias.
__device__ __forceinline__ void store_frag(const NarrowParams& p, int row, int tok, float4 v) {
  float b_lo = 0.f, b_hi = 0.f;
  if (p.bias != nullptr) {
    b_lo = __bfloat162float(p.bias[row]);
    b_hi = __bfloat162float(p.bias[row + 8]);
  }
  __nv_bfloat16* y = p.y + static_cast<int64_t>(tok) * p.ldy + row;
  if (tok < p.m) {
    y[0] = __float2bfloat16_rn(v.x + b_lo);
    y[8] = __float2bfloat16_rn(v.z + b_hi);
  }
  if (tok + 1 < p.m) {
    y[p.ldy] = __float2bfloat16_rn(v.y + b_lo);
    y[p.ldy + 8] = __float2bfloat16_rn(v.w + b_hi);
  }
}
#endif

template <WeightSource S, int TF, int W>
__global__ void __launch_bounds__(32 * W) narrow_kernel(const __grid_constant__ NarrowParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using T = WeightTraits<S>;
  constexpr int TT = 8 * TF;  // tokens per CTA
  // [W][TF][32] C fragments (split clusters only), then the x tile:
  // [k-block][token][64 bf16], 16-byte chunks XOR-swizzled by token.
  extern __shared__ float4 part[];
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x * W + warp;
  const int tok0 = blockIdx.y * TT;
  const int s = p.splits > 1 ? static_cast<int>(cluster_rank()) : 0;
  const int kb0 = s * p.kb / p.splits, nkb = (s + 1) * p.kb / p.splits - kb0;  // nkb <= kChunk
  const uint32_t xs = smem_addr(part) + (p.splits > 1 ? W * TF * 32 * 16 : 0);

  // Issue all weight loads before consuming any to overlap their DRAM latency.
  uint4 wv[kChunk][T::kLoads];
  if (rb < p.nb) {
    const uint8_t* wp = p.w + (static_cast<int64_t>(rb) * p.kb + kb0) * T::kSuperAtomBytes + lane * 16;
#pragma unroll
    for (int i = 0; i < kChunk; ++i)
      if (i < nkb)
#pragma unroll
        for (int l = 0; l < T::kLoads; ++l) wv[i][l] = ldg_stream(wp + i * T::kSuperAtomBytes + l * kPlaneBytes);
  }
  // The CTA's x tile (tokens past m zero-filled), shared by its warps.
  for (int idx = static_cast<int>(threadIdx.x); idx < nkb * TT * 8; idx += 32 * W) {
    const int kbi = idx / (TT * 8), tok = idx / 8 % TT, ch = idx % 8;
    const bool in = tok0 + tok < p.m;
    const __nv_bfloat16* src = p.x + static_cast<int64_t>(in ? tok0 + tok : 0) * p.ldx + (kb0 + kbi) * 64 + ch * 8;
    cp_async_16_zfill(xs + ((kbi * TT + tok) * 8 + (ch ^ (tok & 7))) * 16, src, in ? 16u : 0u);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  float acc[TF][4];
#pragma unroll
  for (int f = 0; f < TF; ++f)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[f][c] = 0.f;
  if (rb < p.nb) {
    // ldmatrix matrix q = lane / 8 selects token block f + q / 2 and
    // k half q % 2. Its row is lane % 8.
    const int q = lane >> 3, r = lane & 7;
#pragma unroll
    for (int i = 0; i < kChunk; ++i) {
      if (i < nkb) {
        uint32_t b[TF][4][2];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (TF == 1) {
            const int tok = r, ch = 2 * j + (q & 1);
            uint32_t v[2];
            ldmatrix_x2(v, xs + ((i * TT + tok) * 8 + (ch ^ (tok & 7))) * 16);
            b[0][j][0] = v[0];
            b[0][j][1] = v[1];
          } else {
#pragma unroll
            for (int f = 0; f < TF; f += 2) {
              const int tok = (f + (q >> 1)) * 8 + r, ch = 2 * j + (q & 1);
              uint32_t v[4];
              ldmatrix_x4(v, xs + ((i * TT + tok) * 8 + (ch ^ (tok & 7))) * 16);
              b[f][j][0] = v[0];
              b[f][j][1] = v[1];
              b[f + 1][j][0] = v[2];
              b[f + 1][j][1] = v[3];
            }
          }
        }
        uint32_t a[4][4];
        T::fragments(wv[i], p.base_x4, p.escapes, a);
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int f = 0; f < TF; ++f) mma_bf16_16816(acc[f], a[j], b[f][j][0], b[f][j][1]);
      }
    }
  }

  if (p.splits == 1) {
    if (rb < p.nb)
#pragma unroll
      for (int f = 0; f < TF; ++f)
        store_frag(p, rb * 16 + g, tok0 + f * 8 + 2 * t, make_float4(acc[f][0], acc[f][1], acc[f][2], acc[f][3]));
    return;
  }
#pragma unroll
  for (int f = 0; f < TF; ++f) part[(warp * TF + f) * 32 + lane] = make_float4(acc[f][0], acc[f][1], acc[f][2], acc[f][3]);
  cluster_sync();
  constexpr int kFrags = W * TF * 32;
  const int i1 = (s + 1) * kFrags / p.splits;
  for (int i = s * kFrags / p.splits + static_cast<int>(threadIdx.x); i < i1; i += 32 * W) {
    const uint32_t addr = smem_addr(part + i);
    float4 v = ld_cluster(addr, 0);
    for (int r = 1; r < p.splits; ++r) {
      const float4 q = ld_cluster(addr, r);
      v.x += q.x;
      v.y += q.y;
      v.z += q.z;
      v.w += q.w;
    }
    const int ln = i % 32, f = i / 32 % TF, wr = i / (32 * TF);
    const int row_block = blockIdx.x * W + wr;
    if (row_block < p.nb) store_frag(p, row_block * 16 + ln / 4, tok0 + f * 8 + 2 * (ln % 4), v);
  }
  cluster_sync();  // no rank exits while others still read its tile
#else
  (void)p;
  __trap();  // never launched: narrow_sm120_supported() requires an SM90+ image
#endif
}

// Configs: token fragments per warp TF = 1 << tf, warps per CTA W = 1 << wi.
using KernelFn = void (*)(NarrowParams);

template <WeightSource S, int TF>
KernelFn kernel_w(int wi) {
  switch (wi) {
    case 0: return narrow_kernel<S, TF, 1>;
    case 1: return narrow_kernel<S, TF, 2>;
    case 2: return narrow_kernel<S, TF, 4>;
    default: return narrow_kernel<S, TF, 8>;
  }
}

template <WeightSource S>
KernelFn kernel_for(int tf, int wi) {
  switch (tf) {
    case 0: return kernel_w<S, 1>(wi);
    case 1: return kernel_w<S, 2>(wi);
    case 2: return kernel_w<S, 4>(wi);
    default: return kernel_w<S, 8>(wi);
  }
}

struct Config {
  int tf, wi, splits;
};

Config decode_config(int config) { return {config & 3, (config >> 2) & 3, config >> 4}; }

}  // namespace

bool narrow_sm120_supported(const DeviceInfo& dev) {
  // nvcc 13.2 generates slower SM110 code for these router gates, so Thor
  // retains the small-M, tcgen05 and decode-GEMM routes.
  if (dev.sm_arch != 90 && dev.sm_arch != 100 && dev.sm_arch != 103 && dev.sm_arch != 120 && dev.sm_arch != 121)
    return false;
  static std::atomic<int8_t> cache[kMaxDevices] = {};  // 0 unknown, 1 yes, -1 no
  const bool cacheable = dev.device >= 0 && dev.device < kMaxDevices;
  if (cacheable) {
    const int8_t c = cache[dev.device].load(std::memory_order_relaxed);
    if (c != 0) return c > 0;
  }
  // The loaded image must hold an SM90+ body (the cluster reduction).
  cudaFuncAttributes attr{};
  bool ok = false;
  if (cudaFuncGetAttributes(&attr, narrow_kernel<WeightSource::kNF12, 1, 1>) == cudaSuccess)
    ok = attr.binaryVersion >= 90;
  else
    cudaGetLastError();
  if (cacheable) cache[dev.device].store(ok ? 1 : -1, std::memory_order_relaxed);
  return ok;
}

// SM90 uses non-portable clusters for K beyond 64 k-blocks. Wider token
// tiles and CTAs amortize x staging as M grows, balanced against grid coverage.
int narrow_sm120_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  const int64_t kb = k / 64;
  int splits = static_cast<int>(std::min<int64_t>(kMaxSplits, kb));
  if (dev.sm_arch == 90 && kb > kMaxSplits * kChunk) {
    splits = m <= 32 ? static_cast<int>((kb + kChunk - 1) / kChunk) : kMaxSplitsSm90;
    const int sz = n <= 128 ? 0 : n <= 256 ? 1 : 2;  // weight width class
    int tf, wi;
    if (m <= 8) {
      tf = 0, wi = sz < 2 ? 1 : 2;
    } else if (m <= 16) {
      tf = 1, wi = 1 + sz;
    } else if (m <= 32) {
      tf = sz == 1 ? 2 : 1, wi = sz == 0 ? 2 : sz == 1 ? 2 : 3;
    } else if (m <= 64) {
      tf = 2, wi = sz == 0 ? 2 : 3;
    } else {
      tf = m <= 128 && sz == 0 ? 2 : 3, wi = 3;
    }
    return tf | (wi << 2) | (splits << 4);
  }
  if (dev.sm_arch == 90) {
    const int tf = m <= 8 ? 0 : m <= 32 ? 1 : m <= 64 ? 2 : m <= 128 ? 1 : 2;
    const int wi = m <= 16 ? 1 : m <= 64 ? 2 : m <= 256 ? 3 : 2;
    return tf | (wi << 2) | (splits << 4);
  }
  if (dev.sm_arch == 100 || dev.sm_arch == 103) {
    // Four splits amortize reduction at large M and short K. Very narrow
    // weights need eight splits again to retain enough parallel work.
    int tf, wi;
    if (m <= 8) {
      tf = 0, wi = k <= 2048 ? 2 : 1;
    } else if (m <= 32) {
      tf = k <= 2048 && (n <= 64 || m <= 16) ? 0 : 1;
      wi = k > 2048 ? 1 : n <= 64 ? 3 : m <= 16 ? 0 : 2;
    } else if (m <= 128) {
      tf = 1, wi = n <= 64 || m > 64 ? 3 : 2;
    } else if (m <= 512) {
      tf = 2, wi = m <= 256 ? 3 : 2;
      if (m > 256 && k <= 2048 && n <= 64) {
        tf = 1, wi = 3;
        splits = std::min(splits, 4);
      }
    } else if (k <= 2048 && n <= 64 && m > 1024) {
      tf = 3, wi = 2;
    } else {
      tf = k <= 2048 && m <= 1024 ? 2 : 3, wi = k <= 2048 ? 3 : 2;
      if (k <= 2048) splits = std::min(splits, 4);
    }
    return tf | (wi << 2) | (splits << 4);
  }
  const int tf = m <= 64 ? 0 : 1;
  const int wi = m <= 32 ? 0 : m <= 128 ? 1 : 3;
  return tf | (wi << 2) | (splits << 4);
}

cudaError_t linear_narrow_sm120(const LinearArgs& a, int config) {
  if (a.dev == nullptr || !narrow_sm120_supported(*a.dev)) return cudaErrorNotSupported;
  if (a.m < 1 || a.m > INT32_MAX || a.n > INT32_MAX) return cudaErrorInvalidValue;
  const Config c = decode_config(config < 0 ? narrow_sm120_heuristic(a.m, a.n, a.k, *a.dev) : config);
  const int kb = static_cast<int>(a.k / 64);
  const int max_splits = a.dev->sm_arch == 90 ? kMaxSplitsSm90 : kMaxSplits;
  if (c.splits < 1 || c.splits > max_splits || c.splits > kb || (kb + c.splits - 1) / c.splits > kChunk)
    return cudaErrorNotSupported;
  const int tf_tokens = 8 << c.tf, warps = 1 << c.wi;
  const int64_t m_tiles = (a.m + tf_tokens - 1) / tf_tokens;
  if (m_tiles > 65535) return cudaErrorNotSupported;
  const KernelFn fn = a.source == WeightSource::kNF12 ? kernel_for<WeightSource::kNF12>(c.tf, c.wi)
                                                      : kernel_for<WeightSource::kBF16T>(c.tf, c.wi);
  const size_t smem = (c.splits > 1 ? static_cast<size_t>(warps) * (tf_tokens / 8) * 32 * 16 : 0) +
                      static_cast<size_t>((kb + c.splits - 1) / c.splits) * tf_tokens * 128;
  if (smem > a.dev->smem_optin_bytes) return cudaErrorNotSupported;
  if (smem > 48 * 1024) {
    // Opt in once per device and kernel to the largest window any config takes.
    static std::atomic<bool> granted[kMaxDevices][32];
    const int dev = a.dev->device;
    const int idx = (a.source == WeightSource::kNF12 ? 0 : 16) + c.tf * 4 + c.wi;
    if (dev < 0 || dev >= kMaxDevices || !granted[dev][idx].load(std::memory_order_acquire)) {
      const int bytes = static_cast<int>(std::min<size_t>(a.dev->smem_optin_bytes, kMaxSmem));
      if (const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
          e != cudaSuccess)
        return e;
      if (dev >= 0 && dev < kMaxDevices) granted[dev][idx].store(true, std::memory_order_release);
    }
  }
  if (c.splits > kMaxSplits) {
    // Once per device and kernel: clusters past the portable size.
    static std::atomic<bool> granted[kMaxDevices][32];
    const int dev = a.dev->device;
    const int idx = (a.source == WeightSource::kNF12 ? 0 : 16) + c.tf * 4 + c.wi;
    if (dev < 0 || dev >= kMaxDevices || !granted[dev][idx].load(std::memory_order_acquire)) {
      if (const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
          e != cudaSuccess)
        return e;
      if (dev >= 0 && dev < kMaxDevices) granted[dev][idx].store(true, std::memory_order_release);
    }
  }
  NarrowParams p{};
  p.x = a.x;
  p.ldx = a.ldx;
  p.w = a.weight;
  p.escapes = a.source == WeightSource::kNF12 ? a.escapes : nullptr;
  p.base_x4 = exp_base_x4(a.exp_base);
  p.bias = a.bias;
  p.y = a.y;
  p.ldy = a.ldy;
  p.m = static_cast<int>(a.m);
  p.nb = static_cast<int>(a.n / 16);
  p.kb = kb;
  p.splits = c.splits;

  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(static_cast<unsigned>((p.nb + warps - 1) / warps), static_cast<unsigned>(m_tiles),
                    static_cast<unsigned>(c.splits));
  lc.blockDim = dim3(32 * warps);
  lc.dynamicSmemBytes = smem;
  lc.stream = a.stream;
  cudaLaunchAttribute attr[2];
  lc.attrs = attr;
  if (c.splits > 1) {
    attr[lc.numAttrs].id = cudaLaunchAttributeClusterDimension;
    attr[lc.numAttrs].val.clusterDim.x = 1;
    attr[lc.numAttrs].val.clusterDim.y = 1;
    attr[lc.numAttrs].val.clusterDim.z = static_cast<unsigned>(c.splits);
    ++lc.numAttrs;
  }
  attr[lc.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[lc.numAttrs].val.programmaticStreamSerializationAllowed = 1;
  ++lc.numAttrs;
  return cudaLaunchKernelEx(&lc, fn, p);
}

}  // namespace nf12::detail
