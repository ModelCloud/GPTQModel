// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 kernel computation.
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
// modified for the NF12 method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
#include "linear/small_m_direct.h"

#include <cooperative_groups.h>

#include <cstdlib>

#include "linear/common.cuh"

namespace nf12::detail {
namespace {

// Each CTA computes 16 output columns for up to eight tokens. Warps split K.
// Each of the four atoms retains its own FP32 accumulation chain.
template <WeightSource S, int Warps, bool Cluster = false>
__global__ void __launch_bounds__(Warps * 32) small_m_direct_kernel(
    const __nv_bfloat16* x, int64_t ldx, int m, const uint8_t* weights, const uint4* escapes,
    uint32_t base, int kb, int chunk, const __nv_bfloat16* bias, __nv_bfloat16* y,
    int64_t ldy, bool weights_ready) {
  using T = WeightTraits<S, true>;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int tile = blockIdx.x;
  int begin = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (Cluster) {
    tile = cooperative_groups::grid_group::cluster_index().x;
    begin = chunk * cooperative_groups::this_cluster().block_rank();
  }
#endif
  const int count = Cluster ? max(0, min(chunk, kb - begin)) : kb;
  const uint8_t* first = weights + (static_cast<size_t>(tile) * kb + begin + warp) *
                                      T::kSuperAtomBytes + lane * 16;
  uint4 ring[2][T::kLoads];
  float acc[4][4]{};
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (!weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  // Only immutable NF12 weights may be prefetched before the dependency wait.
#pragma unroll
  for (int d = 0; d < 2; ++d) {
    if (warp + d * Warps < count) {
#pragma unroll
      for (int j = 0; j < T::kLoads; ++j)
        ring[d][j] = load_global_u4<true>(reinterpret_cast<const uint4*>(
            first + d * Warps * T::kSuperAtomBytes + j * kPlaneBytes));
    }
  }
#if __CUDA_ARCH__ >= 900
  if (weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  // Padded MMA columns reuse the last real token without reading beyond X.
  const __nv_bfloat16* input = x + min(lane >> 2, m - 1) * ldx + begin * 64 + 2 * (lane & 3);
  for (int step = 0; warp + step * Warps < count; step += 2) {
#pragma unroll
    for (int d = 0; d < 2; ++d) {
      const int p = warp + (step + d) * Warps;
      if (p < count) {
        uint32_t a[4][4];
        T::fragments(ring[d], base, escapes, a);
        if (p + 2 * Warps < count) {
#pragma unroll
          for (int j = 0; j < T::kLoads; ++j)
            ring[d][j] = load_global_u4<true>(reinterpret_cast<const uint4*>(
                first + static_cast<size_t>(step + d + 2) * Warps * T::kSuperAtomBytes + j * kPlaneBytes));
        }
#pragma unroll
        for (int atom = 0; atom < 4; ++atom) {
          // Consume each pair immediately to limit activation-register lifetimes.
          const __nv_bfloat16* address = input + p * 64 + atom * 16;
          uint32_t b0, b1;
          asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(b0) : "l"(address) : "memory");
          asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(b1) : "l"(address + 8) : "memory");
          mma_bf16_16816(acc[atom], a[atom], b0, b1);
        }
      }
    }
  }
#pragma unroll
  for (int chain = 1; chain < 4; ++chain)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[0][j] += acc[chain][j];
  // Padding keeps the lane-to-token transpose from introducing bank conflicts.
  __shared__ float partial[Warps][8][20];
  const int row = lane >> 2;
  const int token = 2 * (lane & 3);
  partial[warp][token][row] = acc[0][0];
  partial[warp][token + 1][row] = acc[0][1];
  partial[warp][token][row + 8] = acc[0][2];
  partial[warp][token + 1][row + 8] = acc[0][3];
  __syncthreads();
  for (int i = threadIdx.x; i < m * 16; i += Warps * 32) {
    const int tok = i / 16;
    const int col = i % 16;
    float sum = partial[0][tok][col];
#pragma unroll
    for (int w = 1; w < Warps; ++w) sum += partial[w][tok][col];
    if constexpr (Cluster) {
      partial[0][tok][col] = sum;
    } else {
      const int n = tile * 16 + col;
      if (bias != nullptr) sum += __bfloat162float(bias[n]);
      y[tok * ldy + n] = __float2bfloat16_rn(sum);
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (Cluster) {
    const auto cluster = cooperative_groups::this_cluster();
    cluster.sync();
    const int rank = cluster.block_rank();
    // Eight ranks divide the M * 8 output pairs into M pairs per rank.
    for (int e = rank * m + threadIdx.x; e < (rank + 1) * m; e += Warps * 32) {
      const int tok = e / 8;
      const int col = (e % 8) * 2;
      float2 values[8];
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const float* remote = cluster.map_shared_rank(&partial[0][0][0], r);
        values[r] = *reinterpret_cast<const float2*>(remote + tok * 20 + col);
      }
      float2 sum = values[0];
#pragma unroll
      for (int r = 1; r < 8; ++r) {
        sum.x += values[r].x;
        sum.y += values[r].y;
      }
      const int n = tile * 16 + col;
      if (bias != nullptr) {
        sum.x += __bfloat162float(bias[n]);
        sum.y += __bfloat162float(bias[n + 1]);
      }
      // Scalar stores also support odd output strides and offset views.
      y[tok * ldy + n] = __float2bfloat16_rn(sum.x);
      y[tok * ldy + n + 1] = __float2bfloat16_rn(sum.y);
    }
    // Keep remote partials live until all ranks have finished reading.
    cluster.sync();
  }
#endif
}

using Kernel = void (*)(const __nv_bfloat16*, int64_t, int, const uint8_t*, const uint4*,
                        uint32_t, int, int, const __nv_bfloat16*, __nv_bfloat16*, int64_t, bool);

template <WeightSource S>
Kernel select_kernel(const SmallMDirectConfig& config) {
  if (config.kc != 1)
    return config.kc == 8 && config.warps == 8 ? small_m_direct_kernel<S, 8, true> : nullptr;
  switch (config.warps) {
    case 4: return small_m_direct_kernel<S, 4>;
    case 8: return small_m_direct_kernel<S, 8>;
    case 16: return small_m_direct_kernel<S, 16>;
    default: return nullptr;
  }
}

}  // namespace

SmallMDirectConfig small_m_direct_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (dev.sm_arch != 103 || m < 2 || m > 8) return {};
  if (m > 4 && n >= 128 && n <= 256 && k >= 7168 && k <= 8192) return {8, 8};
  if (n < 512 || n > 3072 || k < 64 || k > 2048) return {};
  if (k <= 128) return {4};
  // Target approximately 16 resident warps per SM across the output grid.
  const int64_t target = int64_t(dev.num_sms) * 16;
  const int64_t grid = n / 16;
  if (grid > dev.num_sms && m > 4 && k > 1536) return {};
  int warps = 4;
  int64_t distance = std::abs(grid * warps - target);
  const int candidates[] = {8, 16};
  for (const int candidate : candidates) {
    // A 16-warp block occupies an entire SM with this register footprint.
    if (candidate == 16 && grid > dev.num_sms) continue;
    const int64_t next = std::abs(grid * candidate - target);
    if (next < distance) {
      warps = candidate;
      distance = next;
    }
  }
  return {warps};
}

cudaError_t linear_small_m_direct(const LinearArgs& a, const SmallMDirectConfig& config) {
  const Kernel kernel = a.source == WeightSource::kNF12 ? select_kernel<WeightSource::kNF12>(config)
                                                       : select_kernel<WeightSource::kBF16T>(config);
  if (kernel == nullptr || (config.kc > 1 && a.dev->sm_arch < 90)) return cudaErrorInvalidValue;
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(a.n / 16 * config.kc);
  launch.blockDim = dim3(config.warps * 32);
  launch.stream = a.stream;
  cudaLaunchAttribute attributes[2]{};
  if (a.dev->sm_arch >= 90) {
    attributes[launch.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[launch.numAttrs++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (config.kc > 1) {
    attributes[launch.numAttrs].id = cudaLaunchAttributeClusterDimension;
    attributes[launch.numAttrs++].val.clusterDim = {static_cast<unsigned int>(config.kc), 1, 1};
  }
  launch.attrs = attributes;
  const int kb = static_cast<int>(a.k / 64);
  const int chunk = (kb + config.kc - 1) / config.kc;
  const bool weights_ready = a.source == WeightSource::kNF12 && a.weights_ready;
  return cudaLaunchKernelEx(&launch, kernel, a.x, a.ldx, static_cast<int>(a.m), a.weight,
                            a.escapes, exp_base_x4(a.exp_base), kb, chunk,
                            a.bias, a.y, a.ldy, weights_ready);
}

}  // namespace nf12::detail
