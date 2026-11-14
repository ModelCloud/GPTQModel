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
#include "linear/gemv.h"

#include <algorithm>
#include <cstdlib>

#include "linear/common.cuh"

namespace nf12::detail {
namespace {

__device__ __forceinline__ uint32_t load_x_pair(const __nv_bfloat16* p) {
  uint32_t value;
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}

template <bool High>
__device__ __forceinline__ float bf16_fma(uint32_t weight, uint32_t x, float sum) {
#if __CUDA_ARCH__ >= 1000
  if constexpr (High) {
    asm("{ .reg .b16 w0, w1, x0, x1; mov.b32 {w0,w1}, %1; mov.b32 {x0,x1}, %2;"
        " fma.rn.f32.bf16 %0, w1, x1, %0; }" : "+f"(sum) : "r"(weight), "r"(x));
  } else {
    asm("{ .reg .b16 w0, w1, x0, x1; mov.b32 {w0,w1}, %1; mov.b32 {x0,x1}, %2;"
        " fma.rn.f32.bf16 %0, w0, x0, %0; }" : "+f"(sum) : "r"(weight), "r"(x));
  }
  return sum;
#else
  const uint32_t a = High ? weight & 0xffff0000u : weight << 16;
  const uint32_t b = High ? x & 0xffff0000u : x << 16;
  return __fmaf_rn(__uint_as_float(a), __uint_as_float(b), sum);
#endif
}

template <WeightSource S>
__device__ __forceinline__ void fragments(const uint4 (&p)[WeightTraits<S>::kLoads],
                                        uint32_t base, const uint4* escapes, uint32_t num_escapes,
                                        uint32_t (&a)[4][4]) {
  if constexpr (S == WeightSource::kNF12) {
    const Group groups[4] = {{p[0].x, p[1].x, p[1].y}, {p[0].y, p[1].z, p[1].w},
                             {p[0].z, p[2].x, p[2].y}, {p[0].w, p[2].z, p[2].w}};
#pragma unroll
    for (int j = 0; j < 4; ++j) decode_bulk(groups[j], base, a[j]);
    if ((p[0].x == 0u) | (p[0].y == 0u) | (p[0].z == 0u) | (p[0].w == 0u)) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        decode_group_any_lane<true>(groups[j], base, escapes, num_escapes, a[j]);
    }
  } else {
    WeightTraits<S, true>::fragments(p, base, escapes, num_escapes, a);
  }
}

// Four output rows share each packed group. The lane's bit 2 selects an
// adjacent row, while each lane accumulates the two rows eight apart.
template <WeightSource S, int Warps>
__global__ void __launch_bounds__(Warps * 32) gemv_four_rows(
    const __nv_bfloat16* x, const uint8_t* weights, const uint4* escapes, uint32_t base, uint32_t num_escapes,
    int kb, const __nv_bfloat16* bias, __nv_bfloat16* output, bool weights_ready) {
  using T = WeightTraits<S, true>;
  constexpr int Parts = Warps * 4;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int part = threadIdx.x >> 3;
  const int packed_lane = (blockIdx.x & 3) * 8 + (lane & 7);
  const uint8_t* first = weights + static_cast<size_t>(blockIdx.x / 4) * kb * T::kSuperAtomBytes + packed_lane * 16;
  float sums[2][4]{};
  uint4 loaded[T::kLoads];
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (!weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (part < kb) {
#pragma unroll
    for (int j = 0; j < T::kLoads; ++j)
      loaded[j] = load_global_u4<true>(reinterpret_cast<const uint4*>(first + part * T::kSuperAtomBytes + j * kPlaneBytes));
  }
#if __CUDA_ARCH__ >= 900
  if (weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  for (int p = part; p < kb; p += Parts) {
    uint32_t values[4][4];
    fragments<S>(loaded, base, escapes, num_escapes, values);
    if (p + Parts < kb) {
#pragma unroll
      for (int j = 0; j < T::kLoads; ++j)
        loaded[j] = load_global_u4<true>(reinterpret_cast<const uint4*>(
            first + static_cast<size_t>(p + Parts) * T::kSuperAtomBytes + j * kPlaneBytes));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int column = p * 64 + j * 16 + 2 * (lane & 3);
      const uint32_t a = load_x_pair(x + column);
      const uint32_t b = load_x_pair(x + column + 8);
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        sums[row][0] = bf16_fma<false>(values[j][row], a, sums[row][0]);
        sums[row][1] = bf16_fma<true>(values[j][row], a, sums[row][1]);
        sums[row][2] = bf16_fma<false>(values[j][row + 2], b, sums[row][2]);
        sums[row][3] = bf16_fma<true>(values[j][row + 2], b, sums[row][3]);
      }
    }
  }
  __shared__ float partial[Warps][4];
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    float sum = sums[row][0];
#pragma unroll
    for (int j = 1; j < 4; ++j) sum += sums[row][j];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    sum += __shfl_xor_sync(0xffffffffu, sum, 8);
    sum += __shfl_xor_sync(0xffffffffu, sum, 16);
    if (lane < 8 && (lane & 3) == 0) partial[warp][(lane >> 2) + row * 2] = sum;
  }
  __syncthreads();
  if (threadIdx.x < 4) {
    const int col = threadIdx.x;
    float sum = partial[0][col];
#pragma unroll
    for (int w = 1; w < Warps; ++w) sum += partial[w][col];
    const int n = (blockIdx.x / 4) * 16 + (blockIdx.x & 3) * 2 + col % 2 + (col / 2) * 8;
    if (bias != nullptr) sum += __bfloat162float(bias[n]);
    output[n] = __float2bfloat16_rn(sum);
  }
}

template <WeightSource S>
__device__ __forceinline__ void row_fragments(const uint4 (&planes)[WeightTraits<S>::kLoads],
                                            uint32_t base, const uint4* escapes, uint32_t num_escapes, bool upper,
                                            uint32_t (&values)[4][2]) {
  if constexpr (S == WeightSource::kNF12) {
    const Group groups[4] = {
        {planes[0].x, planes[1].x, planes[1].y}, {planes[0].y, planes[1].z, planes[1].w},
        {planes[0].z, planes[2].x, planes[2].y}, {planes[0].w, planes[2].z, planes[2].w}};
    const uint32_t selector = upper ? 0x7362 : 0x5140;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const Group& g = groups[j];
      const uint32_t a = (g.meta & 0x87878787u) + base;
      const uint32_t b = (((g.meta << 1) & 0x80808080u) | ((g.meta >> 3) & 0x07070707u)) + base;
      values[j][0] = byte_perm(g.lo0, byte_perm(a, b, 0x5140), selector);
      values[j][1] = byte_perm(g.lo1, byte_perm(a, b, 0x7362), selector);
      if (g.meta == 0 && g.lo0 < num_escapes) {
        const uint4 original = load_global_u4<true>(escapes + g.lo0);
        values[j][0] = upper ? original.y : original.x;
        values[j][1] = upper ? original.w : original.z;
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      values[j][0] = upper ? planes[j].y : planes[j].x;
      values[j][1] = upper ? planes[j].w : planes[j].z;
    }
  }
}

// A narrow output needs more CTAs. Decode only the selected row of each group.
template <WeightSource S, int Warps>
__global__ void __launch_bounds__(Warps * 32) gemv_one_row(
    const __nv_bfloat16* x, const uint8_t* weights, const uint4* escapes, uint32_t base, uint32_t num_escapes,
    int kb, const __nv_bfloat16* bias, __nv_bfloat16* output, bool weights_ready) {
  using T = WeightTraits<S, true>;
  constexpr int Parts = Warps * 8;
  const int lane = threadIdx.x & 31;
  const int part = threadIdx.x >> 2;
  const bool upper = (blockIdx.x & 8) != 0;
  const int packed_lane = (blockIdx.x & 7) * 4 + (lane & 3);
  const uint8_t* first = weights + static_cast<size_t>(blockIdx.x / 16) * kb * T::kSuperAtomBytes + packed_lane * 16;
  uint4 loaded[T::kLoads];
  uint32_t values[4][2];
  float sums[4]{};
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (!weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (part < kb) {
#pragma unroll
    for (int j = 0; j < T::kLoads; ++j)
      loaded[j] = load_global_u4<true>(reinterpret_cast<const uint4*>(first + part * T::kSuperAtomBytes + j * kPlaneBytes));
    row_fragments<S>(loaded, base, escapes, num_escapes, upper, values);
  }
#if __CUDA_ARCH__ >= 900
  if (weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  for (int p = part; p < kb; p += Parts) {
    if (p + Parts < kb) {
#pragma unroll
      for (int j = 0; j < T::kLoads; ++j)
        loaded[j] = load_global_u4<true>(reinterpret_cast<const uint4*>(
            first + static_cast<size_t>(p + Parts) * T::kSuperAtomBytes + j * kPlaneBytes));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int column = p * 64 + j * 16 + 2 * (lane & 3);
      const uint32_t a = load_x_pair(x + column);
      const uint32_t b = load_x_pair(x + column + 8);
      sums[0] = bf16_fma<false>(values[j][0], a, sums[0]);
      sums[1] = bf16_fma<true>(values[j][0], a, sums[1]);
      sums[2] = bf16_fma<false>(values[j][1], b, sums[2]);
      sums[3] = bf16_fma<true>(values[j][1], b, sums[3]);
    }
    if (p + Parts < kb) row_fragments<S>(loaded, base, escapes, num_escapes, upper, values);
  }
  float sum = sums[0];
#pragma unroll
  for (int j = 1; j < 4; ++j) sum += sums[j];
#pragma unroll
  for (int shift = 1; shift < 32; shift *= 2) sum += __shfl_xor_sync(0xffffffffu, sum, shift);
  __shared__ float partial[Warps];
  if (lane == 0) partial[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    sum = partial[0];
#pragma unroll
    for (int w = 1; w < Warps; ++w) sum += partial[w];
    if (bias != nullptr) sum += __bfloat162float(bias[blockIdx.x]);
    output[blockIdx.x] = __float2bfloat16_rn(sum);
  }
}

using Kernel = void (*)(const __nv_bfloat16*, const uint8_t*, const uint4*, uint32_t, uint32_t,
                        int, const __nv_bfloat16*, __nv_bfloat16*, bool);

template <WeightSource S>
Kernel select_kernel(const GemvConfig& config) {
  if (config.rows == 1) {
    switch (config.warps) {
      case 4: return gemv_one_row<S, 4>;
      case 8: return gemv_one_row<S, 8>;
      case 12: return gemv_one_row<S, 12>;
      case 14: return gemv_one_row<S, 14>;
      case 16: return gemv_one_row<S, 16>;
      default: return nullptr;
    }
  }
  if (config.rows == 4) {
    switch (config.warps) {
      case 2: return gemv_four_rows<S, 2>;
      case 3: return gemv_four_rows<S, 3>;
      case 4: return gemv_four_rows<S, 4>;
      case 6: return gemv_four_rows<S, 6>;
      case 8: return gemv_four_rows<S, 8>;
      case 12: return gemv_four_rows<S, 12>;
      case 16: return gemv_four_rows<S, 16>;
      default: return nullptr;
    }
  }
  return nullptr;
}

}  // namespace

GemvConfig gemv_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (dev.sm_arch != 103 || m != 1 || n < 128 || n > 8192 || k < 512 || k > 8192) return {};
  if (k > 2048 && n > 7168) return {};
  if (n <= 256) {
    if (k < 2048) return {};
    return {k <= 2048 ? 4 : k <= 4096 ? 8 : k <= 6144 ? 12 : k <= 7168 ? 14 : 16, 1};
  }
  if (n < 512) return {};
  // Keep approximately 28 warps per SM in flight. Short K caps the split
  // before most threads would have no weight fragment to consume.
  const int64_t target = int64_t(dev.num_sms) * 28;
  const int cap = k <= 1024 ? 4 : k <= 2048 ? 8 : 16;
  int warps = 2;
  int64_t distance = std::abs(n / 4 * warps - target);
  for (const int candidate : {3, 4, 6, 8, 12, 16}) {
    if (candidate > cap) break;
    const int64_t next = std::abs(n / 4 * candidate - target);
    if (next < distance) {
      warps = candidate;
      distance = next;
    }
  }
  return {warps, 4};
}

cudaError_t linear_gemv(const LinearArgs& a, const GemvConfig& config) {
  const Kernel kernel = a.source == WeightSource::kNF12 ? select_kernel<WeightSource::kNF12>(config)
                                                       : select_kernel<WeightSource::kBF16T>(config);
  if (kernel == nullptr) return cudaErrorInvalidValue;
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(a.n / config.rows);
  launch.blockDim = dim3(config.warps * 32);
  launch.stream = a.stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  if (a.dev->sm_arch >= 90) {
    launch.attrs = &attribute;
    launch.numAttrs = 1;
  }
  const bool weights_ready = a.source == WeightSource::kNF12 && a.weights_ready;
  return cudaLaunchKernelEx(&launch, kernel, a.x, a.weight, a.escapes, exp_base_x4(a.exp_base), a.num_escapes,
                            static_cast<int>(a.k / 64), a.bias, a.y, weights_ready);
}

}  // namespace nf12::detail
