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
/*
 * Warp-specialized staging, activation swizzle and partial-sum layout adapted from
 * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu
 * and its upstream implementation:
 * https://github.com/NVIDIA/TensorRT-LLM/blob/619709fc33bd5dc268f19d6a741fe7ed51c0f8f5/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
 *
 * Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2021, NAVER Corp. Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "linear/small_m_async.h"

#include "linear/common.cuh"
#include "linear/small_m.h"

namespace nf12::detail {
namespace {

using Bf16 = __nv_bfloat16;
constexpr int kBarrierBytes = 1024;
constexpr int kActivationElements = 8 * 256;
constexpr int kEscapeAtomBytes = 4 * 32 * sizeof(uint4);

struct Params {
  const Bf16* x;
  int64_t ldx;
  const uint8_t* weights;
  const uint4* escapes;
  uint32_t num_escapes;
  uint32_t base;
  const Bf16* bias;
  Bf16* y;
  int64_t ldy;
  int m;
  int iterations;  // K / 256; each consumer owns K / 4 contiguous elements.
  bool weights_ready;
  bool launch_early;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void arrive_async(uint32_t bar) {
  // All producer lanes advance together and arrive on the same stage.
  bar = __shfl_sync(0xffffffffu, bar, 0);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void arrive_consumer(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ uint4 load_plane(uint32_t address) {
  uint4 value;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "r"(address) : "memory");
  return value;
}

__device__ __forceinline__ int partial_index(int row, int token) {
  return (row % 4) * 8 + (row / 4) * 34 + token;
}

__device__ __forceinline__ void store_output(float (&accum)[4], uint8_t* packed,
                                            const Params& p, int part, int lane) {
  asm volatile("bar.sync 1, 128;" ::: "memory");
  constexpr int kPartialElements = 4 * 34;
  auto* partials = reinterpret_cast<float*>(packed);
#pragma unroll
  for (int row_half = 0; row_half < 2; ++row_half) {
#pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
      const int row = lane / 4 + row_half * 8;
      const int col = (lane % 4) * 2 + col_half;
      partials[part * kPartialElements + partial_index(row, col)] = accum[row_half * 2 + col_half];
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (part == 0) {
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      const int index = reg * 32 + lane;
      const int row = index % 16;
      const int col = index / 16;
      const int offset = partial_index(row, col);
      float sum = 0.0f;
      sum += partials[offset] + partials[offset + kPartialElements] +
             partials[offset + 2 * kPartialElements] + partials[offset + 3 * kPartialElements];
      if (col < p.m) {
        const int output_row = blockIdx.x * 16 + row;
        if (p.bias != nullptr) sum += __bfloat162float(p.bias[output_row]);
        p.y[col * p.ldy + output_row] = __float2bfloat16_rn(sum);
      }
    }
  }
}
#endif

// Raw staging uses two weight and two activation warps. Escape staging uses
// one of each plus two escape-copy warps. Both have four MMA consumers.
template <WeightSource S, int Stages, bool StageEscapes>
__global__ void __launch_bounds__(256, 1) small_m_async_kernel(Params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using T = WeightTraits<S, true>;
  constexpr int Bars = StageEscapes ? 3 : 2;
  constexpr int Full = StageEscapes ? 1 : 0;
  constexpr int Empty = Bars - 1;
  constexpr int WeightWarps = StageEscapes ? 1 : 2;
  constexpr int ActivationEnd = StageEscapes ? 2 : 4;
  extern __shared__ __align__(16) uint8_t memory[];
  auto* barriers = reinterpret_cast<uint64_t*>(memory);
  uint8_t* packed = memory + kBarrierBytes;
  auto* activations = reinterpret_cast<Bf16*>(packed + Stages * 4 * T::kSuperAtomBytes);
  uint8_t* staged_escapes = reinterpret_cast<uint8_t*>(activations + Stages * kActivationElements);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (p.launch_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      if constexpr (StageEscapes) ring_bar_init(smem_addr(barriers + stage * Bars), 32);
      ring_bar_init(smem_addr(barriers + stage * Bars + Full), StageEscapes ? 96 : 128);
      ring_bar_init(smem_addr(barriers + stage * Bars + Empty), 128);
    }
    ring_bar_fence_init();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  int stage = 0;
  int phase = warp < ActivationEnd ? 1 : 0;
  float accum[4]{};
  if (warp < WeightWarps) {
    // The host enables early reads only for the immutable NF12 contract.
    if (!p.weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
    const uint8_t* weights = p.weights + static_cast<size_t>(blockIdx.x) * p.iterations * 4 * T::kSuperAtomBytes;
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
#pragma unroll 1
    for (int iteration = 0; iteration < p.iterations; ++iteration) {
      ring_bar_wait(smem_addr(barriers + stage * Bars + Empty), phase);
#pragma unroll
      for (int half = 0; half < 4 / WeightWarps; ++half) {
        const int part = warp + half * WeightWarps;
        const uint8_t* source = weights + static_cast<size_t>(part * p.iterations + iteration) *
                                            T::kSuperAtomBytes + lane * 16;
        const uint32_t target = smem_addr(packed + (stage * 4 + part) * T::kSuperAtomBytes + lane * 16);
#pragma unroll
        for (int plane = 0; plane < T::kLoads; ++plane)
          asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                       ::"r"(target + plane * kPlaneBytes), "l"(source + plane * kPlaneBytes),
                         "l"(policy) : "memory");
      }
      arrive_async(smem_addr(barriers + stage * Bars));
      if (++stage == Stages) {
        stage = 0;
        phase ^= 1;
      }
    }
  } else if (warp < ActivationEnd) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int local_thread = StageEscapes ? lane : (threadIdx.x & 63);
    const int column = lane * 8;
    const int projected = (column / 64) * (p.iterations * 64) + column % 64;
#pragma unroll 1
    for (int iteration = 0; iteration < p.iterations; ++iteration) {
      ring_bar_wait(smem_addr(barriers + stage * Bars + Empty), phase);
#pragma unroll
      for (int copy = 0; copy < (StageEscapes ? 8 : 4); ++copy) {
        const int token = StageEscapes ? copy : local_thread / 32 + copy * 2;
        const uint32_t target = smem_addr(activations + stage * kActivationElements +
                                         token * 256 + (column ^ (token * 8)));
        // Use a valid source address even for padded tokens; zero-fill keeps
        // every ldmatrix operand initialized without accessing rows beyond M.
        const Bf16* source = p.x + (token < p.m ? token * p.ldx : 0) + projected + iteration * 64;
        const uint32_t bytes = token < p.m ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     ::"r"(target), "l"(source), "r"(bytes) : "memory");
      }
      arrive_async(smem_addr(barriers + stage * Bars + Full));
      if (++stage == Stages) {
        stage = 0;
        phase ^= 1;
      }
    }
  } else if (StageEscapes && warp < 4) {
    if (!p.weights_ready) asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll 1
    for (int iteration = 0; iteration < p.iterations; ++iteration) {
      ring_bar_wait(smem_addr(barriers + stage * Bars), phase);
      if constexpr (S == WeightSource::kNF12) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int part = warp - 2 + half * 2;
          const uint32_t address = smem_addr(packed + (stage * 4 + part) * T::kSuperAtomBytes + lane * 16);
          const uint4 meta = load_plane(address);
          const uint4 low01 = load_plane(address + kPlaneBytes);
          const uint4 low23 = load_plane(address + 2 * kPlaneBytes);
          const uint32_t metadata[4] = {meta.x, meta.y, meta.z, meta.w};
          const uint32_t indices[4] = {low01.x, low01.z, low23.x, low23.z};
#pragma unroll
          for (int group = 0; group < 4; ++group) {
            if (metadata[group] == 0u) {
              const uint32_t target = smem_addr(staged_escapes + (stage * 4 + part) * kEscapeAtomBytes +
                                                (group * 32 + lane) * sizeof(uint4));
              cp_async_16(target, p.escapes + indices[group]);
            }
          }
        }
      }
      arrive_async(smem_addr(barriers + stage * Bars + Full));
      if (++stage == Stages) {
        stage = 0;
        phase ^= 1;
      }
    }
  } else {
    // Explicitly wait in consumers too: escapes, bias and output are coherent
    // for mutable weights, independent of producer-to-consumer visibility.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int part = warp - 4;
    const int token = lane % 8;
    uint32_t activation_offsets[2];
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
      activation_offsets[pair] = (token * 256 + ((part * 64 + (lane / 8) * 8 + pair * 32) ^ (token * 8))) * 2;
#pragma unroll 1
    for (int iteration = 0; iteration < p.iterations; ++iteration) {
      ring_bar_wait(smem_addr(barriers + stage * Bars + Full), phase);
      const uint32_t address = smem_addr(packed + (stage * 4 + part) * T::kSuperAtomBytes + lane * 16);
      uint4 planes[T::kLoads];
#pragma unroll
      for (int plane = 0; plane < T::kLoads; ++plane)
        planes[plane] = load_plane(address + plane * kPlaneBytes);
      uint32_t fragments[4][4];
      if constexpr (StageEscapes && S == WeightSource::kNF12) {
        const Group groups[4] = {{planes[0].x, planes[1].x, planes[1].y},
                                 {planes[0].y, planes[1].z, planes[1].w},
                                 {planes[0].z, planes[2].x, planes[2].y},
                                 {planes[0].w, planes[2].z, planes[2].w}};
#pragma unroll
        for (int group = 0; group < 4; ++group) {
          decode_bulk(groups[group], p.base, fragments[group]);
          if (groups[group].meta == 0u) {
            const uint4 value = load_plane(smem_addr(
                staged_escapes + (stage * 4 + part) * kEscapeAtomBytes + (group * 32 + lane) * sizeof(uint4)));
            fragments[group][0] = value.x;
            fragments[group][1] = value.y;
            fragments[group][2] = value.z;
            fragments[group][3] = value.w;
          }
        }
      } else {
        T::fragments(planes, p.base, p.escapes, p.num_escapes, fragments);
      }
      uint32_t operands[2][4];
      const uint32_t activation_stage = smem_addr(activations + stage * kActivationElements);
#pragma unroll
      for (int pair = 0; pair < 2; ++pair)
        ldmatrix_x4(operands[pair], activation_stage + activation_offsets[pair]);
#pragma unroll
      for (int atom = 0; atom < 4; ++atom)
        mma_bf16_16816(accum, fragments[atom], operands[atom / 2][(atom % 2) * 2],
                      operands[atom / 2][(atom % 2) * 2 + 1]);
      arrive_consumer(smem_addr(barriers + stage * Bars + Empty));
      if (++stage == Stages) {
        stage = 0;
        phase ^= 1;
      }
    }
  }
  // Escape producers must finish before the packed ring becomes reduction scratch.
  if constexpr (StageEscapes) __syncthreads();
  if (warp >= 4) store_output(accum, packed, p, warp - 4, lane);
  if (!p.launch_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#else
  __trap();
#endif
}

template <WeightSource S, int Stages, bool StageEscapes>
cudaError_t launch(const LinearArgs& a) {
  constexpr int escape_bytes = StageEscapes && S == WeightSource::kNF12 ? 4 * kEscapeAtomBytes : 0;
  constexpr int shared = kBarrierBytes +
      Stages * (4 * WeightTraits<S>::kSuperAtomBytes + kActivationElements * 2 + escape_bytes);
  if (shared > a.dev->smem_optin_bytes) return cudaErrorNotSupported;
  const cudaError_t status = small_m_prepare_kernel(
      reinterpret_cast<const void*>(small_m_async_kernel<S, Stages, StageEscapes>), *a.dev, shared);
  if (status != cudaSuccess) return status;
  Params p{a.x, a.ldx, a.weight, a.escapes, a.num_escapes, exp_base_x4(a.exp_base), a.bias, a.y, a.ldy,
           static_cast<int>(a.m), static_cast<int>(a.k / 256), S == WeightSource::kNF12 && a.weights_ready,
           a.k <= 2048 ? a.n > 4096 : a.n <= 4096};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(a.n / 16));
  config.blockDim = dim3(256);
  config.dynamicSmemBytes = shared;
  config.stream = a.stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, small_m_async_kernel<S, Stages, StageEscapes>, p);
}

template <WeightSource S, bool StageEscapes>
cudaError_t launch_stages(const LinearArgs& a, int stages) {
  if constexpr (StageEscapes) {
    return stages == 4 ? launch<S, 4, true>(a) : cudaErrorInvalidValue;
  } else {
    switch (stages) {
      case 2: return launch<S, 2, false>(a);
      case 3: return launch<S, 3, false>(a);
      case 4: return launch<S, 4, false>(a);
      case 8: return launch<S, 8, false>(a);
      default: return cudaErrorInvalidValue;
    }
  }
}

}  // namespace

SmallMAsyncConfig small_m_async_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev) {
  if (dev.sm_arch != 103 || m < 2 || m > 8 || n <= 0 || n > 8192 || n % 16 != 0 || k % 256 != 0) return {};
  if (k >= 512 && k <= 2048 && n >= 4096) return {static_cast<int>(k < 1024 ? k / 256 : 4)};
  if (k >= 4096 && k <= 8192 && n >= 3072) {
    if (n <= 4096) return {8};
    // Four escape stages fit three CTAs per SM. Avoid an extra grid wave.
    if (m <= 4 || n / 16 > int64_t(dev.num_sms) * 3) return {};
    return {4, true};
  }
  return {};
}

cudaError_t linear_small_m_async(const LinearArgs& a, const SmallMAsyncConfig& config) {
  if (a.dev == nullptr) return cudaErrorInvalidValue;
  if (a.dev->sm_arch != 103) return cudaErrorNotSupported;
  if (a.m < 1 || a.m > 8 || a.n <= 0 || a.n % 16 != 0 || a.k <= 0 || a.k % 256 != 0)
    return cudaErrorInvalidValue;
  switch (a.source) {
    case WeightSource::kNF12:
      return config.stage_escapes ? launch_stages<WeightSource::kNF12, true>(a, config.stages)
                                 : launch_stages<WeightSource::kNF12, false>(a, config.stages);
    case WeightSource::kBF16T:
      return config.stage_escapes ? launch_stages<WeightSource::kBF16T, true>(a, config.stages)
                                 : launch_stages<WeightSource::kBF16T, false>(a, config.stages);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace nf12::detail
