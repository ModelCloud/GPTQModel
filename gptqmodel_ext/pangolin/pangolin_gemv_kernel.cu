// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0
// Contact: qubitium@modelcloud.ai, x.com/qubitium

// Native decode-regime GEMV for the planar (gptq_p) GPTQ format at 3/5/6/7
// bits. Register-level decode: each 32-code block loads its `bits` packed
// words once into registers and derives all 32 codes with compile-time
// shifts/masks, so weight-side DRAM traffic is only the packed words.
//
// Layout contract (docs/gptq_planar.md):
// - qweight [K/32*bits, N] int32: every 32 logical rows use `bits` adjacent
//   word rows, low plane first; no code crosses a word boundary.
// - qzeros [G, N/32*bits] int32: same plane scheme along output columns.
// - scales [G, N] fp16/bf16, zeros use v2 semantics (no +1 bias).
// - g_idx must map each 32-row block to a single group (checked in Python;
//   the block group is read from g_idx[row0]).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>


namespace pangolin {

constexpr int kWarpSize = 32;
constexpr int kBlockN = 32;     // minimum column granularity (one qzeros column block)
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kMaxM = 32;       // decode-regime rows handled per launch

template <typename Scalar>
struct ScalarTraits;

template <>
struct ScalarTraits<half> {
  using Vec2 = __half2;
  static __device__ __forceinline__ float to_float(half value) { return __half2float(value); }
  static __device__ __forceinline__ half from_float(float value) { return __float2half_rn(value); }
  static __device__ __forceinline__ float low(Vec2 v) { return __low2float(v); }
  static __device__ __forceinline__ float high(Vec2 v) { return __high2float(v); }
  static __device__ __forceinline__ Vec2 make2(half v) { return __half2half2(v); }
  static __device__ __forceinline__ Vec2 make2(half low, half high) { return __halves2half2(low, high); }
  static __device__ __forceinline__ Vec2 from_floats(float low, float high) { return __floats2half2_rn(low, high); }
  static __device__ __forceinline__ Vec2 fma2(Vec2 a, Vec2 b, Vec2 c) { return __hfma2(a, b, c); }
  static __device__ __forceinline__ Vec2 neg2(Vec2 v) { return __hneg2(v); }
  static __device__ __forceinline__ Vec2 mul2(Vec2 a, Vec2 b) { return __hmul2(a, b); }
  // Shuffle a 16-bit value through a 32-bit warp shuffle and reinterpret.
  static __device__ __forceinline__ half shfl(half v, int k) {
    const unsigned int u = __shfl_sync(0xffffffffu, static_cast<unsigned int>(__half_as_ushort(v)), k);
    return __ushort_as_half(static_cast<unsigned short>(u));
  }
};

template <>
struct ScalarTraits<nv_bfloat16> {
  using Vec2 = __nv_bfloat162;
  static __device__ __forceinline__ float to_float(nv_bfloat16 value) { return __bfloat162float(value); }
  static __device__ __forceinline__ nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }
  static __device__ __forceinline__ float low(Vec2 v) { return __bfloat162float(__low2bfloat16(v)); }
  static __device__ __forceinline__ float high(Vec2 v) { return __bfloat162float(__high2bfloat16(v)); }
  static __device__ __forceinline__ Vec2 make2(nv_bfloat16 v) { return __bfloat162bfloat162(v); }
  static __device__ __forceinline__ Vec2 make2(nv_bfloat16 low, nv_bfloat16 high) { return __halves2bfloat162(low, high); }
  static __device__ __forceinline__ Vec2 from_floats(float low, float high) { return __floats2bfloat162_rn(low, high); }
  static __device__ __forceinline__ Vec2 fma2(Vec2 a, Vec2 b, Vec2 c) { return __hfma2(a, b, c); }
  static __device__ __forceinline__ Vec2 neg2(Vec2 v) { return __hneg2(v); }
  static __device__ __forceinline__ Vec2 mul2(Vec2 a, Vec2 b) { return __hmul2(a, b); }
  // Shuffle a 16-bit value through a 32-bit warp shuffle and reinterpret.
  static __device__ __forceinline__ nv_bfloat16 shfl(nv_bfloat16 v, int k) {
    const unsigned int u = __shfl_sync(0xffffffffu, static_cast<unsigned int>(__bfloat16_as_ushort(v)), k);
    return __ushort_as_bfloat16(static_cast<unsigned short>(u));
  }
};

// Read-only cache load for bandwidth-bound, read-once planar data.
template <typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
  return __ldg(ptr);
}

// Plane widths per bit-width: bits = W0 + W1 + W2, planes packed low-first.
template <int Bits>
struct PlaneSpec;

template <>
struct PlaneSpec<3> {
  static constexpr int kW0 = 2, kW1 = 1, kW2 = 0;
};
template <>
struct PlaneSpec<5> {
  static constexpr int kW0 = 4, kW1 = 1, kW2 = 0;
};
template <>
struct PlaneSpec<6> {
  static constexpr int kW0 = 4, kW1 = 2, kW2 = 0;
};
template <>
struct PlaneSpec<7> {
  static constexpr int kW0 = 4, kW1 = 2, kW2 = 1;
};

template <int Width>
__device__ __forceinline__ int plane_code(const uint32_t* words, int k) {
  constexpr int kPackFactor = 32 / Width;
  constexpr uint32_t kMask = (1u << Width) - 1u;
  return static_cast<int>((words[k / kPackFactor] >> (Width * (k % kPackFactor))) & kMask);
}

// Decode the logical code for lane-column `r` from already-loaded qzero words.
template <int Bits>
__device__ __forceinline__ int decode_zero_from_words(const uint32_t* words, int r) {
  using Spec = PlaneSpec<Bits>;
  int code = plane_code<Spec::kW0>(words, r);
  code |= plane_code<Spec::kW1>(words + Spec::kW0, r) << Spec::kW0;
  if constexpr (Spec::kW2 > 0) {
    code |= plane_code<Spec::kW2>(words + Spec::kW0 + Spec::kW1, r) << (Spec::kW0 + Spec::kW1);
  }
  return code;
}

// Decode the logical code for lane-column `r` of one qzeros column block.
template <int Bits>
__device__ __forceinline__ int decode_zero(const int32_t* qzeros_block, int r) {
  const uint32_t* words_src = reinterpret_cast<const uint32_t*>(qzeros_block);
  uint32_t words[Bits];
#pragma unroll
  for (int i = 0; i < Bits; ++i) {
    words[i] = ldg(words_src + i);
  }
  return decode_zero_from_words<Bits>(words, r);
}

// One block owns kWarpSize * ColsPerLane output columns and a strided set of
// 32-row k-blocks. Each warp decodes whole k-blocks (`bits` packed word loads
// per owned column, reused for all 32 codes in registers); ColsPerLane == 2
// loads adjacent column pairs as one 8-byte `uint2` and shares each shuffled
// activation across both columns, halving shuffle traffic per output. Warp
// partials reduce through shared memory and each split-K block writes its
// partial to its own fp32 workspace slice (no atomics, no zero-init needed).
// The last block to finish a column block sums the slices and converts to the
// output dtype in-kernel, so the whole op is a single kernel launch plus one
// tiny counter memset.
template <typename Scalar, int Bits, int SizeM, int ColsPerLane, int Warps>
__global__ __launch_bounds__(Warps * kWarpSize) void pangolin_gemv_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ qweight,
    const Scalar* __restrict__ scales,
    const int32_t* __restrict__ qzeros,
    const int32_t* __restrict__ g_idx,
    Scalar* __restrict__ output,
    float* __restrict__ workspace,
    int* __restrict__ counters,
    int size_k,
    int size_n,
    int num_groups) {
  using Spec = PlaneSpec<Bits>;
  using AccScalar2 = typename ScalarTraits<Scalar>::Vec2;
  constexpr int kBlockCols = kWarpSize * ColsPerLane;
  constexpr int kPartialCols = kBlockCols;
  // Dynamic shared-memory partials sized at launch to allow large SizeM.
  extern __shared__ float partials[];

  constexpr int kThreads = Warps * kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int column_block = blockIdx.x;
  const int n0 = column_block * kBlockCols + lane * ColsPerLane;
  const int num_k_blocks = size_k / 32;
  const int qzeros_stride = (size_n / 32) * Bits;

  // Use vector half2 accumulation for small-M fp16 decode (the latency-critical
  // path); for bf16 and for larger M use fp32 weights/products to avoid the
  // precision drift seen over long K reductions.  The result of each 32-row
  // k-block is flushed into fp32 shared partials so split-K/warp reductions stay
  // in full precision.
  constexpr bool kHalf2FastPath =
      ColsPerLane == 2 && std::is_same_v<Scalar, half> && SizeM <= 8;
  using AccArray = typename std::conditional<
      kHalf2FastPath, AccScalar2[SizeM], float[SizeM][ColsPerLane]>::type;
  AccArray acc{};
  if constexpr (kHalf2FastPath) {
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
      acc[m] = AccScalar2();
    }
  }

  constexpr int kPartialCount = Warps * SizeM * kPartialCols;
  for (int i = threadIdx.x; i < kPartialCount; i += kThreads) {
    partials[i] = 0.0f;
  }
  __syncthreads();

  for (int blk = blockIdx.y * Warps + warp; blk < num_k_blocks;
       blk += gridDim.y * Warps) {
    const int row0 = blk * 32;
    int group = g_idx[row0];
    if (group < 0) {
      group += num_groups;
    }

    // Fold the zero point into the scale so each code costs one FMA:
    // (code - zero) * scale == code * scale - zero * scale.
    // Load the qzero words once per 32-column block; all columns handled by
    // this lane live in the same block, and ColsPerLane==2 scale loads can be
    // issued as a single vector load.
    float scale[ColsPerLane];
    int zero[ColsPerLane];
    AccScalar2 scale2;
    const int32_t* qzeros_base = qzeros + static_cast<int64_t>(group) * qzeros_stride +
        (n0 / 32) * Bits;
    uint32_t zero_words[Bits];
#pragma unroll
    for (int i = 0; i < Bits; ++i) {
      zero_words[i] = ldg(reinterpret_cast<const uint32_t*>(qzeros_base) + i);
    }
    if constexpr (ColsPerLane == 2) {
      scale2 = ldg(
          reinterpret_cast<const AccScalar2*>(scales + static_cast<int64_t>(group) * size_n + n0));
      scale[0] = ScalarTraits<Scalar>::low(scale2);
      scale[1] = ScalarTraits<Scalar>::high(scale2);
      zero[0] = decode_zero_from_words<Bits>(zero_words, n0 % 32);
      zero[1] = decode_zero_from_words<Bits>(zero_words, (n0 + 1) % 32);
    } else if constexpr (ColsPerLane == 4) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        scale[c] = ScalarTraits<Scalar>::to_float(
            ldg(scales + static_cast<int64_t>(group) * size_n + n0 + c));
        zero[c] = decode_zero_from_words<Bits>(zero_words, (n0 + c) % 32);
      }
    } else {
#pragma unroll
      for (int c = 0; c < ColsPerLane; ++c) {
        scale[c] = ScalarTraits<Scalar>::to_float(
            ldg(scales + static_cast<int64_t>(group) * size_n + n0 + c));
        zero[c] = decode_zero_from_words<Bits>(zero_words, (n0 + c) % 32);
      }
    }

    uint32_t words[ColsPerLane][Bits];
    const int32_t* qweight_block = qweight + static_cast<int64_t>(blk) * Bits * size_n + n0;
#pragma unroll
    for (int w = 0; w < Bits; ++w) {
      const int64_t qweight_offset = static_cast<int64_t>(w) * size_n;
      if constexpr (ColsPerLane == 4) {
        // n0 is a multiple of 4 and size_n % 128 == 0, so the quad load is
        // 16-byte aligned and each lane's 4 columns stay inside one 32-column
        // qzeros block.
        const uint4 quad = ldg(
            reinterpret_cast<const uint4*>(qweight_block + qweight_offset));
        words[0][w] = quad.x;
        words[1][w] = quad.y;
        words[2][w] = quad.z;
        words[3][w] = quad.w;
      } else if constexpr (ColsPerLane == 2) {
        // n0 is even and size_n % 64 == 0, so the pair load is 8-byte aligned.
        const uint2 pair = ldg(
            reinterpret_cast<const uint2*>(qweight_block + qweight_offset));
        words[0][w] = pair.x;
        words[1][w] = pair.y;
      } else {
        words[0][w] = static_cast<uint32_t>(ldg(qweight_block + qweight_offset));
      }
    }

    // Every lane needs all 32 activations of the block: each lane loads one
    // (coalesced) and the unrolled loop broadcasts them with warp shuffles.
    // The half2 path keeps native FP16. At M >= 8, other paths convert once at
    // load time; converting after every shuffled value repeated the same exact
    // cast up to 32 * SizeM times per lane. Keep the original native shuffle at
    // lower M, where the extra float registers do not repay their occupancy cost.
    constexpr bool kPreconvertInput = !kHalf2FastPath && SizeM >= 8;
    using InputArray = typename std::conditional<
        kPreconvertInput, float[SizeM], Scalar[SizeM]>::type;
    InputArray x_lane;
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
      const Scalar value = ldg(input + static_cast<int64_t>(m) * size_k + row0 + lane);
      if constexpr (kPreconvertInput) {
        x_lane[m] = ScalarTraits<Scalar>::to_float(value);
      } else {
        x_lane[m] = value;
      }
    }

    // Decode/compute weights once and reuse them for all 32 activations.
    // ColsPerLane==2 keeps the vector qweight/scale loads.  Small-M fp16 uses
    // a single __hfma2 per pair (decode latency bound), while larger M and all
    // bf16 fall back to fp32 weights/products so long-K reductions stay within
    // tolerance.
    if constexpr (ColsPerLane == 2) {
      if constexpr (kHalf2FastPath) {
        const AccScalar2 zero_scale2 = ScalarTraits<Scalar>::mul2(
            ScalarTraits<Scalar>::from_floats(static_cast<float>(zero[0]),
                                              static_cast<float>(zero[1])),
            scale2);
#pragma unroll
        for (int k = 0; k < 32; ++k) {
          int code0 = plane_code<Spec::kW0>(words[0], k);
          code0 |= plane_code<Spec::kW1>(words[0] + Spec::kW0, k) << Spec::kW0;
          int code1 = plane_code<Spec::kW0>(words[1], k);
          code1 |= plane_code<Spec::kW1>(words[1] + Spec::kW0, k) << Spec::kW0;
          if constexpr (Spec::kW2 > 0) {
            code0 |= plane_code<Spec::kW2>(words[0] + Spec::kW0 + Spec::kW1, k)
                    << (Spec::kW0 + Spec::kW1);
            code1 |= plane_code<Spec::kW2>(words[1] + Spec::kW0 + Spec::kW1, k)
                    << (Spec::kW0 + Spec::kW1);
          }
          const AccScalar2 code2 = ScalarTraits<Scalar>::from_floats(
              static_cast<float>(code0), static_cast<float>(code1));
          const AccScalar2 weight2 = ScalarTraits<Scalar>::fma2(
              code2, scale2, ScalarTraits<Scalar>::neg2(zero_scale2));
#pragma unroll
          for (int m = 0; m < SizeM; ++m) {
            const Scalar activation = ScalarTraits<Scalar>::shfl(x_lane[m], k);
            const AccScalar2 act2 = ScalarTraits<Scalar>::make2(activation);
            acc[m] = ScalarTraits<Scalar>::fma2(act2, weight2, acc[m]);
          }
        }
      } else {
        const float scale0 = ScalarTraits<Scalar>::low(scale2);
        const float scale1 = ScalarTraits<Scalar>::high(scale2);
#pragma unroll
        for (int k = 0; k < 32; ++k) {
          int code0 = plane_code<Spec::kW0>(words[0], k);
          code0 |= plane_code<Spec::kW1>(words[0] + Spec::kW0, k) << Spec::kW0;
          int code1 = plane_code<Spec::kW0>(words[1], k);
          code1 |= plane_code<Spec::kW1>(words[1] + Spec::kW0, k) << Spec::kW0;
          if constexpr (Spec::kW2 > 0) {
            code0 |= plane_code<Spec::kW2>(words[0] + Spec::kW0 + Spec::kW1, k)
                    << (Spec::kW0 + Spec::kW1);
            code1 |= plane_code<Spec::kW2>(words[1] + Spec::kW0 + Spec::kW1, k)
                    << (Spec::kW0 + Spec::kW1);
          }
          const float z0s = static_cast<float>(zero[0]) * scale0;
          const float z1s = static_cast<float>(zero[1]) * scale1;
          const float weight0 = __fmaf_rn(static_cast<float>(code0), scale0, -z0s);
          const float weight1 = __fmaf_rn(static_cast<float>(code1), scale1, -z1s);
#pragma unroll
          for (int m = 0; m < SizeM; ++m) {
            float activation;
            if constexpr (kPreconvertInput) {
              activation = __shfl_sync(0xffffffffu, x_lane[m], k);
            } else {
              activation = ScalarTraits<Scalar>::to_float(
                  ScalarTraits<Scalar>::shfl(x_lane[m], k));
            }
            acc[m][0] = __fmaf_rn(activation, weight0, acc[m][0]);
            acc[m][1] = __fmaf_rn(activation, weight1, acc[m][1]);
          }
        }
      }
    } else {
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        float weight[ColsPerLane];
#pragma unroll
        for (int c = 0; c < ColsPerLane; ++c) {
          int code = plane_code<Spec::kW0>(words[c], k);
          code |= plane_code<Spec::kW1>(words[c] + Spec::kW0, k) << Spec::kW0;
          if constexpr (Spec::kW2 > 0) {
            code |= plane_code<Spec::kW2>(words[c] + Spec::kW0 + Spec::kW1, k)
                    << (Spec::kW0 + Spec::kW1);
          }
          const float z_scale = static_cast<float>(zero[c]) * scale[c];
          weight[c] = __fmaf_rn(static_cast<float>(code), scale[c], -z_scale);
        }
#pragma unroll
        for (int m = 0; m < SizeM; ++m) {
          float activation;
          if constexpr (kPreconvertInput) {
            activation = __shfl_sync(0xffffffffu, x_lane[m], k);
          } else {
            activation = ScalarTraits<Scalar>::to_float(
                ScalarTraits<Scalar>::shfl(x_lane[m], k));
          }
#pragma unroll
          for (int c = 0; c < ColsPerLane; ++c) {
            acc[m][c] = __fmaf_rn(activation, weight[c], acc[m][c]);
          }
        }
      }
    }

    // Flush the k-block partial sum into fp32 shared memory and reset the
    // per-block accumulator so the running total stays in full precision.
    if constexpr (kHalf2FastPath) {
#pragma unroll
      for (int m = 0; m < SizeM; ++m) {
        partials[((warp * SizeM + m) * kPartialCols) + (lane * 2)] +=
            ScalarTraits<Scalar>::low(acc[m]);
        partials[((warp * SizeM + m) * kPartialCols) + (lane * 2 + 1)] +=
            ScalarTraits<Scalar>::high(acc[m]);
        acc[m] = AccScalar2();
      }
    } else {
#pragma unroll
      for (int m = 0; m < SizeM; ++m) {
#pragma unroll
        for (int c = 0; c < ColsPerLane; ++c) {
          partials[((warp * SizeM + m) * kPartialCols) + (lane * ColsPerLane + c)] += acc[m][c];
          acc[m][c] = 0.0f;
        }
      }
    }
  }

  __syncthreads();

  if (warp != 0) {
    return;
  }

  if (gridDim.y == 1) {
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
#pragma unroll
      for (int c = 0; c < ColsPerLane; ++c) {
        const int col = lane * ColsPerLane + c;
        float value = partials[((0 * SizeM + m) * kPartialCols) + col];
#pragma unroll
        for (int w = 1; w < Warps; ++w) {
          value += partials[((w * SizeM + m) * kPartialCols) + col];
        }
        output[static_cast<int64_t>(m) * size_n + n0 + c] = ScalarTraits<Scalar>::from_float(value);
      }
    }
    return;
  }

  // Workspace layout: [split_k, SizeM, N]; each split-K block owns one slice
  // so plain stores replace global atomics and no zero-init pass is needed.
#pragma unroll
  for (int m = 0; m < SizeM; ++m) {
#pragma unroll
    for (int c = 0; c < ColsPerLane; ++c) {
      const int col = lane * ColsPerLane + c;
      float value = partials[((0 * SizeM + m) * kPartialCols) + col];
#pragma unroll
      for (int w = 1; w < Warps; ++w) {
        value += partials[((w * SizeM + m) * kPartialCols) + col];
      }
      workspace[(static_cast<int64_t>(blockIdx.y) * SizeM + m) * size_n + n0 + c] = value;
    }
  }

  // Last split-K block for this column block sums the slices and converts to
  // the output dtype (release/acquire via threadfence + atomic counter).
  __threadfence();
  __shared__ int is_last;
  if (lane == 0) {
    is_last = (atomicAdd(counters + column_block, 1) == static_cast<int>(gridDim.y) - 1);
  }
  __syncwarp();
  if (is_last) {
#pragma unroll
    for (int m = 0; m < SizeM; ++m) {
#pragma unroll
      for (int c = 0; c < ColsPerLane; ++c) {
        float value = 0.0f;
        for (int y = 0; y < static_cast<int>(gridDim.y); ++y) {
          value += workspace[(static_cast<int64_t>(y) * SizeM + m) * size_n + n0 + c];
        }
        output[static_cast<int64_t>(m) * size_n + n0 + c] = ScalarTraits<Scalar>::from_float(value);
      }
    }
  }
}

template <typename Scalar, int Bits, int SizeM, int ColsPerLane, int Warps>
int launch_cols(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  // Size split-K from this instantiation's real occupancy so the grid fills
  // exactly the resident-block wave (a partial second wave runs alone and
  // stretches the whole launch).
  constexpr int kThreads = Warps * kWarpSize;
  const int column_blocks = size_n / (kWarpSize * ColsPerLane);
  // Dynamic shared memory for the partials reduction; must be requested before
  // occupancy/launch when it exceeds the default 48 KiB per block.
  constexpr int kPartialBytes =
      Warps * SizeM * (kWarpSize * ColsPerLane) * sizeof(float);
  auto kernel = pangolin_gemv_kernel<Scalar, Bits, SizeM, ColsPerLane, Warps>;
  // Occupancy is fixed per (instantiation, device); cache it so the decode
  // path does not pay a driver query on every launch. The shared-memory
  // attribute is set inside the same cache miss because it is a per-function,
  // per-device sticky property.
  constexpr int kMaxDevices = 64;
  static std::array<std::atomic<int>, kMaxDevices> occupancy_cache{};
  int device = 0;
  cudaGetDevice(&device);
  int blocks_per_sm = 0;
  if (device >= 0 && device < kMaxDevices) {
    blocks_per_sm = occupancy_cache[static_cast<size_t>(device)].load(std::memory_order_relaxed);
  }
  if (blocks_per_sm == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kPartialBytes));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kThreads, static_cast<size_t>(kPartialBytes));
    blocks_per_sm = std::max(1, blocks_per_sm);
    if (device >= 0 && device < kMaxDevices) {
      occupancy_cache[static_cast<size_t>(device)].store(blocks_per_sm, std::memory_order_relaxed);
    }
  }
  const int wave_slots = std::max(1, blocks_per_sm) * sm_count;
  const int wanted = std::max(1, wave_slots / column_blocks);
  // A split_k larger than num_k_blocks / Warps leaves warps with no k-blocks
  // to process, so clamp the effective cap per instantiation.
  const int max_split_by_warps = std::max(1, (size_k / 32) / Warps);
  const int effective_split_cap = std::min(split_cap, max_split_by_warps);
  // Size split-k to fill one resident wave of blocks per SM. Small batches can
  // launch additional waves when the k-block budget allows, but that is selected
  // by the Warps choice in launch_size_m rather than a blanket multiplier here.
  const int occupancy_split = std::min(effective_split_cap, wanted);
  int split_k = occupancy_split;
  // On local sm_80, the 4096-square 7-bit BF16 M=4 and FP16 M=8 templates are
  // faster with the smallest split that preserves the longest per-warp work.
  // Other shapes/dtypes regress when generalized, so keep this deliberately
  // narrow. For K=4096, 4 warps need 16 splits for two K-blocks per warp; the
  // occupancy-only result of 17 adds workspace/reduction work without
  // shortening that chain.
  constexpr bool kUseWorkBalancedSplit = Bits == 7 &&
      ((std::is_same_v<Scalar, nv_bfloat16> && SizeM == 4) ||
       (std::is_same_v<Scalar, half> && SizeM == 8));
  if constexpr (kUseWorkBalancedSplit) {
    if (size_k == 4096 && size_n == 4096) {
      const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
      if (properties->major == 8 && properties->minor == 0) {
        const int num_k_blocks = size_k / 32;
        const int blocks_per_warp =
            (num_k_blocks + occupancy_split * Warps - 1) / (occupancy_split * Warps);
        split_k = std::max(
            1, (num_k_blocks + blocks_per_warp * Warps - 1) / (blocks_per_warp * Warps));
      }
    }
  }
  if (dry_run) {
    return split_k;
  }
  const dim3 grid(column_blocks, split_k);
  kernel<<<grid, kThreads, kPartialBytes, stream>>>(
      reinterpret_cast<const Scalar*>(input.const_data_ptr()),
      qweight.const_data_ptr<int32_t>(),
      reinterpret_cast<const Scalar*>(scales.const_data_ptr()),
      qzeros.const_data_ptr<int32_t>(),
      g_idx.const_data_ptr<int32_t>(),
      reinterpret_cast<Scalar*>(output.mutable_data_ptr()),
      workspace,
      counters,
      size_k,
      size_n,
      num_groups);
  return split_k;
}

template <typename Scalar, int Bits, int SizeM>
int launch_size_m(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  // ColsPerLane=2 (8-byte vector loads) is the default for all M once N is
  // divisible by 64; otherwise scalar loads are used. This keeps register
  // pressure low and occupancy high on the bandwidth/ILP-sensitive decode path.
  //
  // Low-M (SizeM < 32) kernels use 4 warps per block for higher occupancy when
  // there are enough output columns to fill the device. Tall layers with
  // size_k >= 4 * size_n have many K-blocks and few column blocks, so the extra
  // warps per block reduce the per-warp work and improve parallelism.
  // M=32 keeps 8 warps per block because its larger register footprint already
  // limits occupancy and the extra warps hide latency better.
  constexpr int kDefaultWarps = (SizeM >= 32) ? 8 : 4;
  constexpr int kTallWarps = 8;
  const bool is_tall = static_cast<int64_t>(size_k) >= 4LL * static_cast<int64_t>(size_n);
  const bool use_tall_warps = (SizeM < 32) && is_tall;

  if (size_n % (kWarpSize * 2) == 0) {
    if (use_tall_warps) {
      return launch_cols<Scalar, Bits, SizeM, 2, kTallWarps>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    }
    return launch_cols<Scalar, Bits, SizeM, 2, kDefaultWarps>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters,
        size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
  }
  if (use_tall_warps) {
    return launch_cols<Scalar, Bits, SizeM, 1, kTallWarps>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters,
        size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
  }
  return launch_cols<Scalar, Bits, SizeM, 1, kDefaultWarps>(
      input, qweight, scales, qzeros, g_idx, output, workspace, counters,
      size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
}

template <typename Scalar, int Bits>
int launch_bits(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  switch (size_m) {
    case 1:
      return launch_size_m<Scalar, Bits, 1>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 2:
      return launch_size_m<Scalar, Bits, 2>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 3:
      return launch_size_m<Scalar, Bits, 3>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 4:
      return launch_size_m<Scalar, Bits, 4>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 5:
      return launch_size_m<Scalar, Bits, 5>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 6:
      return launch_size_m<Scalar, Bits, 6>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 7:
      return launch_size_m<Scalar, Bits, 7>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 8:
      return launch_size_m<Scalar, Bits, 8>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 16:
      return launch_size_m<Scalar, Bits, 16>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 32:
      return launch_size_m<Scalar, Bits, 32>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    default:
      TORCH_CHECK(false, "pangolin gemv supports 1..", kMaxM, " input rows, got ", size_m);
  }
}

template <typename Scalar>
int launch_scalar(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    const torch::Tensor& g_idx,
    torch::Tensor& output,
    float* workspace,
    int* counters,
    int64_t bits,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    int split_cap,
    int sm_count,
    cudaStream_t stream,
    bool dry_run = false) {
  switch (bits) {
    case 3:
      return launch_bits<Scalar, 3>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 5:
      return launch_bits<Scalar, 5>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 6:
      return launch_bits<Scalar, 6>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    case 7:
      return launch_bits<Scalar, 7>(
          input, qweight, scales, qzeros, g_idx, output, workspace, counters,
          size_m, size_k, size_n, num_groups, split_cap, sm_count, stream, dry_run);
    default:
      TORCH_CHECK(false, "pangolin gemv supports bits 3/5/6/7, got ", bits);
  }
}

}  // namespace pangolin

torch::Tensor pangolin_gemv_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    int64_t bits) {
  TORCH_CHECK(input.is_cuda(), "pangolin input must be CUDA");
  TORCH_CHECK(
      qweight.is_cuda() && scales.is_cuda() && qzeros.is_cuda() && g_idx.is_cuda(),
      "pangolin weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == qweight.device() && input.device() == scales.device() &&
          input.device() == qzeros.device() && input.device() == g_idx.device(),
      "pangolin tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "pangolin input must be FP16 or BF16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "pangolin qweight must be int32");
  TORCH_CHECK(qzeros.scalar_type() == at::kInt, "pangolin qzeros must be int32");
  TORCH_CHECK(g_idx.scalar_type() == at::kInt, "pangolin g_idx must be int32");
  TORCH_CHECK(scales.scalar_type() == input.scalar_type(), "pangolin scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2 && qzeros.dim() == 2 && g_idx.dim() == 1,
      "pangolin expects input[M,K], qweight/scales/qzeros 2D, g_idx 1D");
  TORCH_CHECK(
      input.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous() &&
          qzeros.is_contiguous() && g_idx.is_contiguous(),
      "pangolin tensors must be contiguous");
  TORCH_CHECK(bits == 3 || bits == 5 || bits == 6 || bits == 7, "pangolin supports bits 3/5/6/7");

  const int64_t size_m = input.size(0);
  const int64_t size_k = input.size(1);
  const int64_t size_n = scales.size(1);
  const int64_t num_groups = scales.size(0);
  constexpr std::array<int64_t, 10> kSupportedM = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32};
  TORCH_CHECK(
      std::find(kSupportedM.begin(), kSupportedM.end(), size_m) != kSupportedM.end(),
      "pangolin gemv supports M in {1,2,3,4,5,6,7,8,16,32}, got ", size_m);
  TORCH_CHECK(size_k % 32 == 0 && size_n % 32 == 0, "pangolin K and N must be divisible by 32");
  TORCH_CHECK(
      qweight.size(0) == (size_k / 32) * bits && qweight.size(1) == size_n,
      "pangolin qweight must have planar shape [K/32*bits, N]");
  TORCH_CHECK(
      qzeros.size(0) == num_groups && qzeros.size(1) == (size_n / 32) * bits,
      "pangolin qzeros must have planar shape [groups, N/32*bits]");
  TORCH_CHECK(g_idx.size(0) == size_k, "pangolin g_idx must have K entries");
  TORCH_CHECK(
      size_k <= std::numeric_limits<int>::max() && size_n <= std::numeric_limits<int>::max(),
      "pangolin dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());

  // Deep split-K supplies SM-level parallelism: the column blocks alone
  // underfill large devices at decode shapes. The exact split is chosen at
  // launch (per-instantiation occupancy); here only cap it so every warp
  // still owns whole k-blocks.
  const int num_k_blocks = static_cast<int>(size_k / 32);
  // Counters are sized for the finest column granularity (32 columns per block)
  // because the launch may pick ColsPerLane=1 for some SizeM/N combinations.
  const int column_blocks = static_cast<int>(size_n / pangolin::kBlockN);
  // 128 bounds the transient fp32 workspace; the occupancy-derived split
  // never usefully exceeds the number of 32-row k-blocks per warp.
  const int split_cap = std::min(128, std::max(1, num_k_blocks));
  const int sm_count = properties->multiProcessorCount;

  auto output = torch::empty({size_m, size_n}, input.options());

  // Sizing-only pass: compute the actual split_k for this instantiation so
  // the workspace is sized exactly. This also warms the per-device occupancy
  // cache and sets the dynamic-shared-memory attribute.
  int split_k = 1;
  if (input.scalar_type() == at::kHalf) {
    split_k = pangolin::launch_scalar<half>(
        input, qweight, scales, qzeros, g_idx, output, nullptr, nullptr, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_cap, sm_count, stream, true);
  } else {
    split_k = pangolin::launch_scalar<nv_bfloat16>(
        input, qweight, scales, qzeros, g_idx, output, nullptr, nullptr, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_cap, sm_count, stream, true);
  }

  // Split-K partials land in per-slice fp32 workspace (written, not
  // accumulated, so only the tiny counter tail needs zeroing) and the last
  // block reduces + converts in-kernel: one gemv launch, no zero/convert
  // kernels on the output tensor.
  torch::Tensor scratch;
  float* workspace = nullptr;
  int* counters = nullptr;
  if (split_k > 1) {
    const int64_t workspace_elems = static_cast<int64_t>(split_k) * size_m * size_n;
    scratch = torch::empty(
        {workspace_elems + column_blocks}, input.options().dtype(at::kFloat));
    workspace = scratch.mutable_data_ptr<float>();
    counters = reinterpret_cast<int*>(workspace + workspace_elems);
    C10_CUDA_CHECK(cudaMemsetAsync(counters, 0, sizeof(int) * column_blocks, stream));
  }

  if (input.scalar_type() == at::kHalf) {
    pangolin::launch_scalar<half>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_k, sm_count, stream, false);
  } else {
    pangolin::launch_scalar<nv_bfloat16>(
        input, qweight, scales, qzeros, g_idx, output, workspace, counters, bits,
        static_cast<int>(size_m), static_cast<int>(size_k), static_cast<int>(size_n),
        static_cast<int>(num_groups), split_k, sm_count, stream, false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
