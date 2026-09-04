// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Fused scale-first fast Walsh-Hadamard transform for QVQ inference.
//
// Replaces the Python butterfly (matmul_hadU_stable) with one kernel launch.
// Bitwise-parity contract (validated): the divisor is fp16(sqrtf(n)) and every
// butterfly add/sub is computed in float32 then rounded to the output dtype,
// exactly matching torch's elementwise half/bfloat16 semantics. Optional
// per-column pre-scale (SU), post-scale (SV), and bias are fused into the same
// launch with the same rounding sequence as the Python forward.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>
#include <limits>
#include <torch/library.h>
#include <torch/types.h>
#include <tuple>

namespace {

template <typename Scalar>
struct QvqHadamardTraits;

template <>
struct QvqHadamardTraits<half> {
  static __device__ __forceinline__ float to_float(half v) { return __half2float(v); }
  static __device__ __forceinline__ half from_float(float v) { return __float2half_rn(v); }
};

template <>
struct QvqHadamardTraits<nv_bfloat16> {
  static __device__ __forceinline__ float to_float(nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
};

template <>
struct QvqHadamardTraits<float> {
  static __device__ __forceinline__ float to_float(float v) { return v; }
  static __device__ __forceinline__ float from_float(float v) { return v; }
};

constexpr int kHadamardThreads = 1024;
constexpr int kHadamardPairMultiblockN = 8192;
constexpr int kHadamardPairMultiblockTile = 256;
constexpr int kHadamardPairMultiblockTiles =
    kHadamardPairMultiblockN / kHadamardPairMultiblockTile;
constexpr int kHadamardPairMultiblockHighThreads = 64;
constexpr int kHadamardPairMultiblockHalf2LowValues =
    kHadamardPairMultiblockTile / 2;
constexpr int kHadamardPairMultiblockHalf2LowThreads =
    kHadamardPairMultiblockTile;
constexpr int kHadamardPairMultiblockHalf2HighThreads = 32;
constexpr int kHadamardOrderedSplitN = 2048;
constexpr int kHadamardOrderedSplitTile = 256;
constexpr int kHadamardOrderedSplitTiles =
    kHadamardOrderedSplitN / kHadamardOrderedSplitTile;
constexpr int kHadamardOrderedSplitHighThreads = 64;
constexpr int kHadamardInputMultiblockN = 2048;
constexpr int kHadamardInputMultiblockTile = 256;
constexpr int kHadamardInputMultiblockTiles =
    kHadamardInputMultiblockN / kHadamardInputMultiblockTile;
constexpr int kHadamardInputMultiblockHalf2Values =
    kHadamardInputMultiblockTile / 2;
constexpr int kHadamardInputMultiblockLowThreads =
    kHadamardInputMultiblockTile;
constexpr int kHadamardInputMultiblockHighThreads = 32;
constexpr int kQwenCompositeN = 5120;
constexpr int kQwenCompositeBase = 40;
constexpr int kQwenCompositeLowN = kQwenCompositeN / kQwenCompositeBase;

__device__ __forceinline__ float round_fp16_unless_overflow(float value) {
  const float narrowed = __half2float(__float2half_rn(value));
  return isfinite(narrowed) ? narrowed : value;
}

// Exact multiblock form of the H100 A41/R0 shared N=2048 input transform.
// The first eight ascending butterfly stages are independent inside each
// contiguous 256-column tile.  The low grid therefore exposes eight CTAs per
// logical row.  It retains the range-safe mode-2 order exactly:
//
//   half(x * SU), unless that narrowing overflows -> divide by half(sqrt(N))
//   -> half -> butterfly bits 1..128, rounding to half after every stage.
//
// Adjacent scalar columns are packed only after their bit-1 sum/difference.
// Bits 2..128 then use native half2 add/sub without reassociation.
__global__ void qvq_hadamard_input_fp16_padded_multiblock_low_kernel(
    const half* __restrict__ input,
    const half* __restrict__ pre_scale,
    half* __restrict__ workspace) {
  __shared__ half2 buf[kHadamardInputMultiblockHalf2Values];
  const int row = static_cast<int>(blockIdx.y);
  const int tile = static_cast<int>(blockIdx.x);
  const int local = static_cast<int>(threadIdx.x);
  const int column = tile * kHadamardInputMultiblockTile + local;
  const int64_t offset =
      static_cast<int64_t>(row) * kHadamardInputMultiblockN + column;
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kHadamardInputMultiblockN))));

  float value = __half2float(input[offset]) * __half2float(pre_scale[column]);
  value = round_fp16_unless_overflow(value);
  const half initial = __float2half_rn(value / divisor);
  const unsigned int peer_bits = __shfl_xor_sync(
      0xffffffffu, static_cast<unsigned int>(__half_as_ushort(initial)), 1);
  if ((local & 1) == 0) {
    constexpr unsigned int kEvenLaneMask = 0x55555555u;
    const int local_pair = local >> 1;
    const half peer = __ushort_as_half(static_cast<unsigned short>(peer_bits));
    const half2 pair = __halves2half2(initial, peer);
    const half2 swapped = __lowhigh2highlow(pair);
    const half2 sum = __hadd2(pair, swapped);
    const half2 difference = __hsub2(pair, swapped);
    half2 packed = __lows2half2(sum, difference);

#pragma unroll
    for (int bit = 1; bit < 16; bit <<= 1) {
      union Half2Bits {
        half2 value;
        unsigned int bits;
      } packed_bits, packed_peer;
      packed_bits.value = packed;
      packed_peer.bits = __shfl_xor_sync(
          kEvenLaneMask, packed_bits.bits, bit * 2);
      packed = (local_pair & bit) == 0
          ? __hadd2(packed, packed_peer.value)
          : __hsub2(packed_peer.value, packed);
    }
    buf[local_pair] = packed;
  }
  __syncthreads();

#pragma unroll
  for (int bit = 16; bit < kHadamardInputMultiblockHalf2Values; bit <<= 1) {
    if (local < kHadamardInputMultiblockHalf2Values) {
      const int peer = local ^ bit;
      if (local < peer) {
        const half2 a = buf[local];
        const half2 b = buf[peer];
        buf[local] = __hadd2(a, b);
        buf[peer] = __hsub2(a, b);
      }
    }
    __syncthreads();
  }

  if (local < kHadamardInputMultiblockHalf2Values) {
    const int64_t pair_offset =
        static_cast<int64_t>(row) * kHadamardInputMultiblockN +
        tile * kHadamardInputMultiblockTile + local * 2;
    *reinterpret_cast<half2*>(workspace + pair_offset) = buf[local];
  }
}

// Each high-stage thread owns two adjacent within-tile columns across all
// eight tiles.  The remaining tile-index stages correspond exactly to scalar
// butterfly bits 256, 512, and 1024.  Valid rows are updated in place only
// after all eight inputs are resident in registers; padded rows are written
// as exact zero without reading uninitialized workspace.
__global__ void qvq_hadamard_input_fp16_padded_multiblock_high_kernel(
    half* workspace,
    int logical_rows) {
  const int row = static_cast<int>(blockIdx.y);
  const int local_pair =
      static_cast<int>(blockIdx.x) * kHadamardInputMultiblockHighThreads +
      static_cast<int>(threadIdx.x);
  const int local = local_pair * 2;
  if (row >= logical_rows) {
    const half2 zero = __float2half2_rn(0.0f);
#pragma unroll
    for (int tile = 0; tile < kHadamardInputMultiblockTiles; ++tile) {
      const int column = tile * kHadamardInputMultiblockTile + local;
      *reinterpret_cast<half2*>(
          workspace + static_cast<int64_t>(row) * kHadamardInputMultiblockN +
          column) = zero;
    }
    return;
  }

  half2 values[kHadamardInputMultiblockTiles];
#pragma unroll
  for (int tile = 0; tile < kHadamardInputMultiblockTiles; ++tile) {
    const int column = tile * kHadamardInputMultiblockTile + local;
    values[tile] = *reinterpret_cast<const half2*>(
        workspace + static_cast<int64_t>(row) * kHadamardInputMultiblockN +
        column);
  }
#pragma unroll
  for (int bit = 1; bit < kHadamardInputMultiblockTiles; bit <<= 1) {
#pragma unroll
    for (int tile = 0; tile < kHadamardInputMultiblockTiles; ++tile) {
      const int peer = tile ^ bit;
      if (tile < peer) {
        const half2 a = values[tile];
        const half2 b = values[peer];
        values[tile] = __hadd2(a, b);
        values[peer] = __hsub2(a, b);
      }
    }
  }
#pragma unroll
  for (int tile = 0; tile < kHadamardInputMultiblockTiles; ++tile) {
    const int column = tile * kHadamardInputMultiblockTile + local;
    *reinterpret_cast<half2*>(
        workspace + static_cast<int64_t>(row) * kHadamardInputMultiblockN +
        column) = values[tile];
  }
}

// One block per row; the row lives in dynamic shared memory. log2n must be >= 1
// (n >= 2) and n * sizeof(Scalar) must fit the device's dynamic shared limit.
template <typename Scalar, typename OutputScalar = Scalar, bool PadTo16 = false>
__global__ void __launch_bounds__(kHadamardThreads) qvq_hadamard_kernel(
    const Scalar* __restrict__ input,
    OutputScalar* __restrict__ output,
    const Scalar* __restrict__ pre_scale,   // optional, [n]
    const Scalar* __restrict__ post_scale,  // optional, [n]
    const Scalar* __restrict__ bias,        // optional, [n]
    int n,
    int scale_mode,
    int logical_rows) {
  // scale_mode 0 mirrors matmul_hadU_stable (normalize FIRST, fp16(sqrtf(n))
  // divisor); scale_mode 1 mirrors matmul_hadU (normalize LAST, float sqrtf(n)
  // divisor). Mode 2 is the range-safe QVQ input epilogue: fuse x*SU/sqrt(n)
  // in FP32 before the first narrow store so an algebraically finite transform
  // cannot overflow solely because x*SU was rounded before normalization.
  // Modes 3/4 consume an FP32 inner result while reproducing mode 0/1's FP16
  // rounding whenever that rounding is finite. Only an overflowing narrowing
  // stays FP32, so ordinary model outputs remain bitwise unchanged.
  extern __shared__ char smem_raw[];
  // Padded layout: element i lives at p(i) = i + (i >> 5) so the butterfly's
  // (i, i^bit) pair reads land in distinct shared banks (2-way conflicts were
  // ~half the kernel's latency). Values are unchanged - purely physical.
  auto p = [](int i) { return i + (i >> 5); };
  Scalar* buf = reinterpret_cast<Scalar*>(smem_raw);
  const int row = static_cast<int>(blockIdx.x);
  const Scalar* in_row = input + static_cast<int64_t>(row) * n;
  OutputScalar* out_row = output + static_cast<int64_t>(row) * n;

  // Normalization divisor semantics mirror the two Python references bitwise:
  //   mode 0 (stable): divide by fp16(sqrtf(n)) with IEEE fp32 division (the
  //     divisor comes from an on-device X.new_tensor(...) -> GPU tensor path);
  //   mode 1 (end-scale): multiply by the fp32-rounded reciprocal 1/sqrtf(n)
  //     (the divisor comes from torch.tensor(...) on CPU -> scalar path, which
  //     torch evaluates as x * fp32(1/d)).
  const bool emulate_fp16 = scale_mode == 3 || scale_mode == 4;
  const bool normalize_first = scale_mode == 0 || scale_mode == 2 || scale_mode == 3;
  const float divisor = normalize_first
      ? (emulate_fp16
          ? __half2float(__float2half_rn(sqrtf(static_cast<float>(n))))
          : QvqHadamardTraits<Scalar>::to_float(
              QvqHadamardTraits<Scalar>::from_float(sqrtf(static_cast<float>(n)))))
      : sqrtf(static_cast<float>(n));
  const float reciprocal = 1.0f / sqrtf(static_cast<float>(n));

  // Modes 3/4 execute the entire butterfly in FP32 storage but round every
  // historical FP16 operation back to FP16 whenever it remains finite. This
  // is stronger than inspecting only the GEMV output: overflow can first
  // occur in a later butterfly, SV multiply, or bias add. Keeping the value in
  // FP32 only at that exact stage preserves all ordinary FP16 results while
  // covering every intermediate range edge.
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = QvqHadamardTraits<Scalar>::to_float(in_row[i]);
    if (emulate_fp16) {
      value = round_fp16_unless_overflow(value);
    }
    if (pre_scale != nullptr) {
      const float su = QvqHadamardTraits<Scalar>::to_float(pre_scale[i]);
      value = QvqHadamardTraits<Scalar>::to_float(in_row[i]) * su;
      if (scale_mode == 2) {
        value = round_fp16_unless_overflow(value);
      } else {
        value = QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
      }
    }
    if (normalize_first) {
      value /= divisor;
      if (emulate_fp16) {
        value = round_fp16_unless_overflow(value);
      }
    }
    buf[p(i)] = QvqHadamardTraits<Scalar>::from_float(value);
  }
  __syncthreads();

  // Butterfly: 1, 2, ..., n/2 (ascending), mirroring matmul_hadU's view recursion;
  // each pair (i, i^bit) once, float add/sub + round.
  for (int bit = 1; bit < n; bit <<= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ bit;
      if (i < j) {
        const float a = QvqHadamardTraits<Scalar>::to_float(buf[p(i)]);
        const float b = QvqHadamardTraits<Scalar>::to_float(buf[p(j)]);
        float sum = a + b;
        float difference = a - b;
        if (emulate_fp16) {
          sum = round_fp16_unless_overflow(sum);
          difference = round_fp16_unless_overflow(difference);
        }
        buf[p(i)] = QvqHadamardTraits<Scalar>::from_float(sum);
        buf[p(j)] = QvqHadamardTraits<Scalar>::from_float(difference);
      }
    }
    __syncthreads();
  }

  // Epilogue: late normalization in scale_mode 1, then optional SV post-scale
  // and bias add, each matching Python rounding.
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = QvqHadamardTraits<Scalar>::to_float(buf[p(i)]);
    if (scale_mode == 1 || scale_mode == 4) {
      // matmul_hadU rounds the division result back to fp16 before the caller
      // applies SV/bias, so round here before the epilogue scales.
      value *= reciprocal;
      value = emulate_fp16
          ? round_fp16_unless_overflow(value)
          : QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
    }
    if (post_scale != nullptr) {
      value *= QvqHadamardTraits<Scalar>::to_float(post_scale[i]);
      value = emulate_fp16
          ? round_fp16_unless_overflow(value)
          : QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
    }
    if (bias != nullptr) {
      value += QvqHadamardTraits<Scalar>::to_float(bias[i]);
      value = emulate_fp16
          ? round_fp16_unless_overflow(value)
          : QvqHadamardTraits<Scalar>::to_float(QvqHadamardTraits<Scalar>::from_float(value));
    }
    out_row[i] = QvqHadamardTraits<OutputScalar>::from_float(value);
  }

  // Hopper P32 consumes exactly sixteen rows.  A padded specialization lets
  // the logical transform blocks also own the disjoint zero tail, deleting a
  // separate allocation/fill/copy boundary.  No padded row participates in a
  // butterfly and every valid-row instruction above is unchanged.
  if constexpr (PadTo16) {
    for (int padded_row = logical_rows + row; padded_row < 16;
         padded_row += logical_rows) {
      OutputScalar* padded = output + static_cast<int64_t>(padded_row) * n;
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        padded[i] = QvqHadamardTraits<OutputScalar>::from_float(0.0f);
      }
    }
  }
}

// Gate and up own independent output transforms with the same geometry.  A
// single grid lets both sets of row blocks run concurrently instead of
// serializing two tiny launches.  The inputs and shared butterfly remain FP32,
// while every historical FP16 operation is emulated exactly as modes 3/4 of
// qvq_hadamard_kernel<float>.  The final FP16 store is the same rounding step
// as the production caller's former output.to(torch.float16) kernels.
//
// Specialization budget: this is one fixed FP32->FP16 device kernel.  Rates,
// row counts, widths, bias presence, and normalization mode stay runtime data.
__global__ void __launch_bounds__(kHadamardThreads) qvq_hadamard_pair_fp32_to_fp16_kernel(
    const float* __restrict__ input0,
    const float* __restrict__ input1,
    half* __restrict__ output0,
    half* __restrict__ output1,
    const float* __restrict__ post_scale0,
    const float* __restrict__ post_scale1,
    const float* __restrict__ bias0,
    const float* __restrict__ bias1,
    int n,
    int rows,
    int scale_mode) {
  extern __shared__ char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  const bool second = static_cast<int>(blockIdx.x) >= rows;
  const int row = static_cast<int>(blockIdx.x) - (second ? rows : 0);
  const float* input = second ? input1 : input0;
  half* output = second ? output1 : output0;
  const float* post_scale = second ? post_scale1 : post_scale0;
  const float* bias = second ? bias1 : bias0;
  const float* in_row = input + static_cast<int64_t>(row) * n;
  half* out_row = output + static_cast<int64_t>(row) * n;
  auto p = [](int i) { return i + (i >> 5); };

  const bool normalize_first = scale_mode == 3;
  const float divisor = __half2float(__float2half_rn(sqrtf(static_cast<float>(n))));
  const float reciprocal = 1.0f / sqrtf(static_cast<float>(n));

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = round_fp16_unless_overflow(in_row[i]);
    if (normalize_first) {
      value = round_fp16_unless_overflow(value / divisor);
    }
    buf[p(i)] = value;
  }
  __syncthreads();

  for (int bit = 1; bit < n; bit <<= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ bit;
      if (i < j) {
        const float a = buf[p(i)];
        const float b = buf[p(j)];
        buf[p(i)] = round_fp16_unless_overflow(a + b);
        buf[p(j)] = round_fp16_unless_overflow(a - b);
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = buf[p(i)];
    if (!normalize_first) {
      value = round_fp16_unless_overflow(value * reciprocal);
    }
    if (post_scale != nullptr) {
      value = round_fp16_unless_overflow(value * post_scale[i]);
    }
    if (bias != nullptr) {
      value = round_fp16_unless_overflow(value + bias[i]);
    }
    out_row[i] = __float2half_rn(value);
  }
}

// Llama 3.2 1B down projection specialization.  The Hopper tensor-core
// kernel produces sixteen disjoint [16, 2048] FP32 split planes.  Folding the
// deterministic left-to-right reduction into output recovery deletes the
// standalone reducer launch and its intermediate [16, 2048] FP32 write/read.
// The accumulation starts at +0 exactly like the fixed split reducer, and all
// subsequent narrowing, butterfly, scale, bias, and final-store operations
// retain qvq_hadamard_kernel<float, half>'s ordering.
__global__ void __launch_bounds__(kHadamardThreads)
qvq_hadamard_ordered_split16_fp32_to_fp16_kernel(
    const float* __restrict__ partial_input,
    half* __restrict__ output,
    const float* __restrict__ post_scale,
    const float* __restrict__ bias,
    int n,
    int logical_rows,
    int scale_mode) {
  extern __shared__ char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  auto p = [](int i) { return i + (i >> 5); };
  const int row = static_cast<int>(blockIdx.x);
  half* out_row = output + static_cast<int64_t>(row) * n;

  const bool normalize_first = scale_mode == 3;
  const float divisor = __half2float(__float2half_rn(sqrtf(static_cast<float>(n))));
  const float reciprocal = 1.0f / sqrtf(static_cast<float>(n));
  constexpr int kSplitCount = 16;
  constexpr int kPhysicalRows = 16;
  const int64_t split_stride = static_cast<int64_t>(kPhysicalRows) * n;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int split = 0; split < kSplitCount; ++split) {
      value += partial_input[
          static_cast<int64_t>(split) * split_stride +
          static_cast<int64_t>(row) * n + i];
    }
    value = round_fp16_unless_overflow(value);
    if (normalize_first) {
      value = round_fp16_unless_overflow(value / divisor);
    }
    buf[p(i)] = value;
  }
  __syncthreads();

  for (int bit = 1; bit < n; bit <<= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ bit;
      if (i < j) {
        const float a = buf[p(i)];
        const float b = buf[p(j)];
        buf[p(i)] = round_fp16_unless_overflow(a + b);
        buf[p(j)] = round_fp16_unless_overflow(a - b);
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float value = buf[p(i)];
    if (!normalize_first) {
      value = round_fp16_unless_overflow(value * reciprocal);
    }
    value = round_fp16_unless_overflow(value * post_scale[i]);
    if (bias != nullptr) {
      value = round_fp16_unless_overflow(value + bias[i]);
    }
    out_row[i] = __float2half_rn(value);
  }
}

// Exact two-stage factorization of the fused split-16 N=2048 down recovery.
// The low kernel owns one contiguous 256-column tile per CTA. It preserves the
// fixed reducer's +0, P0, ..., P15 order, performs the historical initial FP16
// narrowing/normalization, and then executes butterfly bits 1..128. No value
// outside the tile is needed before bit 256, so all eight tiles are independent.
__global__ void qvq_hadamard_ordered_split16_fp32_to_fp16_multiblock_low_kernel(
    const float* __restrict__ partial_input,
    float* __restrict__ workspace,
    int scale_mode) {
  __shared__ float buf[
      kHadamardOrderedSplitTile + kHadamardOrderedSplitTile / 32];
  auto p = [](int i) { return i + (i >> 5); };
  const int row = static_cast<int>(blockIdx.y);
  const int tile = static_cast<int>(blockIdx.x);
  const int local = static_cast<int>(threadIdx.x);
  const int column = tile * kHadamardOrderedSplitTile + local;
  constexpr int kSplitCount = 16;
  constexpr int kPhysicalRows = 16;
  constexpr int64_t kSplitStride =
      static_cast<int64_t>(kPhysicalRows) * kHadamardOrderedSplitN;

  float value = 0.0f;
#pragma unroll
  for (int split = 0; split < kSplitCount; ++split) {
    value += partial_input[
        static_cast<int64_t>(split) * kSplitStride +
        static_cast<int64_t>(row) * kHadamardOrderedSplitN + column];
  }
  value = round_fp16_unless_overflow(value);
  if (scale_mode == 3) {
    const float divisor = __half2float(
        __float2half_rn(sqrtf(static_cast<float>(kHadamardOrderedSplitN))));
    value = round_fp16_unless_overflow(value / divisor);
  }
  buf[p(local)] = value;
  __syncthreads();

#pragma unroll
  for (int bit = 1; bit < kHadamardOrderedSplitTile; bit <<= 1) {
    const int peer = local ^ bit;
    if (local < peer) {
      const float a = buf[p(local)];
      const float b = buf[p(peer)];
      buf[p(local)] = round_fp16_unless_overflow(a + b);
      buf[p(peer)] = round_fp16_unless_overflow(a - b);
    }
    __syncthreads();
  }

  workspace[
      static_cast<int64_t>(row) * kHadamardOrderedSplitN + column] =
      buf[p(local)];
}

// The high kernel keeps one within-tile column from all eight tiles in each
// thread, then executes butterfly bits 256, 512, and 1024 in the original
// ascending order. The FP32 workspace introduces no rounding boundary; scale,
// bias, and the final FP16 store remain identical to the one-block reference.
__global__ void qvq_hadamard_ordered_split16_fp32_to_fp16_multiblock_high_kernel(
    const float* __restrict__ workspace,
    half* __restrict__ output,
    const float* __restrict__ post_scale,
    const float* __restrict__ bias,
    int scale_mode) {
  const int row = static_cast<int>(blockIdx.y);
  const int local = static_cast<int>(blockIdx.x) *
          kHadamardOrderedSplitHighThreads +
      static_cast<int>(threadIdx.x);
  float values[kHadamardOrderedSplitTiles];

#pragma unroll
  for (int tile = 0; tile < kHadamardOrderedSplitTiles; ++tile) {
    const int column = tile * kHadamardOrderedSplitTile + local;
    values[tile] = workspace[
        static_cast<int64_t>(row) * kHadamardOrderedSplitN + column];
  }

#pragma unroll
  for (int bit = 1; bit < kHadamardOrderedSplitTiles; bit <<= 1) {
#pragma unroll
    for (int tile = 0; tile < kHadamardOrderedSplitTiles; ++tile) {
      const int peer = tile ^ bit;
      if (tile < peer) {
        const float a = values[tile];
        const float b = values[peer];
        values[tile] = round_fp16_unless_overflow(a + b);
        values[peer] = round_fp16_unless_overflow(a - b);
      }
    }
  }

  const float reciprocal =
      1.0f / sqrtf(static_cast<float>(kHadamardOrderedSplitN));
#pragma unroll
  for (int tile = 0; tile < kHadamardOrderedSplitTiles; ++tile) {
    const int column = tile * kHadamardOrderedSplitTile + local;
    float value = values[tile];
    if (scale_mode == 4) {
      value = round_fp16_unless_overflow(value * reciprocal);
    }
    value = round_fp16_unless_overflow(value * post_scale[column]);
    if (bias != nullptr) {
      value = round_fp16_unless_overflow(value + bias[column]);
    }
    output[static_cast<int64_t>(row) * kHadamardOrderedSplitN + column] =
        __float2half_rn(value);
  }
}

// Exact two-stage factorization of the N=8192 paired recovery transform.
//
// The ordinary kernel executes butterfly bits 1..4096 in ascending order in
// one CTA.  The first eight bits never cross a contiguous 256-value tile, so
// stage one assigns all 32 independent tiles to separate CTAs.  Stage two
// keeps one within-tile column in a thread and executes the remaining five
// tile-index bits in registers.  The global workspace stores FP32 values and
// therefore introduces no new rounding boundary between bits 128 and 256.
// Every call to round_fp16_unless_overflow remains in the same mathematical
// location as the single-CTA kernel.
//
// Specialization budget: two fixed N=8192 FP32->FP16 device kernels.  Rates,
// rows, bias presence, and normalization mode remain runtime data.  The
// separate public operator keeps this experimental launch geometry out of the
// production dispatch until H100 correctness and latency promotion gates pass.
__global__ void qvq_hadamard_pair_fp32_to_fp16_multiblock_low_kernel(
    const float* __restrict__ input0,
    const float* __restrict__ input1,
    float* __restrict__ workspace,
    int rows,
    int scale_mode) {
  __shared__ float buf[kHadamardPairMultiblockTile + kHadamardPairMultiblockTile / 32];
  const int projection_row = static_cast<int>(blockIdx.y);
  const bool second = projection_row >= rows;
  const int row = projection_row - (second ? rows : 0);
  const float* input = second ? input1 : input0;
  const int tile = static_cast<int>(blockIdx.x);
  const int local = static_cast<int>(threadIdx.x);
  const int column = tile * kHadamardPairMultiblockTile + local;
  auto p = [](int i) { return i + (i >> 5); };

  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kHadamardPairMultiblockN))));
  float value = round_fp16_unless_overflow(
      input[static_cast<int64_t>(row) * kHadamardPairMultiblockN + column]);
  if (scale_mode == 3) {
    value = round_fp16_unless_overflow(value / divisor);
  }
  buf[p(local)] = value;
  __syncthreads();

#pragma unroll
  for (int bit = 1; bit < kHadamardPairMultiblockTile; bit <<= 1) {
    const int peer = local ^ bit;
    if (local < peer) {
      const float a = buf[p(local)];
      const float b = buf[p(peer)];
      buf[p(local)] = round_fp16_unless_overflow(a + b);
      buf[p(peer)] = round_fp16_unless_overflow(a - b);
    }
    __syncthreads();
  }

  workspace[static_cast<int64_t>(projection_row) * kHadamardPairMultiblockN + column] =
      buf[p(local)];
}

// Exact warp-local form of the first eight N=8192 recovery stages. The first
// five butterfly bits never cross a warp, so each lane computes its own sum or
// lower-minus-upper difference from an XOR-shuffled peer and retains the
// historical round_fp16_unless_overflow boundary. Only bits 32, 64, and 128
// use shared memory and CTA barriers. Values stay FP32 throughout so the
// existing overflow-preserving contract is unchanged.
__global__ void qvq_hadamard_pair_fp32_to_fp16_multiblock_warp_low_kernel(
    const float* __restrict__ input0,
    const float* __restrict__ input1,
    float* __restrict__ workspace,
    int rows,
    int scale_mode) {
  __shared__ float buf[
      kHadamardPairMultiblockTile + kHadamardPairMultiblockTile / 32];
  const int projection_row = static_cast<int>(blockIdx.y);
  const bool second = projection_row >= rows;
  const int row = projection_row - (second ? rows : 0);
  const float* input = second ? input1 : input0;
  const int tile = static_cast<int>(blockIdx.x);
  const int local = static_cast<int>(threadIdx.x);
  const int column = tile * kHadamardPairMultiblockTile + local;
  auto p = [](int i) { return i + (i >> 5); };

  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kHadamardPairMultiblockN))));
  float value = round_fp16_unless_overflow(
      input[static_cast<int64_t>(row) * kHadamardPairMultiblockN + column]);
  if (scale_mode == 3) {
    value = round_fp16_unless_overflow(value / divisor);
  }

#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
    const float peer = __shfl_xor_sync(0xffffffffu, value, bit);
    value = (local & bit) == 0
        ? round_fp16_unless_overflow(value + peer)
        : round_fp16_unless_overflow(peer - value);
  }
  buf[p(local)] = value;
  __syncthreads();

#pragma unroll
  for (int bit = 32; bit < kHadamardPairMultiblockTile; bit <<= 1) {
    const int peer = local ^ bit;
    if (local < peer) {
      const float a = buf[p(local)];
      const float b = buf[p(peer)];
      buf[p(local)] = round_fp16_unless_overflow(a + b);
      buf[p(peer)] = round_fp16_unless_overflow(a - b);
    }
    __syncthreads();
  }

  workspace[
      static_cast<int64_t>(projection_row) * kHadamardPairMultiblockN + column] =
      buf[p(local)];
}

__global__ void qvq_hadamard_pair_fp32_to_fp16_multiblock_high_kernel(
    const float* __restrict__ workspace,
    half* __restrict__ output0,
    half* __restrict__ output1,
    const float* __restrict__ post_scale0,
    const float* __restrict__ post_scale1,
    const float* __restrict__ bias0,
    const float* __restrict__ bias1,
    int rows,
    int scale_mode) {
  const int projection_row = static_cast<int>(blockIdx.y);
  const bool second = projection_row >= rows;
  const int row = projection_row - (second ? rows : 0);
  half* output = second ? output1 : output0;
  const float* post_scale = second ? post_scale1 : post_scale0;
  const float* bias = second ? bias1 : bias0;
  const int local = static_cast<int>(blockIdx.x) * kHadamardPairMultiblockHighThreads +
      static_cast<int>(threadIdx.x);

  float values[kHadamardPairMultiblockTiles];
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const int column = tile * kHadamardPairMultiblockTile + local;
    values[tile] = workspace[
        static_cast<int64_t>(projection_row) * kHadamardPairMultiblockN + column];
  }

#pragma unroll
  for (int bit = 1; bit < kHadamardPairMultiblockTiles; bit <<= 1) {
#pragma unroll
    for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
      const int peer = tile ^ bit;
      if (tile < peer) {
        const float a = values[tile];
        const float b = values[peer];
        values[tile] = round_fp16_unless_overflow(a + b);
        values[peer] = round_fp16_unless_overflow(a - b);
      }
    }
  }

  const float reciprocal = 1.0f /
      sqrtf(static_cast<float>(kHadamardPairMultiblockN));
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const int column = tile * kHadamardPairMultiblockTile + local;
    float value = values[tile];
    if (scale_mode == 4) {
      value = round_fp16_unless_overflow(value * reciprocal);
    }
    value = round_fp16_unless_overflow(value * post_scale[column]);
    if (bias != nullptr) {
      value = round_fp16_unless_overflow(value + bias[column]);
    }
    output[static_cast<int64_t>(row) * kHadamardPairMultiblockN + column] =
        __float2half_rn(value);
  }
}

// Consume the already-activated FP16 gate and FP16 up output, reproduce the
// standalone FP16 multiply rounding, then execute the down projection's
// range-safe SU -> Hadamard input transform.  This removes the materialized
// SwiGLU product and one launch without changing SiLU itself or any rounding
// boundary.  One fixed FP16 specialization serves every legal row count and
// transform width.
__global__ void __launch_bounds__(kHadamardThreads) qvq_swiglu_precondition_kernel(
    const half* __restrict__ activated_gate,
    const half* __restrict__ up,
    const half* __restrict__ pre_scale,
    half* __restrict__ output,
    int n) {
  extern __shared__ char smem_raw[];
  half* buf = reinterpret_cast<half*>(smem_raw);
  const int row = static_cast<int>(blockIdx.x);
  const half* gate_row = activated_gate + static_cast<int64_t>(row) * n;
  const half* up_row = up + static_cast<int64_t>(row) * n;
  half* out_row = output + static_cast<int64_t>(row) * n;
  auto p = [](int i) { return i + (i >> 5); };
  const float divisor = __half2float(__float2half_rn(sqrtf(static_cast<float>(n))));

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const half product_half = __float2half_rn(__half2float(gate_row[i]) * __half2float(up_row[i]));
    float value = __half2float(product_half) * __half2float(pre_scale[i]);
    value = round_fp16_unless_overflow(value);
    buf[p(i)] = __float2half_rn(value / divisor);
  }
  __syncthreads();

  for (int bit = 1; bit < n; bit <<= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ bit;
      if (i < j) {
        const float a = __half2float(buf[p(i)]);
        const float b = __half2float(buf[p(j)]);
        buf[p(i)] = __float2half_rn(a + b);
        buf[p(j)] = __float2half_rn(a - b);
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out_row[i] = buf[p(i)];
  }
}

// PyTorch's FP16 SiLU CUDA path evaluates the activation in FP32 as
// x / (1 + exp(-x)) and rounds the result once to FP16.  Keep that exact
// division form: x * sigmoid(x) differs by one FP16 ULP for at least one
// finite input value on Hopper.  An exhaustive 63,488-value finite-FP16 test
// guards this boundary before the product below consumes the rounded result.
__device__ __forceinline__ half qvq_silu_fp16(half gate) {
  const float value = __half2float(gate);
  return __float2half_rn(value / (1.0f + expf(-value)));
}

// Qwen3.8's 17*1024 intermediate axis deliberately folds both surrounding
// Hadamards.  Consume the two FP32 P32 results without materializing the
// recovered gate/up, SiLU output, product, or scaled down input.  Explicit
// round-to-nearest operations preserve the separate PyTorch-kernel contract:
//
//   fp32 inner * fp16-rounded SV [+ fp16-rounded bias]
//     -> fp16 -> fp16 SiLU -> fp16 product -> fp16 down-SU product.
//
// The M16-padded output is cleared by the launcher before logical rows are
// written, so the operation remains directly consumable by Hopper P32.
template <bool HasGateBias, bool HasUpBias>
__global__ void qvq_folded_swiglu_precondition_fp32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ gate_scale,
    const float* __restrict__ up_scale,
    const float* __restrict__ gate_bias,
    const float* __restrict__ up_bias,
    const half* __restrict__ down_scale,
    half* __restrict__ output,
    int64_t logical_values,
    int n) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < logical_values;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int column = static_cast<int>(index % n);
    float gate_value = __fmul_rn(gate[index], gate_scale[column]);
    float up_value = __fmul_rn(up[index], up_scale[column]);
    if constexpr (HasGateBias) {
      gate_value = __fadd_rn(gate_value, gate_bias[column]);
    }
    if constexpr (HasUpBias) {
      up_value = __fadd_rn(up_value, up_bias[column]);
    }
    const half gate_half = __float2half_rn(gate_value);
    const half up_half = __float2half_rn(up_value);
    const half activated_gate = qvq_silu_fp16(gate_half);
    const half product = __float2half_rn(
        __fmul_rn(__half2float(activated_gate), __half2float(up_half)));
    output[index] = __float2half_rn(
        __fmul_rn(__half2float(product), __half2float(down_scale[column])));
  }
}

template <int SplitCount, bool HasGateBias, bool HasUpBias>
__global__ void qvq_folded_swiglu_precondition_ordered_fp32_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ gate_scale,
    const float* __restrict__ up_scale,
    const float* __restrict__ gate_bias,
    const float* __restrict__ up_bias,
    const half* __restrict__ down_scale,
    half* __restrict__ output,
    int64_t logical_values,
    int n) {
  const int64_t plane_values = static_cast<int64_t>(16) * n;
  const float* gate_partials = partials;
  const float* up_partials = partials + SplitCount * plane_values;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < logical_values;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float gate_value = 0.0f;
    float up_value = 0.0f;
#pragma unroll
    for (int split = 0; split < SplitCount; ++split) {
      gate_value = __fadd_rn(
          gate_value, gate_partials[static_cast<int64_t>(split) * plane_values + index]);
      up_value = __fadd_rn(
          up_value, up_partials[static_cast<int64_t>(split) * plane_values + index]);
    }
    const int column = static_cast<int>(index % n);
    gate_value = __fmul_rn(gate_value, gate_scale[column]);
    up_value = __fmul_rn(up_value, up_scale[column]);
    if constexpr (HasGateBias) {
      gate_value = __fadd_rn(gate_value, gate_bias[column]);
    }
    if constexpr (HasUpBias) {
      up_value = __fadd_rn(up_value, up_bias[column]);
    }
    const half activated_gate = qvq_silu_fp16(__float2half_rn(gate_value));
    const half up_half = __float2half_rn(up_value);
    const half product = __float2half_rn(
        __fmul_rn(__half2float(activated_gate), __half2float(up_half)));
    output[index] = __float2half_rn(
        __fmul_rn(__half2float(product), __half2float(down_scale[column])));
  }
}

// Qwen3.8 uses H_5120 = H_40 tensor H_128.  The generic graph-safe fallback
// evaluates both its historical FP16 path and its FP32 overflow rescue.  This
// kernel expresses the same stage-local contract in one launch: every value
// is rounded to FP16 when finite and retained in FP32 only when narrowing
// would overflow.  Ordinary finite rows therefore preserve the established
// FP16 boundaries, while cancellation can still recover an otherwise
// overflowing intermediate before the final FP16 store.
template <
    bool HasBias,
    int SplitCount = 1,
    int CompositeN = kQwenCompositeN,
    int CompositeBase = kQwenCompositeBase>
__global__ void __launch_bounds__(kHadamardThreads)
qvq_qwen_composite_recovery_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    const half* __restrict__ base,
    const float* __restrict__ post_scale,
    const float* __restrict__ bias,
    half* __restrict__ output) {
  constexpr int CompositeLowN = CompositeN / CompositeBase;
  extern __shared__ float shared[];
  float* low = shared;
  const int row = static_cast<int>(blockIdx.x);
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(CompositeN))));

  for (int column = static_cast<int>(threadIdx.x); column < CompositeN;
       column += static_cast<int>(blockDim.x)) {
    float value = 0.0f;
    if constexpr (SplitCount == 1) {
      value = input[static_cast<int64_t>(row) * CompositeN + column];
    } else {
#pragma unroll
      for (int split = 0; split < SplitCount; ++split) {
        value = __fadd_rn(
            value,
            input[
                static_cast<int64_t>(split) * 16 * CompositeN +
                static_cast<int64_t>(row) * CompositeN + column]);
      }
    }
    value = round_fp16_unless_overflow(value);
    low[column] = round_fp16_unless_overflow(value / divisor);
  }
  __syncthreads();

  // The ascending butterflies are independent within each contiguous
  // power-of-two slice (128, 256, or 512 columns for the promoted shapes).
#pragma unroll
  for (int bit = 1; bit < CompositeLowN; bit <<= 1) {
    for (int column = static_cast<int>(threadIdx.x);
         column < CompositeN;
         column += static_cast<int>(blockDim.x)) {
      const int local = column & (CompositeLowN - 1);
      const int peer = column ^ bit;
      if (local < (local ^ bit)) {
        const float a = low[column];
        const float b = low[peer];
        low[column] = round_fp16_unless_overflow(__fadd_rn(a, b));
        low[peer] = round_fp16_unless_overflow(__fsub_rn(a, b));
      }
    }
    __syncthreads();
  }

  // Finish with the canonical H_40 or H_12 base multiply.  Its entries are exactly
  // +/-1 in FP16, so the product has no representation error.  Accumulate in
  // FP32 and reproduce the historical FP16 output boundary afterwards.
  // Each thread owns the same columns in this loop as in the following
  // epilogue, so keep the rounded base result in a register.  The old
  // low/high shared-memory handoff and block barrier did not communicate
  // between threads and only added traffic and a dependency boundary.
  for (int column = static_cast<int>(threadIdx.x); column < CompositeN;
       column += static_cast<int>(blockDim.x)) {
    const int base_row = column / CompositeLowN;
    const int local = column & (CompositeLowN - 1);
    float value = 0.0f;
#pragma unroll
    for (int source = 0; source < CompositeBase; ++source) {
      const float coefficient = __half2float(
          base[base_row * CompositeBase + source]);
      value = __fmaf_rn(
          coefficient,
          low[source * CompositeLowN + local],
          value);
    }
    value = round_fp16_unless_overflow(value);
    const float scale = __half2float(__float2half_rn(post_scale[column]));
    value = round_fp16_unless_overflow(__fmul_rn(value, scale));
    if constexpr (HasBias) {
      const float bias_value = __half2float(__float2half_rn(bias[column]));
      value = round_fp16_unless_overflow(__fadd_rn(value, bias_value));
    }
    output[static_cast<int64_t>(row) * CompositeN + column] =
        __float2half_rn(value);
  }
}

// Parallel form of the same composite recovery contract.  Each H_L slice is
// independent, so expose B low-stage CTAs per row and materialize the exact
// overflow-preserving FP32 boundary once.  The second grid exposes one CTA
// per output base row and retains the canonical source-ascending H_B fused
// multiply-add order.  This changes scheduling only; every R() boundary is
// identical to qvq_qwen_composite_recovery_fp32_to_fp16_kernel.
template <int CompositeN, int CompositeBase>
__global__ void qvq_qwen_composite_recovery_low_kernel(
    const float* __restrict__ input,
    float* __restrict__ workspace) {
  constexpr int CompositeLowN = CompositeN / CompositeBase;
  __shared__ float values[CompositeLowN];
  const int base_slice = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int slice_begin = base_slice * CompositeLowN;
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(CompositeN))));

  for (int local = static_cast<int>(threadIdx.x); local < CompositeLowN;
       local += static_cast<int>(blockDim.x)) {
    float value = input[
        static_cast<int64_t>(row) * CompositeN + slice_begin + local];
    value = round_fp16_unless_overflow(value);
    values[local] = round_fp16_unless_overflow(value / divisor);
  }
  __syncthreads();

#pragma unroll
  for (int bit = 1; bit < CompositeLowN; bit <<= 1) {
    for (int local = static_cast<int>(threadIdx.x); local < CompositeLowN;
         local += static_cast<int>(blockDim.x)) {
      const int peer = local ^ bit;
      if (local < peer) {
        const float a = values[local];
        const float b = values[peer];
        values[local] = round_fp16_unless_overflow(__fadd_rn(a, b));
        values[peer] = round_fp16_unless_overflow(__fsub_rn(a, b));
      }
    }
    __syncthreads();
  }

  for (int local = static_cast<int>(threadIdx.x); local < CompositeLowN;
       local += static_cast<int>(blockDim.x)) {
    workspace[
        static_cast<int64_t>(row) * CompositeN + slice_begin + local] =
        values[local];
  }
}

template <bool HasBias, int CompositeN, int CompositeBase>
__global__ void qvq_qwen_composite_recovery_high_kernel(
    const float* __restrict__ workspace,
    const half* __restrict__ base,
    const float* __restrict__ post_scale,
    const float* __restrict__ bias,
    half* __restrict__ output) {
  constexpr int CompositeLowN = CompositeN / CompositeBase;
  const int base_row = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  for (int local = static_cast<int>(threadIdx.x); local < CompositeLowN;
       local += static_cast<int>(blockDim.x)) {
    float value = 0.0f;
#pragma unroll
    for (int source = 0; source < CompositeBase; ++source) {
      value = __fmaf_rn(
          __half2float(base[base_row * CompositeBase + source]),
          workspace[
              static_cast<int64_t>(row) * CompositeN +
              source * CompositeLowN + local],
          value);
    }
    value = round_fp16_unless_overflow(value);
    const int column = base_row * CompositeLowN + local;
    const float scale = __half2float(__float2half_rn(post_scale[column]));
    value = round_fp16_unless_overflow(__fmul_rn(value, scale));
    if constexpr (HasBias) {
      const float bias_value = __half2float(__float2half_rn(bias[column]));
      value = round_fp16_unless_overflow(__fadd_rn(value, bias_value));
    }
    output[static_cast<int64_t>(row) * CompositeN + column] =
        __float2half_rn(value);
  }
}

// Exact shared Qwen3.8 input transform.  The historical composite path first
// rounds x*SU to FP16, divides by FP16(sqrt(5120)) and rounds again, performs
// H128 with an FP16 boundary after every butterfly, then rounds the H40 base
// result once.  One kernel preserves that order and writes the WGMMA M16 zero
// tail without a separate allocation/fill/copy sequence.
__global__ void __launch_bounds__(kHadamardThreads)
qvq_qwen_composite_input_fp16_padded_kernel(
    const half* __restrict__ input,
    const half* __restrict__ base,
    const half* __restrict__ pre_scale,
    half* __restrict__ output,
    int logical_rows) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= logical_rows) {
    for (int column = static_cast<int>(threadIdx.x); column < kQwenCompositeN;
         column += static_cast<int>(blockDim.x)) {
      output[static_cast<int64_t>(row) * kQwenCompositeN + column] =
          __float2half_rn(0.0f);
    }
    return;
  }

  extern __shared__ float low[];
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kQwenCompositeN))));
  for (int column = static_cast<int>(threadIdx.x); column < kQwenCompositeN;
       column += static_cast<int>(blockDim.x)) {
    const int64_t offset =
        static_cast<int64_t>(row) * kQwenCompositeN + column;
    const half scaled = __float2half_rn(
        __fmul_rn(__half2float(input[offset]), __half2float(pre_scale[column])));
    low[column] = __half2float(
        __float2half_rn(__half2float(scaled) / divisor));
  }
  __syncthreads();

#pragma unroll
  for (int bit = 1; bit < kQwenCompositeLowN; bit <<= 1) {
    for (int column = static_cast<int>(threadIdx.x);
         column < kQwenCompositeN;
         column += static_cast<int>(blockDim.x)) {
      const int local = column & (kQwenCompositeLowN - 1);
      const int peer = column ^ bit;
      if (local < (local ^ bit)) {
        const float a = low[column];
        const float b = low[peer];
        low[column] = __half2float(__float2half_rn(__fadd_rn(a, b)));
        low[peer] = __half2float(__float2half_rn(__fsub_rn(a, b)));
      }
    }
    __syncthreads();
  }

  for (int column = static_cast<int>(threadIdx.x); column < kQwenCompositeN;
       column += static_cast<int>(blockDim.x)) {
    const int base_row = column / kQwenCompositeLowN;
    const int local = column & (kQwenCompositeLowN - 1);
    float value = 0.0f;
#pragma unroll
    for (int source = 0; source < kQwenCompositeBase; ++source) {
      value = __fmaf_rn(
          __half2float(base[base_row * kQwenCompositeBase + source]),
          low[source * kQwenCompositeLowN + local],
          value);
    }
    output[static_cast<int64_t>(row) * kQwenCompositeN + column] =
        __float2half_rn(value);
  }
}

// Produce one selected output of the 32-point high Hadamard while retaining
// the accepted ascending butterfly tree and overflow-preserving rounding at
// every internal node.  Computing one output uses 31 rounded operations.  A
// following 256-thread block therefore exposes all 8192 output columns in
// parallel without materializing recovered gate/up tensors.
__device__ __forceinline__ float qvq_recovery_high_selected(
    const float* __restrict__ workspace_row,
    int local,
    int output_tile) {
  float values[kHadamardPairMultiblockTiles];
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    values[tile] = workspace_row[tile * kHadamardPairMultiblockTile + local];
  }
  int active = kHadamardPairMultiblockTiles;
#pragma unroll
  for (int stage = 0; stage < 5; ++stage) {
    active >>= 1;
    const bool subtract = (output_tile & (1 << stage)) != 0;
#pragma unroll
    for (int index = 0; index < kHadamardPairMultiblockTiles / 2; ++index) {
      if (index < active) {
        const float a = values[index * 2];
        const float b = values[index * 2 + 1];
        values[index] = round_fp16_unless_overflow(
            subtract ? a - b : a + b);
      }
    }
  }
  return values[0];
}

// Produce the two outputs whose tile indices differ only in the final
// 32-point Hadamard bit. They share the first four exact reduction stages:
// 32 input loads and 30 rounded internal operations form the two remaining
// subtrees, then one rounded sum and difference produce output_tile and
// output_tile + 16. This is the same ascending tree as two independent
// qvq_recovery_high_selected calls, but uses 32 rather than 64 loads and 32
// rather than 62 rounded operations.
__device__ __forceinline__ void qvq_recovery_high_pair_selected(
    const float* __restrict__ workspace_row,
    int local,
    int output_tile,
    float& output0,
    float& output1) {
  float values[kHadamardPairMultiblockTiles];
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    values[tile] = workspace_row[tile * kHadamardPairMultiblockTile + local];
  }
  int active = kHadamardPairMultiblockTiles;
#pragma unroll
  for (int stage = 0; stage < 4; ++stage) {
    active >>= 1;
    const bool subtract = (output_tile & (1 << stage)) != 0;
#pragma unroll
    for (int index = 0; index < kHadamardPairMultiblockTiles / 2; ++index) {
      if (index < active) {
        const float a = values[index * 2];
        const float b = values[index * 2 + 1];
        values[index] = round_fp16_unless_overflow(
            subtract ? a - b : a + b);
      }
    }
  }
  output0 = round_fp16_unless_overflow(values[0] + values[1]);
  output1 = round_fp16_unless_overflow(values[0] - values[1]);
}

__device__ __forceinline__ float qvq_round_fp16_known_finite(float value) {
  return __half2float(__float2half_rn(value));
}

// A five-stage FP16 butterfly starting with sum(abs(input)) <= 64000 cannot
// overflow FP16. The margin below 65504 covers worst-case upward rounding at
// every stage. NaN and infinity propagate into abs_sum and fail the ordered
// comparison, selecting the original overflow-preserving path. The common
// bounded path therefore removes the repeated finite-test and select while
// remaining byte-identical.
__device__ __forceinline__ float qvq_recovery_high_selected_bounded(
    const float* __restrict__ workspace_row,
    int local,
    int output_tile) {
  float values[kHadamardPairMultiblockTiles];
  float abs_sum = 0.0f;
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const float value =
        workspace_row[tile * kHadamardPairMultiblockTile + local];
    values[tile] = value;
    abs_sum += fabsf(value);
  }
  int active = kHadamardPairMultiblockTiles;
  if (abs_sum <= 64000.0f) {
#pragma unroll
    for (int stage = 0; stage < 5; ++stage) {
      active >>= 1;
      const bool subtract = (output_tile & (1 << stage)) != 0;
#pragma unroll
      for (int index = 0; index < kHadamardPairMultiblockTiles / 2; ++index) {
        if (index < active) {
          const float a = values[index * 2];
          const float b = values[index * 2 + 1];
          values[index] = qvq_round_fp16_known_finite(
              subtract ? a - b : a + b);
        }
      }
    }
  } else {
#pragma unroll
    for (int stage = 0; stage < 5; ++stage) {
      active >>= 1;
      const bool subtract = (output_tile & (1 << stage)) != 0;
#pragma unroll
      for (int index = 0; index < kHadamardPairMultiblockTiles / 2; ++index) {
        if (index < active) {
          const float a = values[index * 2];
          const float b = values[index * 2 + 1];
          values[index] = round_fp16_unless_overflow(
              subtract ? a - b : a + b);
        }
      }
    }
  }
  return values[0];
}

__device__ __forceinline__ void qvq_recovery_high_pair_selected_bounded(
    const float* __restrict__ workspace_row,
    int local,
    int output_tile,
    float& output0,
    float& output1) {
  float values[kHadamardPairMultiblockTiles];
  float abs_sum = 0.0f;
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const float value =
        workspace_row[tile * kHadamardPairMultiblockTile + local];
    values[tile] = value;
    abs_sum += fabsf(value);
  }
  int active = kHadamardPairMultiblockTiles;
  if (abs_sum <= 64000.0f) {
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      active >>= 1;
      const bool subtract = (output_tile & (1 << stage)) != 0;
#pragma unroll
      for (int index = 0; index < kHadamardPairMultiblockTiles / 2; ++index) {
        if (index < active) {
          const float a = values[index * 2];
          const float b = values[index * 2 + 1];
          values[index] = qvq_round_fp16_known_finite(
              subtract ? a - b : a + b);
        }
      }
    }
    output0 = qvq_round_fp16_known_finite(values[0] + values[1]);
    output1 = qvq_round_fp16_known_finite(values[0] - values[1]);
  } else {
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      active >>= 1;
      const bool subtract = (output_tile & (1 << stage)) != 0;
#pragma unroll
      for (int index = 0; index < kHadamardPairMultiblockTiles / 2; ++index) {
        if (index < active) {
          const float a = values[index * 2];
          const float b = values[index * 2 + 1];
          values[index] = round_fp16_unless_overflow(
              subtract ? a - b : a + b);
        }
      }
    }
    output0 = round_fp16_unless_overflow(values[0] + values[1]);
    output1 = round_fp16_unless_overflow(values[0] - values[1]);
  }
}

// Gate and up select the same recovery outputs.  Once both conservative L1
// bounds prove that the five-stage trees cannot overflow, the two independent
// projections can occupy the low/high lanes of half2.  Stage zero still adds
// the FP32 workspace values and rounds once to FP16; every later __hadd2 or
// __hsub2 consumes FP16 operands and produces the same independently rounded
// FP16 lanes as the scalar FP32-add-then-convert sequence.  If either bound is
// unsafe, both projections use the original overflow-preserving implementation.
__device__ __forceinline__ void qvq_recovery_high_pair_gate_up_bounded_packed(
    const float* __restrict__ gate_workspace_row,
    const float* __restrict__ up_workspace_row,
    int local,
    int output_tile,
    float& gate_output0,
    float& gate_output1,
    float& up_output0,
    float& up_output1) {
  float gate_values[kHadamardPairMultiblockTiles];
  float up_values[kHadamardPairMultiblockTiles];
  float gate_abs_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up_abs_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const int offset = tile * kHadamardPairMultiblockTile + local;
    const float gate_value = gate_workspace_row[offset];
    const float up_value = up_workspace_row[offset];
    gate_values[tile] = gate_value;
    up_values[tile] = up_value;
    gate_abs_sums[tile & 3] += fabsf(gate_value);
    up_abs_sums[tile & 3] += fabsf(up_value);
  }
  const float gate_abs_sum =
      (gate_abs_sums[0] + gate_abs_sums[1]) +
      (gate_abs_sums[2] + gate_abs_sums[3]);
  const float up_abs_sum =
      (up_abs_sums[0] + up_abs_sums[1]) +
      (up_abs_sums[2] + up_abs_sums[3]);
  if (gate_abs_sum <= 64000.0f && up_abs_sum <= 64000.0f) {
    half2 packed_values[kHadamardPairMultiblockTiles / 2];
    const bool subtract0 = (output_tile & 1) != 0;
#pragma unroll
    for (int index = 0; index < kHadamardPairMultiblockTiles / 2; ++index) {
      const float gate_a = gate_values[index * 2];
      const float gate_b = gate_values[index * 2 + 1];
      const float up_a = up_values[index * 2];
      const float up_b = up_values[index * 2 + 1];
      packed_values[index] = __floats2half2_rn(
          subtract0 ? gate_a - gate_b : gate_a + gate_b,
          subtract0 ? up_a - up_b : up_a + up_b);
    }
    int active = kHadamardPairMultiblockTiles / 2;
#pragma unroll
    for (int stage = 1; stage < 4; ++stage) {
      active >>= 1;
      const bool subtract = (output_tile & (1 << stage)) != 0;
#pragma unroll
      for (int index = 0; index < kHadamardPairMultiblockTiles / 4; ++index) {
        if (index < active) {
          const half2 a = packed_values[index * 2];
          const half2 b = packed_values[index * 2 + 1];
          packed_values[index] = subtract ? __hsub2(a, b) : __hadd2(a, b);
        }
      }
    }
    const half2 packed_output0 = __hadd2(packed_values[0], packed_values[1]);
    const half2 packed_output1 = __hsub2(packed_values[0], packed_values[1]);
    gate_output0 = __low2float(packed_output0);
    gate_output1 = __low2float(packed_output1);
    up_output0 = __high2float(packed_output0);
    up_output1 = __high2float(packed_output1);
  } else {
    qvq_recovery_high_pair_selected(
        gate_workspace_row, local, output_tile, gate_output0, gate_output1);
    qvq_recovery_high_pair_selected(
        up_workspace_row, local, output_tile, up_output0, up_output1);
  }
}

// Fuse the selected recovery-high outputs with the exact rounded FP16 SiLU,
// product, down scale/divisor, and first eight down-input Hadamard stages.
// The nonlinear FP16 boundary is still materialized in registers, but the
// separate [gate, up] global tensors and one kernel boundary disappear.
template <bool BoundedRounding>
__global__ void qvq_recovery_high_swiglu_precondition_half2_low_kernel(
    const float* __restrict__ recovery_workspace,
    const float* __restrict__ post_scale0,
    const float* __restrict__ post_scale1,
    const float* __restrict__ bias0,
    const float* __restrict__ bias1,
    const half* __restrict__ pre_scale,
    half* __restrict__ precondition_workspace,
    int rows,
    int scale_mode) {
  __shared__ half2 buf[kHadamardPairMultiblockHalf2LowValues];
  const int row = static_cast<int>(blockIdx.y);
  const int output_tile = static_cast<int>(blockIdx.x);
  const int local = static_cast<int>(threadIdx.x);
  const int column = output_tile * kHadamardPairMultiblockTile + local;
  const int64_t row_offset =
      static_cast<int64_t>(row) * kHadamardPairMultiblockN;
  const float* gate_workspace = recovery_workspace + row_offset;
  const float* up_workspace = recovery_workspace +
      static_cast<int64_t>(rows + row) * kHadamardPairMultiblockN;

  const float reciprocal = 1.0f /
      sqrtf(static_cast<float>(kHadamardPairMultiblockN));
  float gate_value;
  float up_value;
  if constexpr (BoundedRounding) {
    gate_value = qvq_recovery_high_selected_bounded(
        gate_workspace, local, output_tile);
    up_value = qvq_recovery_high_selected_bounded(
        up_workspace, local, output_tile);
  } else {
    gate_value = qvq_recovery_high_selected(
        gate_workspace, local, output_tile);
    up_value = qvq_recovery_high_selected(
        up_workspace, local, output_tile);
  }
  if (scale_mode == 4) {
    gate_value = round_fp16_unless_overflow(gate_value * reciprocal);
    up_value = round_fp16_unless_overflow(up_value * reciprocal);
  }
  gate_value = round_fp16_unless_overflow(
      gate_value * post_scale0[column]);
  up_value = round_fp16_unless_overflow(up_value * post_scale1[column]);
  if (bias0 != nullptr) {
    gate_value = round_fp16_unless_overflow(gate_value + bias0[column]);
  }
  if (bias1 != nullptr) {
    up_value = round_fp16_unless_overflow(up_value + bias1[column]);
  }
  const half gate = __float2half_rn(gate_value);
  const half up = __float2half_rn(up_value);
  const half activated_gate = qvq_silu_fp16(gate);
  const half product_half = __float2half_rn(
      __half2float(activated_gate) * __half2float(up));
  float seed = __half2float(product_half) * __half2float(pre_scale[column]);
  seed = round_fp16_unless_overflow(seed);
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kHadamardPairMultiblockN))));
  const half initial = __float2half_rn(seed / divisor);

  const unsigned int peer_bits = __shfl_xor_sync(
      0xffffffffu, static_cast<unsigned int>(__half_as_ushort(initial)), 1);
  if ((local & 1) == 0) {
    constexpr unsigned int kEvenLaneMask = 0x55555555u;
    const int local_pair = local >> 1;
    const half peer = __ushort_as_half(static_cast<unsigned short>(peer_bits));
    const half2 pair = __halves2half2(initial, peer);
    const half2 swapped = __lowhigh2highlow(pair);
    const half2 sum = __hadd2(pair, swapped);
    const half2 difference = __hsub2(pair, swapped);
    half2 packed = __lows2half2(sum, difference);

#pragma unroll
    for (int bit = 1; bit < 16; bit <<= 1) {
      union Half2Bits {
        half2 value;
        unsigned int bits;
      } packed_bits, packed_peer;
      packed_bits.value = packed;
      packed_peer.bits = __shfl_xor_sync(
          kEvenLaneMask, packed_bits.bits, bit * 2);
      packed = (local_pair & bit) == 0
          ? __hadd2(packed, packed_peer.value)
          : __hsub2(packed_peer.value, packed);
    }
    buf[local_pair] = packed;
  }
  __syncthreads();

#pragma unroll
  for (int bit = 16; bit < kHadamardPairMultiblockHalf2LowValues; bit <<= 1) {
    if (local < kHadamardPairMultiblockHalf2LowValues) {
      const int peer = local ^ bit;
      if (local < peer) {
        const half2 a = buf[local];
        const half2 b = buf[peer];
        buf[local] = __hadd2(a, b);
        buf[peer] = __hsub2(a, b);
      }
    }
    __syncthreads();
  }
  if (local < kHadamardPairMultiblockHalf2LowValues) {
    const int64_t pair_offset = row_offset +
        output_tile * kHadamardPairMultiblockTile + local * 2;
    *reinterpret_cast<half2*>(precondition_workspace + pair_offset) = buf[local];
  }
}

// Phase-64 candidate: one CTA owns both high-transform output tiles that
// differ only in the final tile-index bit. Besides sharing the selected
// recovery tree, the CTA carries both recovered columns through the exact
// FP16 recovery, SiLU/product/pre-scale boundary, and independent half2 low
// transforms. The two shared arrays keep the transforms independent and
// preserve every accepted Phase-63 rounding operation.
template <bool BoundedRounding, bool PackedGateUp = false>
__global__ void qvq_recovery_high_pair_swiglu_precondition_half2_low_kernel(
    const float* __restrict__ recovery_workspace,
    const float* __restrict__ post_scale0,
    const float* __restrict__ post_scale1,
    const float* __restrict__ bias0,
    const float* __restrict__ bias1,
    const half* __restrict__ pre_scale,
    half* __restrict__ precondition_workspace,
    int rows,
    int scale_mode) {
  __shared__ half2 buf[2][kHadamardPairMultiblockHalf2LowValues];
  const int row = static_cast<int>(blockIdx.y);
  const int output_tile0 = static_cast<int>(blockIdx.x);
  const int output_tile1 = output_tile0 + kHadamardPairMultiblockTiles / 2;
  const int local = static_cast<int>(threadIdx.x);
  const int column0 = output_tile0 * kHadamardPairMultiblockTile + local;
  const int column1 = output_tile1 * kHadamardPairMultiblockTile + local;
  const int64_t row_offset =
      static_cast<int64_t>(row) * kHadamardPairMultiblockN;
  const float* gate_workspace = recovery_workspace + row_offset;
  const float* up_workspace = recovery_workspace +
      static_cast<int64_t>(rows + row) * kHadamardPairMultiblockN;

  float gate_value0;
  float gate_value1;
  float up_value0;
  float up_value1;
  if constexpr (BoundedRounding) {
    if constexpr (PackedGateUp) {
      qvq_recovery_high_pair_gate_up_bounded_packed(
          gate_workspace,
          up_workspace,
          local,
          output_tile0,
          gate_value0,
          gate_value1,
          up_value0,
          up_value1);
    } else {
      qvq_recovery_high_pair_selected_bounded(
          gate_workspace, local, output_tile0, gate_value0, gate_value1);
      qvq_recovery_high_pair_selected_bounded(
          up_workspace, local, output_tile0, up_value0, up_value1);
    }
  } else {
    qvq_recovery_high_pair_selected(
        gate_workspace, local, output_tile0, gate_value0, gate_value1);
    qvq_recovery_high_pair_selected(
        up_workspace, local, output_tile0, up_value0, up_value1);
  }
  const float reciprocal = 1.0f /
      sqrtf(static_cast<float>(kHadamardPairMultiblockN));
  if (scale_mode == 4) {
    gate_value0 = round_fp16_unless_overflow(gate_value0 * reciprocal);
    gate_value1 = round_fp16_unless_overflow(gate_value1 * reciprocal);
    up_value0 = round_fp16_unless_overflow(up_value0 * reciprocal);
    up_value1 = round_fp16_unless_overflow(up_value1 * reciprocal);
  }
  gate_value0 = round_fp16_unless_overflow(
      gate_value0 * post_scale0[column0]);
  gate_value1 = round_fp16_unless_overflow(
      gate_value1 * post_scale0[column1]);
  up_value0 = round_fp16_unless_overflow(
      up_value0 * post_scale1[column0]);
  up_value1 = round_fp16_unless_overflow(
      up_value1 * post_scale1[column1]);
  if (bias0 != nullptr) {
    gate_value0 = round_fp16_unless_overflow(gate_value0 + bias0[column0]);
    gate_value1 = round_fp16_unless_overflow(gate_value1 + bias0[column1]);
  }
  if (bias1 != nullptr) {
    up_value0 = round_fp16_unless_overflow(up_value0 + bias1[column0]);
    up_value1 = round_fp16_unless_overflow(up_value1 + bias1[column1]);
  }
  const half gate0 = __float2half_rn(gate_value0);
  const half gate1 = __float2half_rn(gate_value1);
  const half up0 = __float2half_rn(up_value0);
  const half up1 = __float2half_rn(up_value1);
  const half activated_gate0 = qvq_silu_fp16(gate0);
  const half activated_gate1 = qvq_silu_fp16(gate1);
  const half product_half0 = __float2half_rn(
      __half2float(activated_gate0) * __half2float(up0));
  const half product_half1 = __float2half_rn(
      __half2float(activated_gate1) * __half2float(up1));
  float seed0 = __half2float(product_half0) * __half2float(pre_scale[column0]);
  float seed1 = __half2float(product_half1) * __half2float(pre_scale[column1]);
  seed0 = round_fp16_unless_overflow(seed0);
  seed1 = round_fp16_unless_overflow(seed1);
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kHadamardPairMultiblockN))));
  const half initial0 = __float2half_rn(seed0 / divisor);
  const half initial1 = __float2half_rn(seed1 / divisor);

  const unsigned int peer_bits0 = __shfl_xor_sync(
      0xffffffffu, static_cast<unsigned int>(__half_as_ushort(initial0)), 1);
  const unsigned int peer_bits1 = __shfl_xor_sync(
      0xffffffffu, static_cast<unsigned int>(__half_as_ushort(initial1)), 1);
  if ((local & 1) == 0) {
    constexpr unsigned int kEvenLaneMask = 0x55555555u;
    const int local_pair = local >> 1;
    const half peer0 = __ushort_as_half(static_cast<unsigned short>(peer_bits0));
    const half peer1 = __ushort_as_half(static_cast<unsigned short>(peer_bits1));
    const half2 pair0 = __halves2half2(initial0, peer0);
    const half2 pair1 = __halves2half2(initial1, peer1);
    const half2 swapped0 = __lowhigh2highlow(pair0);
    const half2 swapped1 = __lowhigh2highlow(pair1);
    const half2 sum0 = __hadd2(pair0, swapped0);
    const half2 sum1 = __hadd2(pair1, swapped1);
    const half2 difference0 = __hsub2(pair0, swapped0);
    const half2 difference1 = __hsub2(pair1, swapped1);
    half2 packed0 = __lows2half2(sum0, difference0);
    half2 packed1 = __lows2half2(sum1, difference1);

#pragma unroll
    for (int bit = 1; bit < 16; bit <<= 1) {
      union Half2Bits {
        half2 value;
        unsigned int bits;
      } packed_bits0, packed_bits1, packed_peer0, packed_peer1;
      packed_bits0.value = packed0;
      packed_bits1.value = packed1;
      packed_peer0.bits = __shfl_xor_sync(
          kEvenLaneMask, packed_bits0.bits, bit * 2);
      packed_peer1.bits = __shfl_xor_sync(
          kEvenLaneMask, packed_bits1.bits, bit * 2);
      packed0 = (local_pair & bit) == 0
          ? __hadd2(packed0, packed_peer0.value)
          : __hsub2(packed_peer0.value, packed0);
      packed1 = (local_pair & bit) == 0
          ? __hadd2(packed1, packed_peer1.value)
          : __hsub2(packed_peer1.value, packed1);
    }
    buf[0][local_pair] = packed0;
    buf[1][local_pair] = packed1;
  }
  __syncthreads();

#pragma unroll
  for (int bit = 16; bit < kHadamardPairMultiblockHalf2LowValues; bit <<= 1) {
    if (local < kHadamardPairMultiblockHalf2LowValues) {
      const int peer = local ^ bit;
      if (local < peer) {
        const half2 a0 = buf[0][local];
        const half2 b0 = buf[0][peer];
        const half2 a1 = buf[1][local];
        const half2 b1 = buf[1][peer];
        buf[0][local] = __hadd2(a0, b0);
        buf[0][peer] = __hsub2(a0, b0);
        buf[1][local] = __hadd2(a1, b1);
        buf[1][peer] = __hsub2(a1, b1);
      }
    }
    __syncthreads();
  }
  if (local < kHadamardPairMultiblockHalf2LowValues) {
    const int64_t pair_offset0 = row_offset +
        output_tile0 * kHadamardPairMultiblockTile + local * 2;
    const int64_t pair_offset1 = row_offset +
        output_tile1 * kHadamardPairMultiblockTile + local * 2;
    *reinterpret_cast<half2*>(precondition_workspace + pair_offset0) =
        buf[0][local];
    *reinterpret_cast<half2*>(precondition_workspace + pair_offset1) =
        buf[1][local];
  }
}

// Exact N=8192 multiblock factorization of qvq_swiglu_precondition_kernel.
// The workspace is FP16 because the single-CTA reference stores every stage
// in FP16 shared memory.  Splitting after bit 128 therefore preserves the
// existing rounding boundary byte-for-byte while exposing 32 independent low
// tiles and four high-column blocks per MLP row. FuseSilu adds exactly one
// fixed H100 candidate specialization; it consumes the recovered FP16 gate,
// reproduces PyTorch's rounded FP16 SiLU boundary in-register, and deletes the
// standalone activation launch/materialization.
template <bool FuseSilu>
__global__ void qvq_swiglu_precondition_multiblock_low_kernel(
    const half* __restrict__ gate_or_activated_gate,
    const half* __restrict__ up,
    const half* __restrict__ pre_scale,
    half* __restrict__ workspace) {
  __shared__ half buf[kHadamardPairMultiblockTile + kHadamardPairMultiblockTile / 32];
  const int row = static_cast<int>(blockIdx.y);
  const int tile = static_cast<int>(blockIdx.x);
  const int local = static_cast<int>(threadIdx.x);
  const int column = tile * kHadamardPairMultiblockTile + local;
  const int64_t offset =
      static_cast<int64_t>(row) * kHadamardPairMultiblockN + column;
  auto p = [](int i) { return i + (i >> 5); };
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kHadamardPairMultiblockN))));

  const half activated_gate = FuseSilu
      ? qvq_silu_fp16(gate_or_activated_gate[offset])
      : gate_or_activated_gate[offset];
  const half product_half = __float2half_rn(
      __half2float(activated_gate) * __half2float(up[offset]));
  float value = __half2float(product_half) * __half2float(pre_scale[column]);
  value = round_fp16_unless_overflow(value);
  buf[p(local)] = __float2half_rn(value / divisor);
  __syncthreads();

#pragma unroll
  for (int bit = 1; bit < kHadamardPairMultiblockTile; bit <<= 1) {
    const int peer = local ^ bit;
    if (local < peer) {
      const float a = __half2float(buf[p(local)]);
      const float b = __half2float(buf[p(peer)]);
      buf[p(local)] = __float2half_rn(a + b);
      buf[p(peer)] = __float2half_rn(a - b);
    }
    __syncthreads();
  }
  workspace[offset] = buf[p(local)];
}

// Exact vectorized form of the first eight N=8192 butterfly stages. One
// input thread first owns one column, preserving the original eight-warp
// SiLU/setup parallelism. Adjacent lanes exchange their rounded input through
// a fixed XOR shuffle. Even lanes retain packed values through the next four
// within-warp stages. Only the final three cross-warp stages use shared
// memory and block barriers. __lows2half2 retains the rounded low-lane sum
// and difference:
//   pair = [a, b]
//   sum  = hadd2(pair, swap(pair)) = [rn(a+b), rn(b+a)]
//   diff = hsub2(pair, swap(pair)) = [rn(a-b), rn(b-a)]
//   next = lows2half2(sum, diff)    = [rn(a+b), rn(a-b)]
// No butterfly is reassociated and every stage still rounds once to FP16.
template <bool FuseSilu>
__global__ void qvq_swiglu_precondition_multiblock_half2_low_kernel(
    const half* __restrict__ gate_or_activated_gate,
    const half* __restrict__ up,
    const half* __restrict__ pre_scale,
    half* __restrict__ workspace) {
  __shared__ half2 buf[kHadamardPairMultiblockHalf2LowValues];
  const int row = static_cast<int>(blockIdx.y);
  const int tile = static_cast<int>(blockIdx.x);
  const int local = static_cast<int>(threadIdx.x);
  const int column = tile * kHadamardPairMultiblockTile + local;
  const int64_t offset =
      static_cast<int64_t>(row) * kHadamardPairMultiblockN + column;
  const float divisor = __half2float(
      __float2half_rn(sqrtf(static_cast<float>(kHadamardPairMultiblockN))));

  const half activated_gate = FuseSilu
      ? qvq_silu_fp16(gate_or_activated_gate[offset])
      : gate_or_activated_gate[offset];
  const half product_half = __float2half_rn(
      __half2float(activated_gate) * __half2float(up[offset]));
  float value = __half2float(product_half) * __half2float(pre_scale[column]);
  value = round_fp16_unless_overflow(value);
  const half initial = __float2half_rn(value / divisor);
  const unsigned int peer_bits = __shfl_xor_sync(
      0xffffffffu, static_cast<unsigned int>(__half_as_ushort(initial)), 1);
  if ((local & 1) == 0) {
    constexpr unsigned int kEvenLaneMask = 0x55555555u;
    const int local_pair = local >> 1;
    const half peer = __ushort_as_half(static_cast<unsigned short>(peer_bits));
    const half2 pair = __halves2half2(initial, peer);
    const half2 swapped = __lowhigh2highlow(pair);
    const half2 sum = __hadd2(pair, swapped);
    const half2 difference = __hsub2(pair, swapped);
    half2 packed = __lows2half2(sum, difference);

#pragma unroll
    for (int bit = 1; bit < 16; bit <<= 1) {
      union Half2Bits {
        half2 value;
        unsigned int bits;
      } packed_bits, packed_peer;
      packed_bits.value = packed;
      packed_peer.bits = __shfl_xor_sync(
          kEvenLaneMask, packed_bits.bits, bit * 2);
      packed = (local_pair & bit) == 0
          ? __hadd2(packed, packed_peer.value)
          : __hsub2(packed_peer.value, packed);
    }
    buf[local_pair] = packed;
  }
  __syncthreads();

#pragma unroll
  for (int bit = 16; bit < kHadamardPairMultiblockHalf2LowValues; bit <<= 1) {
    if (local < kHadamardPairMultiblockHalf2LowValues) {
      const int peer = local ^ bit;
      if (local < peer) {
        const half2 a = buf[local];
        const half2 b = buf[peer];
        buf[local] = __hadd2(a, b);
        buf[peer] = __hsub2(a, b);
      }
    }
    __syncthreads();
  }
  if (local < kHadamardPairMultiblockHalf2LowValues) {
    const int64_t pair_offset =
        static_cast<int64_t>(row) * kHadamardPairMultiblockN +
        tile * kHadamardPairMultiblockTile + local * 2;
    *reinterpret_cast<half2*>(workspace + pair_offset) = buf[local];
  }
}

template <bool PadTo16>
__global__ void qvq_swiglu_precondition_multiblock_high_kernel(
    const half* __restrict__ workspace,
    half* __restrict__ output,
    int rows) {
  const int row = static_cast<int>(blockIdx.y);
  const int local = static_cast<int>(blockIdx.x) * kHadamardPairMultiblockHighThreads +
      static_cast<int>(threadIdx.x);
  if constexpr (PadTo16) {
    if (row >= rows) {
#pragma unroll
      for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
        const int column = tile * kHadamardPairMultiblockTile + local;
        output[static_cast<int64_t>(row) * kHadamardPairMultiblockN + column] =
            __float2half_rn(0.0f);
      }
      return;
    }
  }
  half values[kHadamardPairMultiblockTiles];
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const int column = tile * kHadamardPairMultiblockTile + local;
    values[tile] = workspace[
        static_cast<int64_t>(row) * kHadamardPairMultiblockN + column];
  }

#pragma unroll
  for (int bit = 1; bit < kHadamardPairMultiblockTiles; bit <<= 1) {
#pragma unroll
    for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
      const int peer = tile ^ bit;
      if (tile < peer) {
        const float a = __half2float(values[tile]);
        const float b = __half2float(values[peer]);
        values[tile] = __float2half_rn(a + b);
        values[peer] = __float2half_rn(a - b);
      }
    }
  }

#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const int column = tile * kHadamardPairMultiblockTile + local;
    output[static_cast<int64_t>(row) * kHadamardPairMultiblockN + column] =
        values[tile];
  }
}

// Exact vectorized form of the high five N=8192 butterfly stages. One thread
// owns two adjacent within-tile columns, so every half2 lane follows the same
// tile-index butterfly and retains the reference's FP16 rounding after each
// add/sub. The experiment preserves four blocks per row while reducing each
// block from two warps to one; CUDA-event timing decides whether the denser
// instruction stream outweighs the lower warp count on H100.
template <bool PadTo16>
__global__ void qvq_swiglu_precondition_multiblock_half2_high_kernel(
    const half* __restrict__ workspace,
    half* __restrict__ output,
    int rows) {
  const int row = static_cast<int>(blockIdx.y);
  const int local_pair =
      static_cast<int>(blockIdx.x) * kHadamardPairMultiblockHalf2HighThreads +
      static_cast<int>(threadIdx.x);
  const int local = local_pair * 2;
  if constexpr (PadTo16) {
    if (row >= rows) {
      const half2 zero = __float2half2_rn(0.0f);
#pragma unroll
      for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
        const int column = tile * kHadamardPairMultiblockTile + local;
        *reinterpret_cast<half2*>(
            output + static_cast<int64_t>(row) * kHadamardPairMultiblockN + column) = zero;
      }
      return;
    }
  }
  half2 values[kHadamardPairMultiblockTiles];
#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const int column = tile * kHadamardPairMultiblockTile + local;
    values[tile] = *reinterpret_cast<const half2*>(
        workspace + static_cast<int64_t>(row) * kHadamardPairMultiblockN + column);
  }

#pragma unroll
  for (int bit = 1; bit < kHadamardPairMultiblockTiles; bit <<= 1) {
#pragma unroll
    for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
      const int peer = tile ^ bit;
      if (tile < peer) {
        const half2 a = values[tile];
        const half2 b = values[peer];
        values[tile] = __hadd2(a, b);
        values[peer] = __hsub2(a, b);
      }
    }
  }

#pragma unroll
  for (int tile = 0; tile < kHadamardPairMultiblockTiles; ++tile) {
    const int column = tile * kHadamardPairMultiblockTile + local;
    *reinterpret_cast<half2*>(
        output + static_cast<int64_t>(row) * kHadamardPairMultiblockN + column) =
        values[tile];
  }
}

at::Tensor qvq_hadamard_cuda(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    const c10::optional<at::Tensor>& bias,
    int64_t scale_mode,
    bool pad_to_16,
    bool output_fp16) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() >= 1, "input must be at least rank one");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "hadamard requires float16, bfloat16, or float32 input");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(scale_mode >= 0 && scale_mode <= 5,
              "scale_mode must be 0/1 (native), 2 (range-safe pre-scale), 3/4 (FP16 emulation), "
              "or 5 (unnormalized composite stage)");
  TORCH_CHECK(scale_mode != 2 || input.scalar_type() == at::kHalf,
              "range-safe pre-scale mode 2 requires float16 input");
  TORCH_CHECK(scale_mode < 3 || scale_mode == 5 || input.scalar_type() == at::kFloat,
              "FP16-emulation scale modes 3/4 require float32 input");
  const int64_t n64 = input.size(-1);
  TORCH_CHECK(n64 >= 2 && (n64 & (n64 - 1)) == 0, "hadamard requires a power-of-two last dim");
  TORCH_CHECK(n64 <= 16384, "hadamard fused kernel supports last dim up to 16384");
  const int n = static_cast<int>(n64);
  const int64_t rows = input.numel() / n;
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "row count exceeds int32 kernel limit");
  TORCH_CHECK(!pad_to_16 || (input.dim() == 2 && rows > 0 && rows <= 16),
              "padded Hadamard requires a nonempty 2D input with at most 16 rows");
  TORCH_CHECK(!output_fp16 || (input.scalar_type() == at::kFloat && scale_mode >= 3 && !pad_to_16),
              "FP16 Hadamard output requires FP32 input, scale mode 3/4, and no M16 padding");

  const auto check_optional = [&](const c10::optional<at::Tensor>& t, const char* name) {
    if (t.has_value()) {
      TORCH_CHECK(t->is_cuda() && t->device() == input.device(), name, " must be on the input device");
      TORCH_CHECK(t->scalar_type() == input.scalar_type(), name, " must match input dtype");
      TORCH_CHECK(t->is_contiguous(), name, " must be contiguous");
      TORCH_CHECK(t->numel() == n, name, " must have ", n, " elements");
    }
  };
  check_optional(pre_scale, "pre_scale");
  check_optional(post_scale, "post_scale");
  check_optional(bias, "bias");

  if (rows == 0) {
    return at::empty_like(input);
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  // Padded layout (i + i/32) needs n/32 extra elements.
  const size_t smem_bytes = static_cast<size_t>(n + n / 32) * input.element_size();
  TORCH_CHECK(smem_bytes <= static_cast<size_t>(properties.sharedMemPerBlockOptin),
              "n too large for the device dynamic shared memory limit");

  at::Tensor output = output_fp16
      ? at::empty(input.sizes(), input.options().dtype(at::kHalf))
      : (pad_to_16 ? at::empty({16, n64}, input.options()) : at::empty_like(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned int>(rows));
#define QVQ_HADAMARD_LAUNCH(SCALAR, OUTPUT_SCALAR, PAD)                                                            \
  {                                                                                                               \
    const SCALAR* in_ptr = reinterpret_cast<const SCALAR*>(input.const_data_ptr());                                \
    OUTPUT_SCALAR* out_ptr = reinterpret_cast<OUTPUT_SCALAR*>(output.mutable_data_ptr());                          \
    const SCALAR* pre = pre_scale.has_value()                                                                      \
        ? reinterpret_cast<const SCALAR*>(pre_scale->const_data_ptr())                                             \
        : nullptr;                                                                                                 \
    const SCALAR* post = post_scale.has_value()                                                                    \
        ? reinterpret_cast<const SCALAR*>(post_scale->const_data_ptr())                                            \
        : nullptr;                                                                                                 \
    const SCALAR* bia = bias.has_value()                                                                           \
        ? reinterpret_cast<const SCALAR*>(bias->const_data_ptr())                                                  \
        : nullptr;                                                                                                 \
    qvq_hadamard_kernel<SCALAR, OUTPUT_SCALAR, PAD><<<grid, kHadamardThreads, smem_bytes, stream>>>(               \
        in_ptr, out_ptr, pre, post, bia, n, static_cast<int>(scale_mode), static_cast<int>(rows));                  \
  }
  if (input.scalar_type() == at::kHalf) {
    if (pad_to_16) {
      QVQ_HADAMARD_LAUNCH(half, half, true)
    } else {
      QVQ_HADAMARD_LAUNCH(half, half, false)
    }
  } else if (input.scalar_type() == at::kBFloat16) {
    if (pad_to_16) {
      QVQ_HADAMARD_LAUNCH(nv_bfloat16, nv_bfloat16, true)
    } else {
      QVQ_HADAMARD_LAUNCH(nv_bfloat16, nv_bfloat16, false)
    }
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qvq_hadamard_kernel<float, float, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    if (pad_to_16) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qvq_hadamard_kernel<float, float, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
      QVQ_HADAMARD_LAUNCH(float, float, true)
    } else if (output_fp16) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qvq_hadamard_kernel<float, half, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
      QVQ_HADAMARD_LAUNCH(float, half, false)
    } else {
      QVQ_HADAMARD_LAUNCH(float, float, false)
    }
  }
#undef QVQ_HADAMARD_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_hadamard_input_fp16_padded_multiblock_cuda(
    const at::Tensor& input,
    const at::Tensor& pre_scale) {
  TORCH_CHECK(input.is_cuda() && pre_scale.is_cuda(),
              "multiblock input Hadamard tensors must be CUDA tensors");
  TORCH_CHECK(input.device() == pre_scale.device(),
              "multiblock input Hadamard tensors must share a device");
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
                  pre_scale.scalar_type() == at::kHalf,
              "multiblock input Hadamard tensors must be float16");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 16 &&
                  input.size(1) == kHadamardInputMultiblockN,
              "multiblock input Hadamard requires an Mx2048 input with M in [1, 16]");
  TORCH_CHECK(input.is_contiguous() && pre_scale.is_contiguous(),
              "multiblock input Hadamard tensors must be contiguous");
  TORCH_CHECK(pre_scale.numel() == kHadamardInputMultiblockN,
              "multiblock input Hadamard pre_scale must contain 2048 values");

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(properties.major == 9,
              "multiblock input Hadamard is an experimental Hopper-only operator");
  const int rows = static_cast<int>(input.size(0));
  at::Tensor output = at::empty(
      {16, kHadamardInputMultiblockN}, input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  qvq_hadamard_input_fp16_padded_multiblock_low_kernel<<<
      dim3(kHadamardInputMultiblockTiles, rows),
      kHadamardInputMultiblockLowThreads,
      0,
      stream>>>(
      reinterpret_cast<const half*>(input.const_data_ptr()),
      reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
      reinterpret_cast<half*>(output.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  qvq_hadamard_input_fp16_padded_multiblock_high_kernel<<<
      dim3(
          kHadamardInputMultiblockHalf2Values /
              kHadamardInputMultiblockHighThreads,
          16),
      kHadamardInputMultiblockHighThreads,
      0,
      stream>>>(
      reinterpret_cast<half*>(output.mutable_data_ptr()), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<at::Tensor, at::Tensor> qvq_hadamard_pair_fp32_to_fp16_cuda(
    const at::Tensor& input0,
    const at::Tensor& input1,
    const at::Tensor& post_scale0,
    const at::Tensor& post_scale1,
    const c10::optional<at::Tensor>& bias0,
    const c10::optional<at::Tensor>& bias1,
    int64_t scale_mode) {
  TORCH_CHECK(input0.is_cuda() && input1.is_cuda(), "paired Hadamard inputs must be CUDA tensors");
  TORCH_CHECK(input0.device() == input1.device(), "paired Hadamard inputs must share a device");
  TORCH_CHECK(input0.scalar_type() == at::kFloat && input1.scalar_type() == at::kFloat,
              "paired Hadamard inputs must be float32");
  TORCH_CHECK(input0.sizes() == input1.sizes(), "paired Hadamard inputs must have identical shapes");
  TORCH_CHECK(input0.dim() >= 1, "paired Hadamard inputs must be at least rank one");
  TORCH_CHECK(input0.is_contiguous() && input1.is_contiguous(), "paired Hadamard inputs must be contiguous");
  TORCH_CHECK(scale_mode == 3 || scale_mode == 4, "paired Hadamard scale_mode must be 3 or 4");
  const int64_t n64 = input0.size(-1);
  TORCH_CHECK(n64 >= 2 && (n64 & (n64 - 1)) == 0,
              "paired Hadamard requires a power-of-two last dimension");
  TORCH_CHECK(n64 <= 16384, "paired Hadamard supports a last dimension up to 16384");
  const int n = static_cast<int>(n64);
  const int64_t rows64 = input0.numel() / n;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max() / 2, "paired Hadamard row count exceeds launch limit");

  const auto check_vector = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == input0.device(), name, " must be on the input device");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() == n, name, " must have one value per output column");
  };
  check_vector(post_scale0, "post_scale0");
  check_vector(post_scale1, "post_scale1");
  const auto check_bias = [&](const c10::optional<at::Tensor>& tensor, const char* name) {
    if (tensor.has_value()) {
      check_vector(*tensor, name);
    }
  };
  check_bias(bias0, "bias0");
  check_bias(bias1, "bias1");

  auto output_options = input0.options().dtype(at::kHalf);
  at::Tensor output0 = at::empty(input0.sizes(), output_options);
  at::Tensor output1 = at::empty(input1.sizes(), output_options);
  if (rows64 == 0) {
    return {output0, output1};
  }

  const c10::cuda::CUDAGuard device_guard(input0.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input0.get_device()));
  const size_t smem_bytes = static_cast<size_t>(n + n / 32) * sizeof(float);
  TORCH_CHECK(smem_bytes <= static_cast<size_t>(properties.sharedMemPerBlockOptin),
              "paired Hadamard width exceeds the device dynamic shared-memory limit");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      qvq_hadamard_pair_fp32_to_fp16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input0.get_device());
  const int rows = static_cast<int>(rows64);
  qvq_hadamard_pair_fp32_to_fp16_kernel<<<static_cast<unsigned int>(rows * 2), kHadamardThreads,
                                           smem_bytes, stream>>>(
      input0.const_data_ptr<float>(),
      input1.const_data_ptr<float>(),
      reinterpret_cast<half*>(output0.mutable_data_ptr()),
      reinterpret_cast<half*>(output1.mutable_data_ptr()),
      post_scale0.const_data_ptr<float>(),
      post_scale1.const_data_ptr<float>(),
      bias0.has_value() ? bias0->const_data_ptr<float>() : nullptr,
      bias1.has_value() ? bias1->const_data_ptr<float>() : nullptr,
      n,
      rows,
      static_cast<int>(scale_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output0, output1};
}

at::Tensor qvq_hadamard_ordered_split16_fp32_to_fp16_cuda(
    const at::Tensor& partial_input,
    const at::Tensor& post_scale,
    const c10::optional<at::Tensor>& bias,
    int64_t scale_mode,
    int64_t logical_rows,
    bool multiblock) {
  TORCH_CHECK(partial_input.is_cuda(), "ordered split recovery input must be CUDA");
  TORCH_CHECK(partial_input.scalar_type() == at::kFloat,
              "ordered split recovery input must be float32");
  TORCH_CHECK(partial_input.is_contiguous(),
              "ordered split recovery input must be contiguous");
  TORCH_CHECK(partial_input.dim() == 3 && partial_input.size(0) == 16 &&
                  partial_input.size(1) == 16 && partial_input.size(2) == 2048,
              "ordered split recovery requires [16, 16, 2048] partials");
  TORCH_CHECK(logical_rows >= 1 && logical_rows <= 16,
              "ordered split recovery logical_rows must be in [1, 16]");
  TORCH_CHECK(scale_mode == 3 || scale_mode == 4,
              "ordered split recovery scale_mode must be 3 or 4");
  TORCH_CHECK(post_scale.is_cuda() && post_scale.device() == partial_input.device() &&
                  post_scale.scalar_type() == at::kFloat && post_scale.is_contiguous() &&
                  post_scale.numel() == 2048,
              "ordered split recovery post_scale must be contiguous CUDA float32[2048]");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == partial_input.device() &&
                    bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
                    bias->numel() == 2048,
                "ordered split recovery bias must be contiguous CUDA float32[2048]");
  }

  const c10::cuda::CUDAGuard device_guard(partial_input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, partial_input.get_device()));
  TORCH_CHECK(properties.major == 9 && properties.minor == 0,
              "ordered split recovery requires Hopper SM90");
  constexpr int n = kHadamardOrderedSplitN;
  at::Tensor output = at::empty(
      {logical_rows, n}, partial_input.options().dtype(at::kHalf));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(partial_input.get_device());
  if (multiblock) {
    at::Tensor workspace = at::empty(
        {logical_rows, n}, partial_input.options().dtype(at::kFloat));
    const dim3 low_grid(
        kHadamardOrderedSplitTiles,
        static_cast<unsigned int>(logical_rows));
    qvq_hadamard_ordered_split16_fp32_to_fp16_multiblock_low_kernel<<<
        low_grid, kHadamardOrderedSplitTile, 0, stream>>>(
        partial_input.const_data_ptr<float>(),
        workspace.mutable_data_ptr<float>(),
        static_cast<int>(scale_mode));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 high_grid(
        kHadamardOrderedSplitTile / kHadamardOrderedSplitHighThreads,
        static_cast<unsigned int>(logical_rows));
    qvq_hadamard_ordered_split16_fp32_to_fp16_multiblock_high_kernel<<<
        high_grid, kHadamardOrderedSplitHighThreads, 0, stream>>>(
        workspace.const_data_ptr<float>(),
        reinterpret_cast<half*>(output.mutable_data_ptr()),
        post_scale.const_data_ptr<float>(),
        bias.has_value() ? bias->const_data_ptr<float>() : nullptr,
        static_cast<int>(scale_mode));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  const size_t smem_bytes = static_cast<size_t>(n + n / 32) * sizeof(float);
  TORCH_CHECK(smem_bytes <= static_cast<size_t>(properties.sharedMemPerBlockOptin),
              "ordered split recovery exceeds dynamic shared-memory capacity");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      qvq_hadamard_ordered_split16_fp32_to_fp16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));

  qvq_hadamard_ordered_split16_fp32_to_fp16_kernel<<<
      static_cast<unsigned int>(logical_rows), kHadamardThreads, smem_bytes, stream>>>(
      partial_input.const_data_ptr<float>(),
      reinterpret_cast<half*>(output.mutable_data_ptr()),
      post_scale.const_data_ptr<float>(),
      bias.has_value() ? bias->const_data_ptr<float>() : nullptr,
      n,
      static_cast<int>(logical_rows),
      static_cast<int>(scale_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<at::Tensor, at::Tensor> qvq_hadamard_pair_fp32_to_fp16_multiblock_cuda(
    const at::Tensor& input0,
    const at::Tensor& input1,
    const at::Tensor& post_scale0,
    const at::Tensor& post_scale1,
    const c10::optional<at::Tensor>& bias0,
    const c10::optional<at::Tensor>& bias1,
    int64_t scale_mode,
    bool warp_low) {
  TORCH_CHECK(input0.is_cuda() && input1.is_cuda(), "multiblock paired Hadamard inputs must be CUDA tensors");
  TORCH_CHECK(input0.device() == input1.device(), "multiblock paired Hadamard inputs must share a device");
  TORCH_CHECK(input0.scalar_type() == at::kFloat && input1.scalar_type() == at::kFloat,
              "multiblock paired Hadamard inputs must be float32");
  TORCH_CHECK(input0.sizes() == input1.sizes(),
              "multiblock paired Hadamard inputs must have identical shapes");
  TORCH_CHECK(input0.dim() >= 1 && input0.is_contiguous() && input1.is_contiguous(),
              "multiblock paired Hadamard inputs must be contiguous with rank >= 1");
  TORCH_CHECK(input0.size(-1) == kHadamardPairMultiblockN,
              "multiblock paired Hadamard requires last dimension 8192");
  TORCH_CHECK(scale_mode == 3 || scale_mode == 4,
              "multiblock paired Hadamard scale_mode must be 3 or 4");
  const int64_t rows64 = input0.numel() / kHadamardPairMultiblockN;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max() / 2,
              "multiblock paired Hadamard row count exceeds launch limit");

  const auto check_vector = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == input0.device(), name, " must be on the input device");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() == kHadamardPairMultiblockN,
                name, " must have one value per output column");
  };
  check_vector(post_scale0, "post_scale0");
  check_vector(post_scale1, "post_scale1");
  if (bias0.has_value()) {
    check_vector(*bias0, "bias0");
  }
  if (bias1.has_value()) {
    check_vector(*bias1, "bias1");
  }

  auto output_options = input0.options().dtype(at::kHalf);
  at::Tensor output0 = at::empty(input0.sizes(), output_options);
  at::Tensor output1 = at::empty(input1.sizes(), output_options);
  if (rows64 == 0) {
    return {output0, output1};
  }

  const c10::cuda::CUDAGuard device_guard(input0.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input0.get_device()));
  TORCH_CHECK(properties.major == 9,
              "multiblock paired Hadamard is an experimental Hopper-only operator");
  const int rows = static_cast<int>(rows64);
  at::Tensor workspace = at::empty(
      {rows * 2, kHadamardPairMultiblockN}, input0.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input0.get_device());
  if (warp_low) {
    qvq_hadamard_pair_fp32_to_fp16_multiblock_warp_low_kernel<<<
        dim3(kHadamardPairMultiblockTiles, rows * 2),
        kHadamardPairMultiblockTile,
        0,
        stream>>>(
        input0.const_data_ptr<float>(),
        input1.const_data_ptr<float>(),
        workspace.mutable_data_ptr<float>(),
        rows,
        static_cast<int>(scale_mode));
  } else {
    qvq_hadamard_pair_fp32_to_fp16_multiblock_low_kernel<<<
        dim3(kHadamardPairMultiblockTiles, rows * 2),
        kHadamardPairMultiblockTile,
        0,
        stream>>>(
        input0.const_data_ptr<float>(),
        input1.const_data_ptr<float>(),
        workspace.mutable_data_ptr<float>(),
        rows,
        static_cast<int>(scale_mode));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  qvq_hadamard_pair_fp32_to_fp16_multiblock_high_kernel<<<
      dim3(kHadamardPairMultiblockTile / kHadamardPairMultiblockHighThreads, rows * 2),
      kHadamardPairMultiblockHighThreads,
      0,
      stream>>>(
      workspace.const_data_ptr<float>(),
      reinterpret_cast<half*>(output0.mutable_data_ptr()),
      reinterpret_cast<half*>(output1.mutable_data_ptr()),
      post_scale0.const_data_ptr<float>(),
      post_scale1.const_data_ptr<float>(),
      bias0.has_value() ? bias0->const_data_ptr<float>() : nullptr,
      bias1.has_value() ? bias1->const_data_ptr<float>() : nullptr,
      rows,
      static_cast<int>(scale_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output0, output1};
}

at::Tensor qvq_hadamard_pair_swiglu_precondition_multiblock_cuda(
    const at::Tensor& input0,
    const at::Tensor& input1,
    const at::Tensor& post_scale0,
    const at::Tensor& post_scale1,
    const c10::optional<at::Tensor>& bias0,
    const c10::optional<at::Tensor>& bias1,
    const at::Tensor& pre_scale,
    int64_t scale_mode,
    bool pad_to_16,
    bool pair_tiles,
    bool bounded_rounding,
    bool packed_gate_up) {
  TORCH_CHECK(input0.is_cuda() && input1.is_cuda() && pre_scale.is_cuda(),
              "fused recovery/precondition tensors must be CUDA tensors");
  TORCH_CHECK(input0.device() == input1.device() && input0.device() == pre_scale.device(),
              "fused recovery/precondition tensors must share a device");
  TORCH_CHECK(input0.scalar_type() == at::kFloat && input1.scalar_type() == at::kFloat,
              "fused recovery inputs must be float32");
  TORCH_CHECK(pre_scale.scalar_type() == at::kHalf,
              "fused recovery pre_scale must be float16");
  TORCH_CHECK(input0.sizes() == input1.sizes() && input0.dim() == 2,
              "fused recovery inputs must have identical 2D shapes");
  TORCH_CHECK(input0.is_contiguous() && input1.is_contiguous() &&
                  pre_scale.is_contiguous(),
              "fused recovery/precondition tensors must be contiguous");
  TORCH_CHECK(input0.size(1) == kHadamardPairMultiblockN &&
                  pre_scale.numel() == kHadamardPairMultiblockN,
              "fused recovery/precondition requires N=8192");
  TORCH_CHECK(scale_mode == 3 || scale_mode == 4,
              "fused recovery scale_mode must be 3 or 4");
  const int64_t rows64 = input0.size(0);
  TORCH_CHECK(rows64 > 0 && rows64 <= 16,
              "fused recovery/precondition requires one through sixteen rows");

  const auto check_float_vector = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == input0.device(),
                name, " must be on the input device");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.is_contiguous() &&
                    tensor.numel() == kHadamardPairMultiblockN,
                name, " must be contiguous float32[8192]");
  };
  check_float_vector(post_scale0, "post_scale0");
  check_float_vector(post_scale1, "post_scale1");
  if (bias0.has_value()) {
    check_float_vector(*bias0, "bias0");
  }
  if (bias1.has_value()) {
    check_float_vector(*bias1, "bias1");
  }

  const c10::cuda::CUDAGuard device_guard(input0.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input0.get_device()));
  TORCH_CHECK(properties.major == 9,
              "fused recovery/precondition is a Hopper-only operator");
  const int rows = static_cast<int>(rows64);
  auto half_options = input0.options().dtype(at::kHalf);
  at::Tensor recovery_workspace = at::empty(
      {rows * 2, kHadamardPairMultiblockN}, input0.options());
  at::Tensor precondition_workspace = at::empty(
      {rows, kHadamardPairMultiblockN}, half_options);
  at::Tensor output = pad_to_16
      ? at::empty({16, kHadamardPairMultiblockN}, half_options)
      : at::empty({rows, kHadamardPairMultiblockN}, half_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input0.get_device());

  qvq_hadamard_pair_fp32_to_fp16_multiblock_warp_low_kernel<<<
      dim3(kHadamardPairMultiblockTiles, rows * 2),
      kHadamardPairMultiblockTile,
      0,
      stream>>>(
      input0.const_data_ptr<float>(),
      input1.const_data_ptr<float>(),
      recovery_workspace.mutable_data_ptr<float>(),
      rows,
      static_cast<int>(scale_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (pair_tiles) {
    if (bounded_rounding) {
      if (packed_gate_up) {
        qvq_recovery_high_pair_swiglu_precondition_half2_low_kernel<true, true><<<
            dim3(kHadamardPairMultiblockTiles / 2, rows),
            kHadamardPairMultiblockTile,
            0,
            stream>>>(
            recovery_workspace.const_data_ptr<float>(),
            post_scale0.const_data_ptr<float>(),
            post_scale1.const_data_ptr<float>(),
            bias0.has_value() ? bias0->const_data_ptr<float>() : nullptr,
            bias1.has_value() ? bias1->const_data_ptr<float>() : nullptr,
            reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
            reinterpret_cast<half*>(precondition_workspace.mutable_data_ptr()),
            rows,
            static_cast<int>(scale_mode));
      } else {
        qvq_recovery_high_pair_swiglu_precondition_half2_low_kernel<true><<<
            dim3(kHadamardPairMultiblockTiles / 2, rows),
            kHadamardPairMultiblockTile,
            0,
            stream>>>(
            recovery_workspace.const_data_ptr<float>(),
            post_scale0.const_data_ptr<float>(),
            post_scale1.const_data_ptr<float>(),
            bias0.has_value() ? bias0->const_data_ptr<float>() : nullptr,
            bias1.has_value() ? bias1->const_data_ptr<float>() : nullptr,
            reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
            reinterpret_cast<half*>(precondition_workspace.mutable_data_ptr()),
            rows,
            static_cast<int>(scale_mode));
      }
    } else {
      qvq_recovery_high_pair_swiglu_precondition_half2_low_kernel<false><<<
          dim3(kHadamardPairMultiblockTiles / 2, rows),
          kHadamardPairMultiblockTile,
          0,
          stream>>>(
          recovery_workspace.const_data_ptr<float>(),
          post_scale0.const_data_ptr<float>(),
          post_scale1.const_data_ptr<float>(),
          bias0.has_value() ? bias0->const_data_ptr<float>() : nullptr,
          bias1.has_value() ? bias1->const_data_ptr<float>() : nullptr,
          reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
          reinterpret_cast<half*>(precondition_workspace.mutable_data_ptr()),
          rows,
          static_cast<int>(scale_mode));
    }
  } else {
    if (bounded_rounding) {
      qvq_recovery_high_swiglu_precondition_half2_low_kernel<true><<<
          dim3(kHadamardPairMultiblockTiles, rows),
          kHadamardPairMultiblockTile,
          0,
          stream>>>(
          recovery_workspace.const_data_ptr<float>(),
          post_scale0.const_data_ptr<float>(),
          post_scale1.const_data_ptr<float>(),
          bias0.has_value() ? bias0->const_data_ptr<float>() : nullptr,
          bias1.has_value() ? bias1->const_data_ptr<float>() : nullptr,
          reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
          reinterpret_cast<half*>(precondition_workspace.mutable_data_ptr()),
          rows,
          static_cast<int>(scale_mode));
    } else {
      qvq_recovery_high_swiglu_precondition_half2_low_kernel<false><<<
          dim3(kHadamardPairMultiblockTiles, rows),
          kHadamardPairMultiblockTile,
          0,
          stream>>>(
          recovery_workspace.const_data_ptr<float>(),
          post_scale0.const_data_ptr<float>(),
          post_scale1.const_data_ptr<float>(),
          bias0.has_value() ? bias0->const_data_ptr<float>() : nullptr,
          bias1.has_value() ? bias1->const_data_ptr<float>() : nullptr,
          reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
          reinterpret_cast<half*>(precondition_workspace.mutable_data_ptr()),
          rows,
          static_cast<int>(scale_mode));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int output_rows = pad_to_16 ? 16 : rows;
  if (pad_to_16) {
    qvq_swiglu_precondition_multiblock_half2_high_kernel<true><<<
        dim3(
            kHadamardPairMultiblockTile /
                (2 * kHadamardPairMultiblockHalf2HighThreads),
            output_rows),
        kHadamardPairMultiblockHalf2HighThreads,
        0,
        stream>>>(
        reinterpret_cast<const half*>(precondition_workspace.const_data_ptr()),
        reinterpret_cast<half*>(output.mutable_data_ptr()),
        rows);
  } else {
    qvq_swiglu_precondition_multiblock_half2_high_kernel<false><<<
        dim3(
            kHadamardPairMultiblockTile /
                (2 * kHadamardPairMultiblockHalf2HighThreads),
            output_rows),
        kHadamardPairMultiblockHalf2HighThreads,
        0,
        stream>>>(
        reinterpret_cast<const half*>(precondition_workspace.const_data_ptr()),
        reinterpret_cast<half*>(output.mutable_data_ptr()),
        rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_qwen_composite_input_fp16_padded_cuda(
    const at::Tensor& input,
    const at::Tensor& base,
    const at::Tensor& pre_scale) {
  TORCH_CHECK(input.is_cuda() && base.is_cuda() && pre_scale.is_cuda(),
              "Qwen composite input tensors must be CUDA tensors");
  TORCH_CHECK(
      input.device() == base.device() && input.device() == pre_scale.device(),
      "Qwen composite input tensors must share a device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.size(0) >= 1 && input.size(0) <= 16 &&
          input.size(1) == kQwenCompositeN && input.is_contiguous(),
      "Qwen composite input must be contiguous FP16 [1..16, 5120]");
  TORCH_CHECK(
      base.scalar_type() == at::kHalf && base.is_contiguous() &&
          base.numel() == kQwenCompositeBase * kQwenCompositeBase,
      "Qwen composite input base must be contiguous FP16 [40, 40]");
  TORCH_CHECK(
      pre_scale.scalar_type() == at::kHalf && pre_scale.is_contiguous() &&
          pre_scale.numel() == kQwenCompositeN,
      "Qwen composite input scale must be contiguous FP16 [5120]");

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(
      properties.major == 9 && properties.minor == 0 &&
          std::strcmp(properties.name, "NVIDIA H100") == 0,
      "Qwen composite input requires the measured physical H100");
  const size_t smem_bytes = kQwenCompositeN * sizeof(float);
  auto output = at::empty(
      {16, kQwenCompositeN}, input.options().dtype(at::kHalf));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      qvq_qwen_composite_input_fp16_padded_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  qvq_qwen_composite_input_fp16_padded_kernel<<<
      16,
      kHadamardThreads,
      smem_bytes,
      stream>>>(
      reinterpret_cast<const half*>(input.const_data_ptr()),
      reinterpret_cast<const half*>(base.const_data_ptr()),
      reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
      reinterpret_cast<half*>(output.mutable_data_ptr()),
      static_cast<int>(input.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_qwen_composite_recovery_fp32_to_fp16_cuda(
    const at::Tensor& input,
    const at::Tensor& base,
    const at::Tensor& post_scale,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(input.is_cuda() && base.is_cuda() && post_scale.is_cuda(),
              "Qwen composite recovery tensors must be CUDA tensors");
  TORCH_CHECK(
      input.device() == base.device() && input.device() == post_scale.device(),
      "Qwen composite recovery tensors must share a device");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat && input.dim() == 2 &&
          input.size(0) >= 1 && input.size(0) <= 16 &&
          input.is_contiguous(),
      "Qwen composite recovery input must be contiguous FP32 [1..16, N]");
  const int64_t composite_n = input.size(1);
  const int64_t composite_base =
      composite_n == 6144 ? 12 : 40;
  TORCH_CHECK(
      composite_n == 5120 || composite_n == 6144 || composite_n == 10240,
      "Qwen composite recovery supports N=5120, 6144, or 10240");
  TORCH_CHECK(
      base.scalar_type() == at::kHalf && base.is_contiguous() &&
          base.numel() == composite_base * composite_base,
      "Qwen composite recovery base has the wrong square geometry");
  TORCH_CHECK(
      post_scale.scalar_type() == at::kFloat && post_scale.is_contiguous() &&
          post_scale.numel() == composite_n,
      "Qwen composite recovery scale must be contiguous FP32 [N]");
  if (bias.has_value()) {
    TORCH_CHECK(
            bias->is_cuda() && bias->device() == input.device() &&
            bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
            bias->numel() == composite_n,
        "Qwen composite recovery bias must be contiguous FP32 [N]");
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  TORCH_CHECK(
      properties.major == 9 && properties.minor == 0 &&
          std::strcmp(properties.name, "NVIDIA H100") == 0,
      "Qwen composite recovery requires the measured physical H100");
  const size_t smem_bytes = composite_n * sizeof(float);
  auto output = at::empty(input.sizes(), input.options().dtype(at::kHalf));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
#define QVQ_LAUNCH_QWEN_COMPOSITE_MULTIBLOCK(HAS_BIAS, N, BASE)               \
  {                                                                            \
    auto workspace = at::empty(input.sizes(), input.options().dtype(at::kFloat)); \
    constexpr int low_n = N / BASE;                                            \
    constexpr int threads = low_n < 256 ? low_n : 256;                         \
    const dim3 grid(BASE, static_cast<unsigned>(input.size(0)));                \
    qvq_qwen_composite_recovery_low_kernel<N, BASE><<<                          \
        grid, threads, 0, stream>>>(                                            \
        input.const_data_ptr<float>(),                                         \
        workspace.mutable_data_ptr<float>());                                  \
    qvq_qwen_composite_recovery_high_kernel<HAS_BIAS, N, BASE><<<              \
        grid, threads, 0, stream>>>(                                            \
        workspace.const_data_ptr<float>(),                                     \
        reinterpret_cast<const half*>(base.const_data_ptr()),                  \
        post_scale.const_data_ptr<float>(),                                    \
        bias.has_value() ? bias->const_data_ptr<float>() : nullptr,            \
        reinterpret_cast<half*>(output.mutable_data_ptr()));                   \
  }
  if (composite_n == 6144 && bias.has_value()) {
    QVQ_LAUNCH_QWEN_COMPOSITE_MULTIBLOCK(true, 6144, 12)
  } else if (composite_n == 6144) {
    QVQ_LAUNCH_QWEN_COMPOSITE_MULTIBLOCK(false, 6144, 12)
  } else if (composite_n == 10240 && bias.has_value()) {
    QVQ_LAUNCH_QWEN_COMPOSITE_MULTIBLOCK(true, 10240, 40)
  } else if (composite_n == 10240) {
    QVQ_LAUNCH_QWEN_COMPOSITE_MULTIBLOCK(false, 10240, 40)
  }
#undef QVQ_LAUNCH_QWEN_COMPOSITE_MULTIBLOCK

  if (composite_n != 5120) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
#define QVQ_LAUNCH_QWEN_COMPOSITE_RECOVERY(HAS_BIAS, N, BASE)                \
  {                                                                           \
    auto kernel =                                                             \
        qvq_qwen_composite_recovery_fp32_to_fp16_kernel<                      \
            HAS_BIAS, 1, N, BASE>;                                            \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                       \
        kernel,                                                               \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                          \
        static_cast<int>(smem_bytes)));                                       \
    kernel<<<                                                                 \
        static_cast<unsigned>(input.size(0)),                                 \
        kHadamardThreads,                                                      \
        smem_bytes,                                                           \
        stream>>>(                                                            \
        input.const_data_ptr<float>(),                                        \
        reinterpret_cast<const half*>(base.const_data_ptr()),                 \
        post_scale.const_data_ptr<float>(),                                   \
        bias.has_value() ? bias->const_data_ptr<float>() : nullptr,           \
        reinterpret_cast<half*>(output.mutable_data_ptr()));                  \
  }
  if (bias.has_value()) {
    QVQ_LAUNCH_QWEN_COMPOSITE_RECOVERY(true, 5120, 40)
  } else {
    QVQ_LAUNCH_QWEN_COMPOSITE_RECOVERY(false, 5120, 40)
  }
#undef QVQ_LAUNCH_QWEN_COMPOSITE_RECOVERY
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_qwen_composite_ordered_recovery_fp32_to_fp16_cuda(
    const at::Tensor& partials,
    const at::Tensor& base,
    const at::Tensor& post_scale,
    const std::optional<at::Tensor>& bias,
    int64_t split_count,
    int64_t logical_rows) {
  TORCH_CHECK(partials.is_cuda() && base.is_cuda() && post_scale.is_cuda(),
              "Qwen ordered composite recovery tensors must be CUDA tensors");
  TORCH_CHECK(
      partials.device() == base.device() && partials.device() == post_scale.device(),
      "Qwen ordered composite recovery tensors must share a device");
  TORCH_CHECK(
      partials.scalar_type() == at::kFloat && partials.is_contiguous() &&
          (split_count == 17 || split_count == 34) &&
          partials.numel() == split_count * 16 * kQwenCompositeN,
      "Qwen ordered composite partials must be contiguous FP32 split-major M16x5120");
  TORCH_CHECK(logical_rows >= 1 && logical_rows <= 16,
              "Qwen ordered composite recovery requires one through sixteen rows");
  TORCH_CHECK(
      base.scalar_type() == at::kHalf && base.is_contiguous() &&
          base.numel() == kQwenCompositeBase * kQwenCompositeBase,
      "Qwen ordered composite recovery base must be contiguous FP16 [40, 40]");
  TORCH_CHECK(
      post_scale.scalar_type() == at::kFloat && post_scale.is_contiguous() &&
          post_scale.numel() == kQwenCompositeN,
      "Qwen ordered composite recovery scale must be contiguous FP32 [5120]");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->is_cuda() && bias->device() == partials.device() &&
            bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
            bias->numel() == kQwenCompositeN,
        "Qwen ordered composite recovery bias must be contiguous FP32 [5120]");
  }

  const c10::cuda::CUDAGuard device_guard(partials.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, partials.get_device()));
  TORCH_CHECK(
      properties.major == 9 && properties.minor == 0 &&
          std::strcmp(properties.name, "NVIDIA H100") == 0,
      "Qwen ordered composite recovery requires the measured physical H100");
  const size_t smem_bytes = kQwenCompositeN * sizeof(float);
  auto output = at::empty(
      {logical_rows, kQwenCompositeN}, partials.options().dtype(at::kHalf));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(partials.get_device());
#define QVQ_LAUNCH_QWEN_ORDERED_COMPOSITE_RECOVERY(HAS_BIAS, SPLIT_COUNT)      \
  {                                                                            \
    auto kernel = qvq_qwen_composite_recovery_fp32_to_fp16_kernel<             \
        HAS_BIAS, SPLIT_COUNT>;                                                 \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                        \
        kernel,                                                                \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                           \
        static_cast<int>(smem_bytes)));                                        \
    kernel<<<                                                                  \
        static_cast<unsigned>(logical_rows),                                   \
        kHadamardThreads,                                                       \
        smem_bytes,                                                            \
        stream>>>(                                                             \
        partials.const_data_ptr<float>(),                                      \
        reinterpret_cast<const half*>(base.const_data_ptr()),                  \
        post_scale.const_data_ptr<float>(),                                    \
        bias.has_value() ? bias->const_data_ptr<float>() : nullptr,            \
        reinterpret_cast<half*>(output.mutable_data_ptr()));                   \
  }
  if (split_count == 17) {
    if (bias.has_value()) {
      QVQ_LAUNCH_QWEN_ORDERED_COMPOSITE_RECOVERY(true, 17)
    } else {
      QVQ_LAUNCH_QWEN_ORDERED_COMPOSITE_RECOVERY(false, 17)
    }
  } else if (bias.has_value()) {
    QVQ_LAUNCH_QWEN_ORDERED_COMPOSITE_RECOVERY(true, 34)
  } else {
    QVQ_LAUNCH_QWEN_ORDERED_COMPOSITE_RECOVERY(false, 34)
  }
#undef QVQ_LAUNCH_QWEN_ORDERED_COMPOSITE_RECOVERY
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_folded_swiglu_precondition_fp32_cuda(
    const at::Tensor& gate,
    const at::Tensor& up,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale,
    const std::optional<at::Tensor>& gate_bias,
    const std::optional<at::Tensor>& up_bias,
    const at::Tensor& down_scale) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(),
              "folded SwiGLU inputs must be CUDA tensors");
  TORCH_CHECK(
      gate.device() == up.device() && gate.device() == gate_scale.device() &&
          gate.device() == up_scale.device() && gate.device() == down_scale.device(),
      "folded SwiGLU tensors must share a device");
  TORCH_CHECK(gate.scalar_type() == at::kFloat && up.scalar_type() == at::kFloat,
              "folded SwiGLU inputs must be float32");
  TORCH_CHECK(gate.dim() == 2 && gate.sizes() == up.sizes() &&
                  gate.is_contiguous() && up.is_contiguous(),
              "folded SwiGLU inputs must be equal contiguous 2D tensors");
  const int64_t rows = gate.size(0);
  const int64_t n64 = gate.size(1);
  TORCH_CHECK(rows >= 1 && rows <= 16,
              "folded SwiGLU requires one through sixteen rows");
  TORCH_CHECK(n64 > 0 && n64 <= std::numeric_limits<int>::max(),
              "folded SwiGLU width exceeds int32 range");
  for (const auto& named : {
           std::pair<const char*, const at::Tensor&>{"gate_scale", gate_scale},
           {"up_scale", up_scale}}) {
    TORCH_CHECK(
        named.second.scalar_type() == at::kFloat && named.second.is_contiguous() &&
            named.second.numel() == n64,
        named.first, " must be contiguous float32 with one value per column");
  }
  TORCH_CHECK(
      down_scale.scalar_type() == at::kHalf && down_scale.is_contiguous() &&
          down_scale.numel() == n64,
      "down_scale must be contiguous float16 with one value per column");
  auto validate_bias = [&](const std::optional<at::Tensor>& bias, const char* name) {
    if (!bias.has_value()) {
      return;
    }
    TORCH_CHECK(
        bias->device() == gate.device() && bias->scalar_type() == at::kFloat &&
            bias->is_contiguous() && bias->numel() == n64,
        name, " must be contiguous float32 with one value per column");
  };
  validate_bias(gate_bias, "gate_bias");
  validate_bias(up_bias, "up_bias");
  const c10::cuda::CUDAGuard device_guard(gate.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, gate.get_device()));
  TORCH_CHECK(properties.major == 9,
              "folded SwiGLU precondition requires Hopper SM90");

  const int n = static_cast<int>(n64);
  auto output = at::empty({16, n64}, gate.options().dtype(at::kHalf));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(gate.get_device());
  if (rows < 16) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.mutable_data_ptr<at::Half>(),
        0,
        static_cast<size_t>(output.numel()) * sizeof(at::Half),
        stream));
  }
  constexpr int kThreads = 256;
  const int64_t logical_values = rows * n64;
  const int blocks = static_cast<int>((logical_values + kThreads - 1) / kThreads);
  const float* gate_bias_ptr = gate_bias.has_value()
      ? gate_bias->const_data_ptr<float>()
      : nullptr;
  const float* up_bias_ptr = up_bias.has_value()
      ? up_bias->const_data_ptr<float>()
      : nullptr;
#define QVQ_LAUNCH_FOLDED_SWIGLU(HAS_GATE_BIAS, HAS_UP_BIAS)                    \
  qvq_folded_swiglu_precondition_fp32_kernel<HAS_GATE_BIAS, HAS_UP_BIAS>       \
      <<<blocks, kThreads, 0, stream>>>(                                        \
          gate.const_data_ptr<float>(),                                         \
          up.const_data_ptr<float>(),                                           \
          gate_scale.const_data_ptr<float>(),                                   \
          up_scale.const_data_ptr<float>(),                                     \
          gate_bias_ptr,                                                        \
          up_bias_ptr,                                                          \
          reinterpret_cast<const half*>(down_scale.const_data_ptr<at::Half>()), \
          reinterpret_cast<half*>(output.mutable_data_ptr<at::Half>()),         \
          logical_values,                                                       \
          n)
  if (gate_bias.has_value() && up_bias.has_value()) {
    QVQ_LAUNCH_FOLDED_SWIGLU(true, true);
  } else if (gate_bias.has_value()) {
    QVQ_LAUNCH_FOLDED_SWIGLU(true, false);
  } else if (up_bias.has_value()) {
    QVQ_LAUNCH_FOLDED_SWIGLU(false, true);
  } else {
    QVQ_LAUNCH_FOLDED_SWIGLU(false, false);
  }
#undef QVQ_LAUNCH_FOLDED_SWIGLU
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_folded_swiglu_precondition_ordered_fp32_cuda(
    const at::Tensor& partials,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale,
    const std::optional<at::Tensor>& gate_bias,
    const std::optional<at::Tensor>& up_bias,
    const at::Tensor& down_scale,
    int64_t split_count,
    int64_t logical_rows) {
  TORCH_CHECK(partials.is_cuda() && partials.scalar_type() == at::kFloat &&
                  partials.is_contiguous(),
              "ordered folded SwiGLU partials must be contiguous CUDA float32");
  TORCH_CHECK(split_count == 5 || split_count == 10,
              "ordered folded SwiGLU requires split count five or ten");
  TORCH_CHECK(logical_rows >= 1 && logical_rows <= 16,
              "ordered folded SwiGLU requires one through sixteen logical rows");
  const int64_t n64 = gate_scale.numel();
  TORCH_CHECK(n64 > 0 && n64 <= std::numeric_limits<int>::max(),
              "ordered folded SwiGLU width exceeds int32 range");
  TORCH_CHECK(
      partials.numel() == 2 * split_count * 16 * n64,
      "ordered folded SwiGLU partial layout must contain two child-major split planes");
  for (const auto& named : {
           std::pair<const char*, const at::Tensor&>{"gate_scale", gate_scale},
           {"up_scale", up_scale}}) {
    TORCH_CHECK(
        named.second.device() == partials.device() &&
            named.second.scalar_type() == at::kFloat &&
            named.second.is_contiguous() && named.second.numel() == n64,
        named.first, " must be contiguous CUDA float32 with one value per column");
  }
  TORCH_CHECK(
      down_scale.device() == partials.device() &&
          down_scale.scalar_type() == at::kHalf && down_scale.is_contiguous() &&
          down_scale.numel() == n64,
      "down_scale must be contiguous CUDA float16 with one value per column");
  auto validate_bias = [&](const std::optional<at::Tensor>& bias, const char* name) {
    if (!bias.has_value()) {
      return;
    }
    TORCH_CHECK(
        bias->device() == partials.device() && bias->scalar_type() == at::kFloat &&
            bias->is_contiguous() && bias->numel() == n64,
        name, " must be contiguous CUDA float32 with one value per column");
  };
  validate_bias(gate_bias, "gate_bias");
  validate_bias(up_bias, "up_bias");

  const c10::cuda::CUDAGuard device_guard(partials.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, partials.get_device()));
  TORCH_CHECK(properties.major == 9,
              "ordered folded SwiGLU precondition requires Hopper SM90");
  const int n = static_cast<int>(n64);
  auto output = at::empty({16, n64}, partials.options().dtype(at::kHalf));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(partials.get_device());
  if (logical_rows < 16) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.mutable_data_ptr<at::Half>(),
        0,
        static_cast<size_t>(output.numel()) * sizeof(at::Half),
        stream));
  }
  constexpr int kThreads = 256;
  const int64_t logical_values = logical_rows * n64;
  const int blocks = static_cast<int>((logical_values + kThreads - 1) / kThreads);
  const float* gate_bias_ptr = gate_bias.has_value()
      ? gate_bias->const_data_ptr<float>()
      : nullptr;
  const float* up_bias_ptr = up_bias.has_value()
      ? up_bias->const_data_ptr<float>()
      : nullptr;
#define QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU_SPLIT(                               \
    SPLIT_COUNT, HAS_GATE_BIAS, HAS_UP_BIAS)                                  \
  qvq_folded_swiglu_precondition_ordered_fp32_kernel<                           \
      SPLIT_COUNT, HAS_GATE_BIAS, HAS_UP_BIAS><<<blocks, kThreads, 0, stream>>>(\
      partials.const_data_ptr<float>(),                                         \
      gate_scale.const_data_ptr<float>(),                                       \
      up_scale.const_data_ptr<float>(),                                         \
      gate_bias_ptr,                                                            \
      up_bias_ptr,                                                              \
      reinterpret_cast<const half*>(down_scale.const_data_ptr<at::Half>()),     \
      reinterpret_cast<half*>(output.mutable_data_ptr<at::Half>()),             \
      logical_values,                                                           \
      n)
#define QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU(SPLIT_COUNT)                         \
  do {                                                                          \
    if (gate_bias.has_value() && up_bias.has_value()) {                         \
      QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU_SPLIT(SPLIT_COUNT, true, true);          \
    } else if (gate_bias.has_value()) {                                         \
      QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU_SPLIT(SPLIT_COUNT, true, false);         \
    } else if (up_bias.has_value()) {                                           \
      QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU_SPLIT(SPLIT_COUNT, false, true);         \
    } else {                                                                    \
      QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU_SPLIT(SPLIT_COUNT, false, false);        \
    }                                                                           \
  } while (false)
  if (split_count == 5) {
    QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU(5);
  } else {
    QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU(10);
  }
#undef QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU
#undef QVQ_LAUNCH_ORDERED_FOLDED_SWIGLU_SPLIT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_swiglu_precondition_cuda(
    const at::Tensor& activated_gate,
    const at::Tensor& up,
    const at::Tensor& pre_scale) {
  TORCH_CHECK(activated_gate.is_cuda() && up.is_cuda() && pre_scale.is_cuda(),
              "SwiGLU precondition tensors must be CUDA tensors");
  TORCH_CHECK(activated_gate.device() == up.device() && activated_gate.device() == pre_scale.device(),
              "SwiGLU precondition tensors must share a device");
  TORCH_CHECK(activated_gate.scalar_type() == at::kHalf && up.scalar_type() == at::kHalf &&
                  pre_scale.scalar_type() == at::kHalf,
              "SwiGLU precondition tensors must be float16");
  TORCH_CHECK(activated_gate.sizes() == up.sizes(),
              "activated gate and up tensors must have identical shapes");
  TORCH_CHECK(activated_gate.dim() >= 1, "SwiGLU precondition inputs must be at least rank one");
  TORCH_CHECK(activated_gate.is_contiguous() && up.is_contiguous() && pre_scale.is_contiguous(),
              "SwiGLU precondition tensors must be contiguous");
  const int64_t n64 = activated_gate.size(-1);
  TORCH_CHECK(n64 >= 2 && (n64 & (n64 - 1)) == 0,
              "SwiGLU precondition requires a power-of-two last dimension");
  TORCH_CHECK(n64 <= 16384, "SwiGLU precondition supports a last dimension up to 16384");
  TORCH_CHECK(pre_scale.numel() == n64, "SwiGLU precondition scale must have one value per column");
  const int n = static_cast<int>(n64);
  const int64_t rows64 = activated_gate.numel() / n;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "SwiGLU precondition row count exceeds launch limit");
  at::Tensor output = at::empty_like(activated_gate);
  if (rows64 == 0) {
    return output;
  }

  const c10::cuda::CUDAGuard device_guard(activated_gate.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, activated_gate.get_device()));
  const size_t smem_bytes = static_cast<size_t>(n + n / 32) * sizeof(half);
  TORCH_CHECK(smem_bytes <= static_cast<size_t>(properties.sharedMemPerBlockOptin),
              "SwiGLU precondition width exceeds the device dynamic shared-memory limit");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      qvq_swiglu_precondition_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(activated_gate.get_device());
  qvq_swiglu_precondition_kernel<<<static_cast<unsigned int>(rows64), kHadamardThreads, smem_bytes, stream>>>(
      reinterpret_cast<const half*>(activated_gate.const_data_ptr()),
      reinterpret_cast<const half*>(up.const_data_ptr()),
      reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
      reinterpret_cast<half*>(output.mutable_data_ptr()),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qvq_swiglu_precondition_multiblock_cuda(
    const at::Tensor& activated_gate,
    const at::Tensor& up,
    const at::Tensor& pre_scale,
    bool half2_high,
    bool fuse_silu,
    bool half2_low,
    bool pad_to_16) {
  TORCH_CHECK(activated_gate.is_cuda() && up.is_cuda() && pre_scale.is_cuda(),
              "multiblock SwiGLU precondition tensors must be CUDA tensors");
  TORCH_CHECK(activated_gate.device() == up.device() && activated_gate.device() == pre_scale.device(),
              "multiblock SwiGLU precondition tensors must share a device");
  TORCH_CHECK(activated_gate.scalar_type() == at::kHalf && up.scalar_type() == at::kHalf &&
                  pre_scale.scalar_type() == at::kHalf,
              "multiblock SwiGLU precondition tensors must be float16");
  TORCH_CHECK(activated_gate.sizes() == up.sizes(),
              "multiblock activated gate and up tensors must have identical shapes");
  TORCH_CHECK(activated_gate.dim() >= 1 && activated_gate.is_contiguous() &&
                  up.is_contiguous() && pre_scale.is_contiguous(),
              "multiblock SwiGLU precondition tensors must be contiguous with rank >= 1");
  TORCH_CHECK(activated_gate.size(-1) == kHadamardPairMultiblockN,
              "multiblock SwiGLU precondition requires last dimension 8192");
  TORCH_CHECK(pre_scale.numel() == kHadamardPairMultiblockN,
              "multiblock SwiGLU precondition scale must have 8192 values");
  const int64_t rows64 = activated_gate.numel() / kHadamardPairMultiblockN;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(),
              "multiblock SwiGLU precondition row count exceeds launch limit");
  TORCH_CHECK(!pad_to_16 || (activated_gate.dim() == 2 && rows64 > 0 && rows64 <= 16),
              "padded multiblock SwiGLU precondition requires a nonempty 2D input with at most 16 rows");
  at::Tensor output = pad_to_16
      ? at::empty({16, kHadamardPairMultiblockN}, activated_gate.options())
      : at::empty_like(activated_gate);
  if (rows64 == 0) {
    return output;
  }

  const c10::cuda::CUDAGuard device_guard(activated_gate.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, activated_gate.get_device()));
  TORCH_CHECK(properties.major == 9,
              "multiblock SwiGLU precondition is a Hopper-only operator");
  at::Tensor workspace = at::empty_like(activated_gate);
  const int rows = static_cast<int>(rows64);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(activated_gate.get_device());
  if (half2_low && fuse_silu) {
    qvq_swiglu_precondition_multiblock_half2_low_kernel<true><<<
        dim3(kHadamardPairMultiblockTiles, rows),
        kHadamardPairMultiblockHalf2LowThreads,
        0,
        stream>>>(
        reinterpret_cast<const half*>(activated_gate.const_data_ptr()),
        reinterpret_cast<const half*>(up.const_data_ptr()),
        reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
        reinterpret_cast<half*>(workspace.mutable_data_ptr()));
  } else if (half2_low) {
    qvq_swiglu_precondition_multiblock_half2_low_kernel<false><<<
        dim3(kHadamardPairMultiblockTiles, rows),
        kHadamardPairMultiblockHalf2LowThreads,
        0,
        stream>>>(
        reinterpret_cast<const half*>(activated_gate.const_data_ptr()),
        reinterpret_cast<const half*>(up.const_data_ptr()),
        reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
        reinterpret_cast<half*>(workspace.mutable_data_ptr()));
  } else if (fuse_silu) {
    qvq_swiglu_precondition_multiblock_low_kernel<true><<<
        dim3(kHadamardPairMultiblockTiles, rows),
        kHadamardPairMultiblockTile,
        0,
        stream>>>(
        reinterpret_cast<const half*>(activated_gate.const_data_ptr()),
        reinterpret_cast<const half*>(up.const_data_ptr()),
        reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
        reinterpret_cast<half*>(workspace.mutable_data_ptr()));
  } else {
    qvq_swiglu_precondition_multiblock_low_kernel<false><<<
        dim3(kHadamardPairMultiblockTiles, rows),
        kHadamardPairMultiblockTile,
        0,
        stream>>>(
        reinterpret_cast<const half*>(activated_gate.const_data_ptr()),
        reinterpret_cast<const half*>(up.const_data_ptr()),
        reinterpret_cast<const half*>(pre_scale.const_data_ptr()),
        reinterpret_cast<half*>(workspace.mutable_data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int output_rows = pad_to_16 ? 16 : rows;
  if (half2_high) {
    if (pad_to_16) {
      qvq_swiglu_precondition_multiblock_half2_high_kernel<true><<<
          dim3(
              kHadamardPairMultiblockTile /
                  (2 * kHadamardPairMultiblockHalf2HighThreads),
              output_rows),
          kHadamardPairMultiblockHalf2HighThreads,
          0,
          stream>>>(
          reinterpret_cast<const half*>(workspace.const_data_ptr()),
          reinterpret_cast<half*>(output.mutable_data_ptr()),
          rows);
    } else {
      qvq_swiglu_precondition_multiblock_half2_high_kernel<false><<<
          dim3(
              kHadamardPairMultiblockTile /
                  (2 * kHadamardPairMultiblockHalf2HighThreads),
              output_rows),
          kHadamardPairMultiblockHalf2HighThreads,
          0,
          stream>>>(
          reinterpret_cast<const half*>(workspace.const_data_ptr()),
          reinterpret_cast<half*>(output.mutable_data_ptr()),
          rows);
    }
  } else {
    if (pad_to_16) {
      qvq_swiglu_precondition_multiblock_high_kernel<true><<<
          dim3(
              kHadamardPairMultiblockTile / kHadamardPairMultiblockHighThreads,
              output_rows),
          kHadamardPairMultiblockHighThreads,
          0,
          stream>>>(
          reinterpret_cast<const half*>(workspace.const_data_ptr()),
          reinterpret_cast<half*>(output.mutable_data_ptr()),
          rows);
    } else {
      qvq_swiglu_precondition_multiblock_high_kernel<false><<<
          dim3(
              kHadamardPairMultiblockTile / kHadamardPairMultiblockHighThreads,
              output_rows),
          kHadamardPairMultiblockHighThreads,
          0,
          stream>>>(
          reinterpret_cast<const half*>(workspace.const_data_ptr()),
          reinterpret_cast<half*>(output.mutable_data_ptr()),
          rows);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace



namespace {

// torch rejects duplicate def() calls even for byte-identical schemas, and
// cross-bundle schema lookups are unreliable during JIT-plugin static init.
// The CPU and CUDA QVQ extensions intentionally share these op schemas, so
// each def runs at most once per process: a duplicate registration throws
// c10::Error before mutating dispatcher state and is swallowed here.
template <typename DefFn>
void qvq_def_shared_schema(DefFn&& def_fn) {
  try {
    def_fn();
  } catch (const std::exception&) {
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  qvq_def_shared_schema([&] {
    m.def("hadamard(Tensor input, Tensor? pre_scale, Tensor? post_scale, Tensor? bias, int scale_mode, bool pad_to_16=False, bool output_fp16=False) -> Tensor");
});
  m.def("hadamard_pair_fp32_to_fp16(Tensor input0, Tensor input1, Tensor post_scale0, Tensor post_scale1, Tensor? bias0, Tensor? bias1, int scale_mode) -> (Tensor, Tensor)");
  m.def("hadamard_input_fp16_padded_multiblock(Tensor input, Tensor pre_scale) -> Tensor");
  m.def("hadamard_ordered_split16_fp32_to_fp16(Tensor partial_input, Tensor post_scale, Tensor? bias, int scale_mode, int logical_rows, bool multiblock=False) -> Tensor");
  m.def("hadamard_pair_fp32_to_fp16_multiblock(Tensor input0, Tensor input1, Tensor post_scale0, Tensor post_scale1, Tensor? bias0, Tensor? bias1, int scale_mode, bool warp_low=False) -> (Tensor, Tensor)");
  m.def("hadamard_pair_swiglu_precondition_multiblock(Tensor input0, Tensor input1, Tensor post_scale0, Tensor post_scale1, Tensor? bias0, Tensor? bias1, Tensor pre_scale, int scale_mode, bool pad_to_16=False, bool pair_tiles=False, bool bounded_rounding=False, bool packed_gate_up=False) -> Tensor");
  m.def("folded_swiglu_precondition_fp32(Tensor gate, Tensor up, Tensor gate_scale, Tensor up_scale, Tensor? gate_bias, Tensor? up_bias, Tensor down_scale) -> Tensor");
  m.def("folded_swiglu_precondition_ordered_fp32(Tensor partials, Tensor gate_scale, Tensor up_scale, Tensor? gate_bias, Tensor? up_bias, Tensor down_scale, int split_count, int logical_rows) -> Tensor");
  m.def("qwen_composite_recovery_fp32_to_fp16(Tensor input, Tensor base, Tensor post_scale, Tensor? bias) -> Tensor");
  m.def("qwen_composite_ordered_recovery_fp32_to_fp16(Tensor partials, Tensor base, Tensor post_scale, Tensor? bias, int split_count, int logical_rows) -> Tensor");
  m.def("qwen_composite_input_fp16_padded(Tensor input, Tensor base, Tensor pre_scale) -> Tensor");
  m.def("swiglu_precondition(Tensor activated_gate, Tensor up, Tensor pre_scale) -> Tensor");
  m.def("swiglu_precondition_multiblock(Tensor activated_gate, Tensor up, Tensor pre_scale, bool half2_high=False, bool fuse_silu=False, bool half2_low=False, bool pad_to_16=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("hadamard", &qvq_hadamard_cuda);
  m.impl("hadamard_pair_fp32_to_fp16", &qvq_hadamard_pair_fp32_to_fp16_cuda);
  m.impl("hadamard_input_fp16_padded_multiblock", &qvq_hadamard_input_fp16_padded_multiblock_cuda);
  m.impl("hadamard_ordered_split16_fp32_to_fp16", &qvq_hadamard_ordered_split16_fp32_to_fp16_cuda);
  m.impl("hadamard_pair_fp32_to_fp16_multiblock", &qvq_hadamard_pair_fp32_to_fp16_multiblock_cuda);
  m.impl("hadamard_pair_swiglu_precondition_multiblock", &qvq_hadamard_pair_swiglu_precondition_multiblock_cuda);
  m.impl("folded_swiglu_precondition_fp32", &qvq_folded_swiglu_precondition_fp32_cuda);
  m.impl("folded_swiglu_precondition_ordered_fp32", &qvq_folded_swiglu_precondition_ordered_fp32_cuda);
  m.impl("qwen_composite_recovery_fp32_to_fp16", &qvq_qwen_composite_recovery_fp32_to_fp16_cuda);
  m.impl("qwen_composite_ordered_recovery_fp32_to_fp16", &qvq_qwen_composite_ordered_recovery_fp32_to_fp16_cuda);
  m.impl("qwen_composite_input_fp16_padded", &qvq_qwen_composite_input_fp16_padded_cuda);
  m.impl("swiglu_precondition", &qvq_swiglu_precondition_cuda);
  m.impl("swiglu_precondition_multiblock", &qvq_swiglu_precondition_multiblock_cuda);
}
