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

__device__ __forceinline__ float round_fp16_unless_overflow(float value) {
  const float narrowed = __half2float(__float2half_rn(value));
  return isfinite(narrowed) ? narrowed : value;
}

// One block per row; the row lives in dynamic shared memory. log2n must be >= 1
// (n >= 2) and n * sizeof(Scalar) must fit the device's dynamic shared limit.
template <typename Scalar>
__global__ void __launch_bounds__(kHadamardThreads) qvq_hadamard_kernel(
    const Scalar* __restrict__ input,
    Scalar* __restrict__ output,
    const Scalar* __restrict__ pre_scale,   // optional, [n]
    const Scalar* __restrict__ post_scale,  // optional, [n]
    const Scalar* __restrict__ bias,        // optional, [n]
    int n,
    int scale_mode) {
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
  Scalar* out_row = output + static_cast<int64_t>(row) * n;

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
    out_row[i] = QvqHadamardTraits<Scalar>::from_float(value);
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
    int64_t scale_mode) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() >= 1, "input must be at least rank one");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "hadamard requires float16, bfloat16, or float32 input");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(scale_mode >= 0 && scale_mode <= 4,
              "scale_mode must be 0/1 (native), 2 (range-safe pre-scale), or 3/4 (FP16 emulation)");
  TORCH_CHECK(scale_mode != 2 || input.scalar_type() == at::kHalf,
              "range-safe pre-scale mode 2 requires float16 input");
  TORCH_CHECK(scale_mode < 3 || input.scalar_type() == at::kFloat,
              "FP16-emulation scale modes 3/4 require float32 input");
  const int64_t n64 = input.size(-1);
  TORCH_CHECK(n64 >= 2 && (n64 & (n64 - 1)) == 0, "hadamard requires a power-of-two last dim");
  TORCH_CHECK(n64 <= 16384, "hadamard fused kernel supports last dim up to 16384");
  const int n = static_cast<int>(n64);
  const int64_t rows = input.numel() / n;
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "row count exceeds int32 kernel limit");

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

  at::Tensor output = at::empty_like(input);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const dim3 grid(static_cast<unsigned int>(rows));
#define QVQ_HADAMARD_LAUNCH(SCALAR)                                                                                \
  {                                                                                                               \
    const SCALAR* in_ptr = reinterpret_cast<const SCALAR*>(input.const_data_ptr());                                \
    SCALAR* out_ptr = reinterpret_cast<SCALAR*>(output.mutable_data_ptr());                                        \
    const SCALAR* pre = pre_scale.has_value()                                                                      \
        ? reinterpret_cast<const SCALAR*>(pre_scale->const_data_ptr())                                             \
        : nullptr;                                                                                                 \
    const SCALAR* post = post_scale.has_value()                                                                    \
        ? reinterpret_cast<const SCALAR*>(post_scale->const_data_ptr())                                            \
        : nullptr;                                                                                                 \
    const SCALAR* bia = bias.has_value()                                                                           \
        ? reinterpret_cast<const SCALAR*>(bias->const_data_ptr())                                                  \
        : nullptr;                                                                                                 \
    qvq_hadamard_kernel<SCALAR><<<grid, kHadamardThreads, smem_bytes, stream>>>(                                   \
        in_ptr, out_ptr, pre, post, bia, n, static_cast<int>(scale_mode));                                          \
  }
  if (input.scalar_type() == at::kHalf) {
    QVQ_HADAMARD_LAUNCH(half)
  } else if (input.scalar_type() == at::kBFloat16) {
    QVQ_HADAMARD_LAUNCH(nv_bfloat16)
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qvq_hadamard_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    QVQ_HADAMARD_LAUNCH(float)
  }
#undef QVQ_HADAMARD_LAUNCH
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
    m.def("hadamard(Tensor input, Tensor? pre_scale, Tensor? post_scale, Tensor? bias, int scale_mode) -> Tensor");
});
  m.def("hadamard_pair_fp32_to_fp16(Tensor input0, Tensor input1, Tensor post_scale0, Tensor post_scale1, Tensor? bias0, Tensor? bias1, int scale_mode) -> (Tensor, Tensor)");
  m.def("hadamard_pair_fp32_to_fp16_multiblock(Tensor input0, Tensor input1, Tensor post_scale0, Tensor post_scale1, Tensor? bias0, Tensor? bias1, int scale_mode, bool warp_low=False) -> (Tensor, Tensor)");
  m.def("swiglu_precondition(Tensor activated_gate, Tensor up, Tensor pre_scale) -> Tensor");
  m.def("swiglu_precondition_multiblock(Tensor activated_gate, Tensor up, Tensor pre_scale, bool half2_high=False, bool fuse_silu=False, bool half2_low=False, bool pad_to_16=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("hadamard", &qvq_hadamard_cuda);
  m.impl("hadamard_pair_fp32_to_fp16", &qvq_hadamard_pair_fp32_to_fp16_cuda);
  m.impl("hadamard_pair_fp32_to_fp16_multiblock", &qvq_hadamard_pair_fp32_to_fp16_multiblock_cuda);
  m.impl("swiglu_precondition", &qvq_swiglu_precondition_cuda);
  m.impl("swiglu_precondition_multiblock", &qvq_swiglu_precondition_multiblock_cuda);
}
