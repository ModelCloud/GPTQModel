// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kGroupSize = 128;
constexpr int kPackFactor = 8;
constexpr int kPackedRowsPerGroup = kGroupSize / kPackFactor;
constexpr int kBlockN = 16;
constexpr int kSharedStride = kBlockN + 2;

constexpr int kV0SizeK = 4096;
constexpr int kV0Groups = kV0SizeK / kGroupSize;
constexpr int kV0Warps = kV0Groups / 2;
constexpr int kV0Threads = kV0Warps * 32;

constexpr int kGeneralMaxWarps = 8;
constexpr int kGeneralMaxThreads = kGeneralMaxWarps * 32;

constexpr int kWideSizeK = 12288;
constexpr int kWideGroups = kWideSizeK / kGroupSize;
constexpr int kWideWarps = 16;
constexpr int kWideThreads = kWideWarps * 32;

constexpr int kMultiRowWarps = 16;
constexpr int kMultiRowThreads = kMultiRowWarps * 32;
constexpr int kMultiRowMaxRows = 4;

constexpr int kHmmaBlockM = 16;
constexpr int kHmmaBlockN = 64;
constexpr int kHmmaBlockK = kGroupSize;
constexpr int kHmmaWarps = kHmmaBlockN / 16;
constexpr int kHmmaThreads = kHmmaWarps * 32;
constexpr int kHmmaPackedKWords = kHmmaBlockK / kPackFactor;
constexpr int kHmmaReuseBlockM = 64;
constexpr int kHmmaReuseWarps = 8;
constexpr int kHmmaReuseThreads = kHmmaReuseWarps * 32;
constexpr int kHmmaFragmentElements = 16 * 16;
constexpr int kCpAsyncBytes = 16;
constexpr int kScalarsPerCpAsync = kCpAsyncBytes / sizeof(half);
constexpr int kHmmaReuseACopies =
    kHmmaReuseBlockM * kHmmaBlockK / kScalarsPerCpAsync;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;
constexpr int kMmaLanes = 32;
constexpr int kMmaLaneTileN = 16;
constexpr int kMmaLaneWords = kMmaLanes;
constexpr int kMmaLaneM32BlockM = 32;
constexpr int kMmaLaneM32WarpN = 32;
constexpr int kMmaLaneM32Warps = 4;
constexpr int kMmaLaneM32Threads = kMmaLaneM32Warps * kMmaLanes;
constexpr int kMmaLaneM32N32BlockN = 32;
constexpr int kMmaLaneM32N32Warps = 2;
constexpr int kMmaLaneM32N32Threads = kMmaLaneM32N32Warps * kMmaLanes;
constexpr int kMmaLanePaddedBlockM = 16;
constexpr int kMmaLanePaddedBlockN = 16;
constexpr int kMmaLanePaddedThreads = kMmaLanes;
constexpr int kMmaLaneSplitK4Warps = 4;
constexpr int kMmaLaneSplitK8Warps = 8;
constexpr int kMmaLaneSplitK12Warps = 12;
constexpr int kMmaLaneSplitK16Warps = 16;
constexpr int kMmaLaneSplitK24Warps = 24;
constexpr int kMmaLaneSplitK4Threads = kMmaLaneSplitK4Warps * kMmaLanes;
constexpr int kMmaLaneSplitK8Threads = kMmaLaneSplitK8Warps * kMmaLanes;
constexpr int kMmaLaneSplitK12Threads = kMmaLaneSplitK12Warps * kMmaLanes;
constexpr int kMmaLaneSplitK16Threads = kMmaLaneSplitK16Warps * kMmaLanes;
constexpr int kMmaLaneSplitK24Threads = kMmaLaneSplitK24Warps * kMmaLanes;
constexpr int kMmaLaneSplitKFragments = 2;
constexpr int kMmaLaneSplitKN32Fragments = 4;
constexpr int kMmaLaneSplitKN64Fragments = 8;
constexpr int kMmaLaneAccumulatorValues = 4;
constexpr int kMmaLaneSplitKN64K12SharedBytes =
    kMmaLaneSplitK12Warps *
    kMmaLaneSplitKN64Fragments *
    kMmaLanes *
    kMmaLaneAccumulatorValues *
    sizeof(float);
constexpr int kMmaLaneSplitKN64SharedBytes =
    kMmaLaneSplitK24Warps *
    kMmaLaneSplitKN64Fragments *
    kMmaLanes *
    kMmaLaneAccumulatorValues *
    sizeof(float);

enum class HmmaLaunchMode {
  kAuto,
  kV0,
  kM64V1,
  kM64V2,
  kM64V2SyncA128,
  kM64V3,
};

static_assert(kV0Groups == 32);
static_assert(kV0Warps == 16);
static_assert(kV0Threads == 512);
static_assert(kGeneralMaxThreads == 256);
static_assert(kWideGroups == 96);
static_assert(kWideThreads == 512);
static_assert(kMultiRowThreads == 512);
static_assert(kHmmaWarps == 4);
static_assert(kHmmaThreads == 128);
static_assert(kHmmaPackedKWords == kPackedRowsPerGroup);
static_assert(kHmmaReuseWarps == 8);
static_assert(kHmmaReuseThreads == 256);
static_assert(sizeof(half) == sizeof(__nv_bfloat16));
static_assert(kHmmaBlockK % kScalarsPerCpAsync == 0);
static_assert(kHmmaReuseACopies % kHmmaReuseThreads == 0);
static_assert(kMmaLaneTileN == 2 * kMmaN);
static_assert(kMmaLaneM32BlockM == 2 * kMmaM);
static_assert(kMmaLaneM32WarpN == 2 * kMmaLaneTileN);
static_assert(kMmaLaneM32Threads == 128);
static_assert(kMmaLaneM32N32BlockN == kMmaLaneM32WarpN);
static_assert(kMmaLaneM32N32Threads == 64);
static_assert(kMmaLanePaddedBlockM == kMmaM);
static_assert(kMmaLanePaddedBlockN == kMmaLaneTileN);
static_assert(kMmaLanePaddedThreads == 32);
static_assert(kMmaLaneSplitK4Threads == 128);
static_assert(kMmaLaneSplitK8Threads == 256);
static_assert(kMmaLaneSplitK12Threads == 384);
static_assert(kMmaLaneSplitK16Threads == 512);
static_assert(kMmaLaneSplitK24Threads == 768);
static_assert(kMmaLaneSplitKN64K12SharedBytes == 48 * 1024);
static_assert(kMmaLaneSplitKN64SharedBytes == 96 * 1024);

template <typename Scalar>
struct ScalarTraits;

template <>
struct ScalarTraits<half> {
  using TorchScalar = at::Half;

  __device__ __forceinline__ static float to_float(half value) {
    return __half2float(value);
  }

  __device__ __forceinline__ static half from_float(float value) {
    return __float2half_rn(value);
  }

  __device__ __forceinline__ static uint32_t pair_bits(float low, float high) {
    union {
      half2 values;
      uint32_t bits;
    } packed;
    packed.values = __floats2half2_rn(low, high);
    return packed.bits;
  }
};

template <>
struct ScalarTraits<__nv_bfloat16> {
  using TorchScalar = at::BFloat16;

  __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }

  __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }

  __device__ __forceinline__ static uint32_t pair_bits(float low, float high) {
    union {
      __nv_bfloat162 values;
      uint32_t bits;
    } packed;
    packed.values = __floats2bfloat162_rn(low, high);
    return packed.bits;
  }
};

template <typename Scalar>
__device__ __forceinline__ uint32_t dequantized_pair_bits(
    uint32_t word,
    float scale,
    int code_offset) {
  const int low_code = static_cast<int>((word >> (code_offset * 4)) & 0x0fu);
  const int high_code = static_cast<int>((word >> ((code_offset + 1) * 4)) & 0x0fu);
  return ScalarTraits<Scalar>::pair_bits(
      static_cast<float>(low_code - 8) * scale,
      static_cast<float>(high_code - 8) * scale);
}

template <typename Scalar>
__device__ __forceinline__ void store_dequantized_word_128(
    Scalar* destination,
    uint32_t word,
    float scale) {
  const uint4 packed = make_uint4(
      dequantized_pair_bits<Scalar>(word, scale, 0),
      dequantized_pair_bits<Scalar>(word, scale, 2),
      dequantized_pair_bits<Scalar>(word, scale, 4),
      dequantized_pair_bits<Scalar>(word, scale, 6));
  *reinterpret_cast<uint4*>(destination) = packed;
}

struct MmaFragmentA {
  uint32_t values[4];
};

struct MmaFragmentB {
  uint32_t values[2];
};

struct MmaFragmentC {
  float values[4];
};

__device__ __forceinline__ void load_mma_fragment_a(
    MmaFragmentA& fragment,
    const void* shared_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(fragment.values[0]),
        "=r"(fragment.values[1]),
        "=r"(fragment.values[2]),
        "=r"(fragment.values[3])
      : "r"(shared_address));
#else
  fragment.values[0] = 0;
  fragment.values[1] = 0;
  fragment.values[2] = 0;
  fragment.values[3] = 0;
#endif
}

template <typename Scalar>
struct MmaInstruction;

template <>
struct MmaInstruction<half> {
  __device__ __forceinline__ static void run(
      const MmaFragmentA& a,
      const MmaFragmentB& b,
      MmaFragmentC& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c.values[0]),
          "=f"(c.values[1]),
          "=f"(c.values[2]),
          "=f"(c.values[3])
        : "r"(a.values[0]),
          "r"(a.values[1]),
          "r"(a.values[2]),
          "r"(a.values[3]),
          "r"(b.values[0]),
          "r"(b.values[1]),
          "f"(c.values[0]),
          "f"(c.values[1]),
          "f"(c.values[2]),
          "f"(c.values[3]));
#endif
  }
};

template <>
struct MmaInstruction<__nv_bfloat16> {
  __device__ __forceinline__ static void run(
      const MmaFragmentA& a,
      const MmaFragmentB& b,
      MmaFragmentC& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c.values[0]),
          "=f"(c.values[1]),
          "=f"(c.values[2]),
          "=f"(c.values[3])
        : "r"(a.values[0]),
          "r"(a.values[1]),
          "r"(a.values[2]),
          "r"(a.values[3]),
          "r"(b.values[0]),
          "r"(b.values[1]),
          "f"(c.values[0]),
          "f"(c.values[1]),
          "f"(c.values[2]),
          "f"(c.values[3]));
#endif
  }
};

template <int Lut>
__device__ __forceinline__ uint32_t mma_lop3(
    uint32_t a,
    uint32_t b,
    uint32_t c) {
  uint32_t result;
  asm volatile(
      "lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(result)
      : "r"(a), "r"(b), "r"(c), "n"(Lut));
  return result;
}

template <typename Scalar>
struct MmaLaneDequant;

template <>
struct MmaLaneDequant<half> {
  __device__ __forceinline__ static void run(
      uint32_t packed_word,
      half scale,
      MmaFragmentB& fragment) {
    constexpr uint32_t kLowMask = 0x000f000f;
    constexpr uint32_t kHighMask = 0x00f000f0;
    constexpr uint32_t kExponent = 0x64006400;
    constexpr uint32_t kSubtract = 0x64086408;
    constexpr uint32_t kMultiply = 0x2c002c00;
    constexpr uint32_t kAdd = 0xd480d480;
    constexpr int kLop3AndOr = (0xf0 & 0xcc) | 0xaa;

    const uint32_t low =
        mma_lop3<kLop3AndOr>(packed_word, kLowMask, kExponent);
    const uint32_t high =
        mma_lop3<kLop3AndOr>(packed_word, kHighMask, kExponent);
    union {
      uint32_t bits;
      half2 values;
    } low_pair, high_pair, constant, add_pair, result;
    low_pair.bits = low;
    high_pair.bits = high;
    constant.bits = kSubtract;
    const half2 scale_pair = __half2half2(scale);
    result.values = __hmul2(__hsub2(low_pair.values, constant.values), scale_pair);
    fragment.values[0] = result.bits;
    constant.bits = kMultiply;
    add_pair.bits = kAdd;
    result.values =
        __hfma2(high_pair.values, constant.values, add_pair.values);
    result.values = __hmul2(result.values, scale_pair);
    fragment.values[1] = result.bits;
  }
};

template <>
struct MmaLaneDequant<__nv_bfloat16> {
  __device__ __forceinline__ static void run(
      uint32_t packed_word,
      __nv_bfloat16 scale,
      MmaFragmentB& fragment) {
    constexpr uint32_t kMask = 0x000f000f;
    constexpr uint32_t kExponent = 0x43004300;
    constexpr uint32_t kSubtract = 0x43084308;
    constexpr int kLop3AndOr = (0xf0 & 0xcc) | 0xaa;

    const uint32_t low =
        mma_lop3<kLop3AndOr>(packed_word, kMask, kExponent);
    const uint32_t high =
        mma_lop3<kLop3AndOr>(packed_word >> 4, kMask, kExponent);
    union {
      uint32_t bits;
      __nv_bfloat162 values;
    } low_pair, high_pair, subtract_pair, result;
    low_pair.bits = low;
    high_pair.bits = high;
    subtract_pair.bits = kSubtract;
    const __nv_bfloat162 scale_pair = __bfloat162bfloat162(scale);
    result.values =
        __hmul2(__hsub2(low_pair.values, subtract_pair.values), scale_pair);
    fragment.values[0] = result.bits;
    result.values =
        __hmul2(__hsub2(high_pair.values, subtract_pair.values), scale_pair);
    fragment.values[1] = result.bits;
  }
};

template <typename Scalar>
__device__ __forceinline__ void load_mma_fragment_a_global(
    MmaFragmentA& fragment,
    const Scalar* input,
    int row_stride,
    int tile_row,
    int tile_column,
    int lane) {
  // Invert the four-matrix redistribution performed by ldmatrix.x4 for the
  // row-major m16n8k16 A operand. Each destination register owns one aligned
  // pair from the logical 16x16 tile, but those pairs are not a contiguous
  // eight-element row segment for a lane.
  const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int address_column = (lane >> 4) * 8;
#pragma unroll
  for (int register_index = 0; register_index < 4; ++register_index) {
    const int matrix_column = address_column + register_index * 2;
    const int source_row =
        (address_row >> 2) +
        ((matrix_column & 8) >> 1) +
        ((matrix_column & 2) << 2);
    const int source_column =
        (address_row & 3) * 2 +
        ((matrix_column & 4) << 1);
    fragment.values[register_index] =
        *reinterpret_cast<const uint32_t*>(
            input +
            static_cast<int64_t>(tile_row + source_row) * row_stride +
            tile_column +
            source_column);
  }
}

template <typename Scalar>
__device__ __forceinline__ void load_mma_fragment_a_global_guard_m(
    MmaFragmentA& fragment,
    const Scalar* input,
    int size_m,
    int row_stride,
    int tile_column,
    int lane) {
  const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int address_column = (lane >> 4) * 8;
#pragma unroll
  for (int register_index = 0; register_index < 4; ++register_index) {
    const int matrix_column = address_column + register_index * 2;
    const int source_row =
        (address_row >> 2) +
        ((matrix_column & 8) >> 1) +
        ((matrix_column & 2) << 2);
    const int source_column =
        (address_row & 3) * 2 +
        ((matrix_column & 4) << 1);
    fragment.values[register_index] =
        source_row < size_m
        ? *reinterpret_cast<const uint32_t*>(
              input + static_cast<int64_t>(source_row) * row_stride +
              tile_column + source_column)
        : 0u;
  }
}

template <typename Scalar>
__device__ __forceinline__ void run_mma_lane_tile(
    const MmaFragmentA& fragment_a,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ scales,
    Scalar* __restrict__ output,
    int lane) {
  const int quad = lane >> 2;
  const int thread_in_quad = lane & 3;
  const uint32_t packed_word =
      static_cast<uint32_t>(packed_lane_qweight[lane]);
  MmaFragmentB fragment_b_0;
  MmaFragmentB fragment_b_1;
  MmaLaneDequant<Scalar>::run(
      packed_word,
      scales[quad],
      fragment_b_0);
  MmaLaneDequant<Scalar>::run(
      packed_word >> 8,
      scales[quad + kMmaN],
      fragment_b_1);

  MmaFragmentC accumulator_0 = {};
  MmaFragmentC accumulator_1 = {};
  MmaInstruction<Scalar>::run(fragment_a, fragment_b_0, accumulator_0);
  MmaInstruction<Scalar>::run(fragment_a, fragment_b_1, accumulator_1);

  const int output_row_0 = quad;
  const int output_row_1 = quad + 8;
  const int output_column = thread_in_quad * 2;
  output[output_row_0 * kMmaLaneTileN + output_column] =
      ScalarTraits<Scalar>::from_float(accumulator_0.values[0]);
  output[output_row_0 * kMmaLaneTileN + output_column + 1] =
      ScalarTraits<Scalar>::from_float(accumulator_0.values[1]);
  output[output_row_1 * kMmaLaneTileN + output_column] =
      ScalarTraits<Scalar>::from_float(accumulator_0.values[2]);
  output[output_row_1 * kMmaLaneTileN + output_column + 1] =
      ScalarTraits<Scalar>::from_float(accumulator_0.values[3]);
  output[output_row_0 * kMmaLaneTileN + output_column + kMmaN] =
      ScalarTraits<Scalar>::from_float(accumulator_1.values[0]);
  output[output_row_0 * kMmaLaneTileN + output_column + kMmaN + 1] =
      ScalarTraits<Scalar>::from_float(accumulator_1.values[1]);
  output[output_row_1 * kMmaLaneTileN + output_column + kMmaN] =
      ScalarTraits<Scalar>::from_float(accumulator_1.values[2]);
  output[output_row_1 * kMmaLaneTileN + output_column + kMmaN + 1] =
      ScalarTraits<Scalar>::from_float(accumulator_1.values[3]);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLanes) void amplin_mma_lane_tile_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ scales,
    Scalar* __restrict__ output) {
  __shared__ __align__(32) Scalar shared_a[kMmaM * kMmaK];

  const int lane = threadIdx.x;
  reinterpret_cast<uint4*>(shared_a)[lane] =
      reinterpret_cast<const uint4*>(input)[lane];
  __syncthreads();

  const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int address_column = (lane >> 4) * 8;
  MmaFragmentA fragment_a;
  load_mma_fragment_a(
      fragment_a,
      shared_a + address_row * kMmaK + address_column);
  run_mma_lane_tile(
      fragment_a,
      packed_lane_qweight,
      scales,
      output,
      lane);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLanes) void amplin_mma_lane_tile_global_a_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ scales,
    Scalar* __restrict__ output) {
  const int lane = threadIdx.x;
  MmaFragmentA fragment_a;
  load_mma_fragment_a_global(
      fragment_a,
      input,
      kMmaK,
      0,
      0,
      lane);
  run_mma_lane_tile(
      fragment_a,
      packed_lane_qweight,
      scales,
      output,
      lane);
}

template <typename Scalar>
__device__ __forceinline__ void store_mma_fragment(
    const MmaFragmentC& fragment,
    Scalar* output,
    int size_n,
    int row_base,
    int column_base,
    int lane) {
  const int quad = lane >> 2;
  const int thread_in_quad = lane & 3;
  const int column = column_base + thread_in_quad * 2;
  output[static_cast<int64_t>(row_base + quad) * size_n + column] =
      ScalarTraits<Scalar>::from_float(fragment.values[0]);
  output[static_cast<int64_t>(row_base + quad) * size_n + column + 1] =
      ScalarTraits<Scalar>::from_float(fragment.values[1]);
  output[static_cast<int64_t>(row_base + quad + 8) * size_n + column] =
      ScalarTraits<Scalar>::from_float(fragment.values[2]);
  output[static_cast<int64_t>(row_base + quad + 8) * size_n + column + 1] =
      ScalarTraits<Scalar>::from_float(fragment.values[3]);
}

template <typename Scalar>
__device__ __forceinline__ void store_mma_fragment_guard_m(
    const MmaFragmentC& fragment,
    Scalar* output,
    int size_m,
    int size_n,
    int column_base,
    int lane) {
  const int quad = lane >> 2;
  const int thread_in_quad = lane & 3;
  const int column = column_base + thread_in_quad * 2;
  if (quad < size_m) {
    output[static_cast<int64_t>(quad) * size_n + column] =
        ScalarTraits<Scalar>::from_float(fragment.values[0]);
    output[static_cast<int64_t>(quad) * size_n + column + 1] =
        ScalarTraits<Scalar>::from_float(fragment.values[1]);
  }
  if (quad + 8 < size_m) {
    output[static_cast<int64_t>(quad + 8) * size_n + column] =
        ScalarTraits<Scalar>::from_float(fragment.values[2]);
    output[static_cast<int64_t>(quad + 8) * size_n + column + 1] =
        ScalarTraits<Scalar>::from_float(fragment.values[3]);
  }
}

__device__ __forceinline__ void cp_async_16(
    void* shared_destination,
    const void* global_source);
__device__ __forceinline__ void cp_async_commit_group();
__device__ __forceinline__ void cp_async_wait_all();

template <typename Scalar>
__global__ __launch_bounds__(kHmmaReuseThreads) void amplin_mma_lane_m64_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ __align__(32) Scalar shared_a[kHmmaReuseBlockM * kHmmaBlockK];

  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int warp_n = warp & 3;
  const int warp_m = (warp >> 2) * 32;
  const int quad = lane >> 2;
  const int address_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int address_column = (lane >> 4) * 8;
  const int tile_m = static_cast<int>(blockIdx.y);
  const int tile_n = static_cast<int>(blockIdx.x);
  const int global_m = tile_m * kHmmaReuseBlockM;
  const int global_n = tile_n * kHmmaBlockN + warp_n * kMmaLaneTileN;

  MmaFragmentC accumulator_00 = {};
  MmaFragmentC accumulator_01 = {};
  MmaFragmentC accumulator_10 = {};
  MmaFragmentC accumulator_11 = {};

  for (int group = 0; group < num_groups; ++group) {
    for (
        int copy_index = thread;
        copy_index < kHmmaReuseACopies;
        copy_index += kHmmaReuseThreads) {
      const int row = copy_index / (kHmmaBlockK / kScalarsPerCpAsync);
      const int row_copy =
          copy_index - row * (kHmmaBlockK / kScalarsPerCpAsync);
      const int group_k = row_copy * kScalarsPerCpAsync;
      cp_async_16(
          shared_a + row * kHmmaBlockK + group_k,
          input +
              static_cast<int64_t>(global_m + row) * size_k +
              group * kHmmaBlockK +
              group_k);
    }
    cp_async_commit_group();

    const int scale_column = warp_n * kMmaLaneTileN + quad;
    const int64_t scale_offset =
        (static_cast<int64_t>(tile_n) * num_groups + group) *
        kHmmaBlockN;
    const Scalar scale_0 = packed_scales[scale_offset + scale_column];
    const Scalar scale_1 =
        packed_scales[scale_offset + scale_column + kMmaN];

    cp_async_wait_all();
    __syncthreads();

#pragma unroll
    for (int k_step = 0; k_step < kHmmaBlockK / kMmaK; ++k_step) {
      const int64_t word_offset =
          (((static_cast<int64_t>(tile_n) * num_groups + group) *
                (kHmmaBlockK / kMmaK) +
            k_step) *
               kHmmaWarps +
           warp_n) *
              kMmaLanes +
          lane;
      const uint32_t packed_word =
          static_cast<uint32_t>(packed_lane_qweight[word_offset]);
      MmaFragmentB fragment_b_0;
      MmaFragmentB fragment_b_1;
      MmaLaneDequant<Scalar>::run(
          packed_word,
          scale_0,
          fragment_b_0);
      MmaLaneDequant<Scalar>::run(
          packed_word >> 8,
          scale_1,
          fragment_b_1);

      MmaFragmentA fragment_a_0;
      MmaFragmentA fragment_a_1;
      const int group_k = k_step * kMmaK + address_column;
      load_mma_fragment_a(
          fragment_a_0,
          shared_a +
              (warp_m + address_row) * kHmmaBlockK +
              group_k);
      load_mma_fragment_a(
          fragment_a_1,
          shared_a +
              (warp_m + kMmaM + address_row) * kHmmaBlockK +
              group_k);
      MmaInstruction<Scalar>::run(
          fragment_a_0,
          fragment_b_0,
          accumulator_00);
      MmaInstruction<Scalar>::run(
          fragment_a_0,
          fragment_b_1,
          accumulator_01);
      MmaInstruction<Scalar>::run(
          fragment_a_1,
          fragment_b_0,
          accumulator_10);
      MmaInstruction<Scalar>::run(
          fragment_a_1,
          fragment_b_1,
          accumulator_11);
    }
    __syncthreads();
  }

  store_mma_fragment(
      accumulator_00,
      output,
      size_n,
      global_m + warp_m,
      global_n,
      lane);
  store_mma_fragment(
      accumulator_01,
      output,
      size_n,
      global_m + warp_m,
      global_n + kMmaN,
      lane);
  store_mma_fragment(
      accumulator_10,
      output,
      size_n,
      global_m + warp_m + kMmaM,
      global_n,
      lane);
  store_mma_fragment(
      accumulator_11,
      output,
      size_n,
      global_m + warp_m + kMmaM,
      global_n + kMmaN,
      lane);
}

template <typename Scalar>
__global__ __launch_bounds__(kHmmaReuseThreads) void amplin_mma_lane_m64_global_a_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_n = warp & 3;
  const int warp_m = (warp >> 2) * 32;
  const int quad = lane >> 2;
  const int tile_m = static_cast<int>(blockIdx.y);
  const int tile_n = static_cast<int>(blockIdx.x);
  const int global_m = tile_m * kHmmaReuseBlockM;
  const int global_n = tile_n * kHmmaBlockN + warp_n * kMmaLaneTileN;

  MmaFragmentC accumulator_00 = {};
  MmaFragmentC accumulator_01 = {};
  MmaFragmentC accumulator_10 = {};
  MmaFragmentC accumulator_11 = {};

  for (int group = 0; group < num_groups; ++group) {
    const int scale_column = warp_n * kMmaLaneTileN + quad;
    const int64_t scale_offset =
        (static_cast<int64_t>(tile_n) * num_groups + group) *
        kHmmaBlockN;
    const Scalar scale_0 = packed_scales[scale_offset + scale_column];
    const Scalar scale_1 =
        packed_scales[scale_offset + scale_column + kMmaN];

#pragma unroll
    for (int k_step = 0; k_step < kHmmaBlockK / kMmaK; ++k_step) {
      const int64_t word_offset =
          (((static_cast<int64_t>(tile_n) * num_groups + group) *
                (kHmmaBlockK / kMmaK) +
            k_step) *
               kHmmaWarps +
           warp_n) *
              kMmaLanes +
          lane;
      const uint32_t packed_word =
          static_cast<uint32_t>(packed_lane_qweight[word_offset]);
      MmaFragmentB fragment_b_0;
      MmaFragmentB fragment_b_1;
      MmaLaneDequant<Scalar>::run(
          packed_word,
          scale_0,
          fragment_b_0);
      MmaLaneDequant<Scalar>::run(
          packed_word >> 8,
          scale_1,
          fragment_b_1);

      MmaFragmentA fragment_a_0;
      MmaFragmentA fragment_a_1;
      const int global_k = group * kHmmaBlockK + k_step * kMmaK;
      load_mma_fragment_a_global(
          fragment_a_0,
          input,
          size_k,
          global_m + warp_m,
          global_k,
          lane);
      load_mma_fragment_a_global(
          fragment_a_1,
          input,
          size_k,
          global_m + warp_m + kMmaM,
          global_k,
          lane);
      MmaInstruction<Scalar>::run(
          fragment_a_0,
          fragment_b_0,
          accumulator_00);
      MmaInstruction<Scalar>::run(
          fragment_a_0,
          fragment_b_1,
          accumulator_01);
      MmaInstruction<Scalar>::run(
          fragment_a_1,
          fragment_b_0,
          accumulator_10);
      MmaInstruction<Scalar>::run(
          fragment_a_1,
          fragment_b_1,
          accumulator_11);
    }
  }

  store_mma_fragment(
      accumulator_00,
      output,
      size_n,
      global_m + warp_m,
      global_n,
      lane);
  store_mma_fragment(
      accumulator_01,
      output,
      size_n,
      global_m + warp_m,
      global_n + kMmaN,
      lane);
  store_mma_fragment(
      accumulator_10,
      output,
      size_n,
      global_m + warp_m + kMmaM,
      global_n,
      lane);
  store_mma_fragment(
      accumulator_11,
      output,
      size_n,
      global_m + warp_m + kMmaM,
      global_n + kMmaN,
      lane);
}

template <typename Scalar, bool GuardN>
__device__ __forceinline__ void run_mma_lane_m16_n32_global_a(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups,
    int packed_tile_n,
    int lane_n_tile,
    int global_m,
    int global_n,
    int lane) {
  const int quad = lane >> 2;

  MmaFragmentC accumulator_0 = {};
  MmaFragmentC accumulator_1 = {};
  MmaFragmentC accumulator_2 = {};
  MmaFragmentC accumulator_3 = {};

  for (int group = 0; group < num_groups; ++group) {
    const int64_t scale_offset =
        (static_cast<int64_t>(packed_tile_n) * num_groups + group) *
        kHmmaBlockN;
    const int scale_column = lane_n_tile * kMmaLaneTileN + quad;
    const Scalar scale_0 = packed_scales[scale_offset + scale_column];
    const Scalar scale_1 =
        packed_scales[scale_offset + scale_column + kMmaN];
    const Scalar scale_2 =
        packed_scales[scale_offset + scale_column + kMmaLaneTileN];
    const Scalar scale_3 =
        packed_scales[
            scale_offset + scale_column + kMmaLaneTileN + kMmaN];

#pragma unroll
    for (int k_step = 0; k_step < kHmmaBlockK / kMmaK; ++k_step) {
      MmaFragmentA fragment_a;
      const int global_k = group * kHmmaBlockK + k_step * kMmaK;
      load_mma_fragment_a_global(
          fragment_a,
          input,
          size_k,
          global_m,
          global_k,
          lane);

      {
        const int64_t word_offset =
            (((static_cast<int64_t>(packed_tile_n) * num_groups + group) *
                  (kHmmaBlockK / kMmaK) +
              k_step) *
                 kHmmaWarps +
             lane_n_tile) *
                kMmaLanes +
            lane;
        const uint32_t packed_word =
            static_cast<uint32_t>(packed_lane_qweight[word_offset]);
        MmaFragmentB fragment_b_0;
        MmaFragmentB fragment_b_1;
        MmaLaneDequant<Scalar>::run(
            packed_word,
            scale_0,
            fragment_b_0);
        MmaLaneDequant<Scalar>::run(
            packed_word >> 8,
            scale_1,
            fragment_b_1);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b_0,
            accumulator_0);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b_1,
            accumulator_1);
      }

      {
        const int64_t word_offset =
            (((static_cast<int64_t>(packed_tile_n) * num_groups + group) *
                  (kHmmaBlockK / kMmaK) +
              k_step) *
                 kHmmaWarps +
             lane_n_tile + 1) *
                kMmaLanes +
            lane;
        const uint32_t packed_word =
            static_cast<uint32_t>(packed_lane_qweight[word_offset]);
        MmaFragmentB fragment_b_0;
        MmaFragmentB fragment_b_1;
        MmaLaneDequant<Scalar>::run(
            packed_word,
            scale_2,
            fragment_b_0);
        MmaLaneDequant<Scalar>::run(
            packed_word >> 8,
            scale_3,
            fragment_b_1);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b_0,
            accumulator_2);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b_1,
            accumulator_3);
      }
    }
  }

  if constexpr (!GuardN) {
    store_mma_fragment(
        accumulator_0,
        output,
        size_n,
        global_m,
        global_n,
        lane);
    store_mma_fragment(
        accumulator_1,
        output,
        size_n,
        global_m,
        global_n + kMmaN,
        lane);
    store_mma_fragment(
        accumulator_2,
        output,
        size_n,
        global_m,
        global_n + kMmaLaneTileN,
        lane);
    store_mma_fragment(
        accumulator_3,
        output,
        size_n,
        global_m,
        global_n + kMmaLaneTileN + kMmaN,
        lane);
  } else {
    if (global_n < size_n) {
      store_mma_fragment(
          accumulator_0,
          output,
          size_n,
          global_m,
          global_n,
          lane);
    }
    if (global_n + kMmaN < size_n) {
      store_mma_fragment(
          accumulator_1,
          output,
          size_n,
          global_m,
          global_n + kMmaN,
          lane);
    }
    if (global_n + kMmaLaneTileN < size_n) {
      store_mma_fragment(
          accumulator_2,
          output,
          size_n,
          global_m,
          global_n + kMmaLaneTileN,
          lane);
    }
    if (global_n + kMmaLaneTileN + kMmaN < size_n) {
      store_mma_fragment(
          accumulator_3,
          output,
          size_n,
          global_m,
          global_n + kMmaLaneTileN + kMmaN,
          lane);
    }
  }
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneM32Threads) void amplin_mma_lane_m32_global_a_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_m = (warp >> 1) * kMmaM;
  const int warp_n = (warp & 1) * kMmaLaneM32WarpN;
  const int tile_m = static_cast<int>(blockIdx.y);
  const int tile_n = static_cast<int>(blockIdx.x);
  run_mma_lane_m16_n32_global_a<Scalar, false>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_k,
      size_n,
      num_groups,
      tile_n,
      warp_n / kMmaLaneTileN,
      tile_m * kMmaLaneM32BlockM + warp_m,
      tile_n * kHmmaBlockN + warp_n,
      lane);
}

template <typename Scalar, bool GuardN>
__global__ __launch_bounds__(kMmaLaneM32N32Threads) void amplin_mma_lane_m32_n32_global_a_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile_m = static_cast<int>(blockIdx.y);
  const int tile_n = static_cast<int>(blockIdx.x);
  const int packed_tile_n = tile_n >> 1;
  const int lane_n_tile = (tile_n & 1) * (kMmaLaneM32WarpN / kMmaLaneTileN);
  run_mma_lane_m16_n32_global_a<Scalar, GuardN>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_k,
      size_n,
      num_groups,
      packed_tile_n,
      lane_n_tile,
      tile_m * kMmaLaneM32BlockM + warp * kMmaM,
      tile_n * kMmaLaneM32N32BlockN,
      lane);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLanePaddedThreads) void amplin_mma_lane_m16_n16_padded_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  const int lane = threadIdx.x;
  const int quad = lane >> 2;
  const int tile_n = static_cast<int>(blockIdx.x);
  const int packed_tile_n = tile_n / kHmmaWarps;
  const int lane_n_tile = tile_n - packed_tile_n * kHmmaWarps;
  const int global_n = tile_n * kMmaLanePaddedBlockN;

  MmaFragmentC accumulator_0 = {};
  MmaFragmentC accumulator_1 = {};

  for (int group = 0; group < num_groups; ++group) {
    const int64_t scale_offset =
        (static_cast<int64_t>(packed_tile_n) * num_groups + group) *
        kHmmaBlockN;
    const int scale_column = lane_n_tile * kMmaLaneTileN + quad;
    const Scalar scale_0 = packed_scales[scale_offset + scale_column];
    const Scalar scale_1 = packed_scales[scale_offset + scale_column + kMmaN];

#pragma unroll
    for (int k_step = 0; k_step < kHmmaBlockK / kMmaK; ++k_step) {
      MmaFragmentA fragment_a;
      const int global_k = group * kHmmaBlockK + k_step * kMmaK;
      load_mma_fragment_a_global_guard_m(
          fragment_a,
          input,
          size_m,
          size_k,
          global_k,
          lane);

      const int64_t word_offset =
          (((static_cast<int64_t>(packed_tile_n) * num_groups + group) *
                (kHmmaBlockK / kMmaK) +
            k_step) *
               kHmmaWarps +
           lane_n_tile) *
              kMmaLanes +
          lane;
      const uint32_t packed_word =
          static_cast<uint32_t>(packed_lane_qweight[word_offset]);
      MmaFragmentB fragment_b_0;
      MmaFragmentB fragment_b_1;
      MmaLaneDequant<Scalar>::run(
          packed_word,
          scale_0,
          fragment_b_0);
      MmaLaneDequant<Scalar>::run(
          packed_word >> 8,
          scale_1,
          fragment_b_1);
      MmaInstruction<Scalar>::run(
          fragment_a,
          fragment_b_0,
          accumulator_0);
      MmaInstruction<Scalar>::run(
          fragment_a,
          fragment_b_1,
          accumulator_1);
    }
  }

  store_mma_fragment_guard_m(
      accumulator_0,
      output,
      size_m,
      size_n,
      global_n,
      lane);
  store_mma_fragment_guard_m(
      accumulator_1,
      output,
      size_m,
      size_n,
      global_n + kMmaN,
      lane);
}

template <typename Scalar, int SplitKWarps>
__device__ __forceinline__ void amplin_mma_lane_m16_n16_splitk_body(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    float* __restrict__ partials) {
  const int lane = threadIdx.x & (kMmaLanes - 1);
  const int warp = threadIdx.x / kMmaLanes;
  const int quad = lane >> 2;
  const int tile_n = static_cast<int>(blockIdx.x);
  const int packed_tile_n = tile_n / kHmmaWarps;
  const int lane_n_tile = tile_n - packed_tile_n * kHmmaWarps;
  const int global_n = tile_n * kMmaLanePaddedBlockN;

  MmaFragmentC accumulator_0 = {};
  MmaFragmentC accumulator_1 = {};

  for (int group = warp; group < num_groups; group += SplitKWarps) {
    const int64_t scale_offset =
        (static_cast<int64_t>(packed_tile_n) * num_groups + group) *
        kHmmaBlockN;
    const int scale_column = lane_n_tile * kMmaLaneTileN + quad;
    const Scalar scale_0 = packed_scales[scale_offset + scale_column];
    const Scalar scale_1 = packed_scales[scale_offset + scale_column + kMmaN];

#pragma unroll
    for (int k_step = 0; k_step < kHmmaBlockK / kMmaK; ++k_step) {
      MmaFragmentA fragment_a;
      const int global_k = group * kHmmaBlockK + k_step * kMmaK;
      load_mma_fragment_a_global_guard_m(
          fragment_a,
          input,
          size_m,
          size_k,
          global_k,
          lane);

      const int64_t word_offset =
          (((static_cast<int64_t>(packed_tile_n) * num_groups + group) *
                (kHmmaBlockK / kMmaK) +
            k_step) *
               kHmmaWarps +
           lane_n_tile) *
              kMmaLanes +
          lane;
      const uint32_t packed_word =
          static_cast<uint32_t>(packed_lane_qweight[word_offset]);
      MmaFragmentB fragment_b_0;
      MmaFragmentB fragment_b_1;
      MmaLaneDequant<Scalar>::run(
          packed_word,
          scale_0,
          fragment_b_0);
      MmaLaneDequant<Scalar>::run(
          packed_word >> 8,
          scale_1,
          fragment_b_1);
      MmaInstruction<Scalar>::run(
          fragment_a,
          fragment_b_0,
          accumulator_0);
      MmaInstruction<Scalar>::run(
          fragment_a,
          fragment_b_1,
          accumulator_1);
    }
  }

  const int warp_partial_base =
      warp *
      kMmaLaneSplitKFragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues;
  const int lane_partial_base = lane * kMmaLaneAccumulatorValues;
#pragma unroll
  for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
    partials[warp_partial_base + lane_partial_base + value] =
        accumulator_0.values[value];
    partials[
        warp_partial_base +
        kMmaLanes * kMmaLaneAccumulatorValues +
        lane_partial_base +
        value] = accumulator_1.values[value];
  }
  __syncthreads();

  if (warp == 0) {
    MmaFragmentC reduced_0 = {};
    MmaFragmentC reduced_1 = {};
#pragma unroll
    for (int partial_warp = 0; partial_warp < SplitKWarps; ++partial_warp) {
      const int partial_base =
          partial_warp *
          kMmaLaneSplitKFragments *
          kMmaLanes *
          kMmaLaneAccumulatorValues;
#pragma unroll
      for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
        reduced_0.values[value] +=
            partials[partial_base + lane_partial_base + value];
        reduced_1.values[value] +=
            partials[
                partial_base +
                kMmaLanes * kMmaLaneAccumulatorValues +
                lane_partial_base +
                value];
      }
    }
    store_mma_fragment_guard_m(
        reduced_0,
        output,
        size_m,
        size_n,
        global_n,
        lane);
    store_mma_fragment_guard_m(
        reduced_1,
        output,
        size_m,
        size_n,
        global_n + kMmaN,
        lane);
  }
}

template <
    typename Scalar,
    int SplitKWarps,
    bool PipelineKSteps = false,
    bool InterleavedN32Words = false>
__device__ __forceinline__ void amplin_mma_lane_m16_n32_splitk12_body(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    float* __restrict__ partials) {
  const int lane = threadIdx.x & (kMmaLanes - 1);
  const int warp = threadIdx.x / kMmaLanes;
  const int quad = lane >> 2;
  const int tile_n = static_cast<int>(blockIdx.x) * 2;
  const int packed_tile_n = tile_n / kHmmaWarps;
  const int lane_n_tile = tile_n - packed_tile_n * kHmmaWarps;
  const int global_n = tile_n * kMmaLanePaddedBlockN;

  MmaFragmentC accumulator_0 = {};
  MmaFragmentC accumulator_1 = {};
  MmaFragmentC accumulator_2 = {};
  MmaFragmentC accumulator_3 = {};

  for (int group = warp; group < num_groups; group += SplitKWarps) {
    const int64_t scale_offset =
        (static_cast<int64_t>(packed_tile_n) * num_groups + group) *
        kHmmaBlockN;
    const int scale_column_0 = lane_n_tile * kMmaLaneTileN + quad;
    const int scale_column_1 = scale_column_0 + kMmaLaneTileN;
    const Scalar scale_0 = packed_scales[scale_offset + scale_column_0];
    const Scalar scale_1 = packed_scales[scale_offset + scale_column_0 + kMmaN];
    const Scalar scale_2 = packed_scales[scale_offset + scale_column_1];
    const Scalar scale_3 = packed_scales[scale_offset + scale_column_1 + kMmaN];

    if constexpr (PipelineKSteps) {
      constexpr int kSteps = kHmmaBlockK / kMmaK;
      int64_t group_word_base;
      if constexpr (InterleavedN32Words) {
        group_word_base =
            (((static_cast<int64_t>(blockIdx.x) * num_groups + group) *
                  kSteps *
                  kMmaLanes +
              lane) *
             2);
      } else {
        group_word_base =
            (((static_cast<int64_t>(packed_tile_n) * num_groups + group) *
                  kSteps *
                  kHmmaWarps +
              lane_n_tile) *
                 kMmaLanes) +
            lane;
      }
      MmaFragmentA fragment_a;
      load_mma_fragment_a_global_guard_m(
          fragment_a,
          input,
          size_m,
          size_k,
          group * kHmmaBlockK,
          lane);
      uint32_t packed_word_0;
      uint32_t packed_word_1;
      if constexpr (InterleavedN32Words) {
        const uint2 packed_pair =
            *reinterpret_cast<const uint2*>(
                packed_lane_qweight + group_word_base);
        packed_word_0 = packed_pair.x;
        packed_word_1 = packed_pair.y;
      } else {
        packed_word_0 =
            static_cast<uint32_t>(packed_lane_qweight[group_word_base]);
        packed_word_1 =
            static_cast<uint32_t>(
                packed_lane_qweight[group_word_base + kMmaLanes]);
      }

#pragma unroll
      for (int k_step = 0; k_step < kSteps; ++k_step) {
        MmaFragmentA next_fragment_a;
        uint32_t next_packed_word_0 = 0;
        uint32_t next_packed_word_1 = 0;
        if (k_step + 1 < kSteps) {
          const int next_k_step = k_step + 1;
          load_mma_fragment_a_global_guard_m(
              next_fragment_a,
              input,
              size_m,
              size_k,
              group * kHmmaBlockK + next_k_step * kMmaK,
              lane);
          const int64_t next_word_offset =
              group_word_base +
              static_cast<int64_t>(next_k_step) *
                  (InterleavedN32Words ? 2 : kHmmaWarps) *
                  kMmaLanes;
          if constexpr (InterleavedN32Words) {
            const uint2 next_packed_pair =
                *reinterpret_cast<const uint2*>(
                    packed_lane_qweight + next_word_offset);
            next_packed_word_0 = next_packed_pair.x;
            next_packed_word_1 = next_packed_pair.y;
          } else {
            next_packed_word_0 =
                static_cast<uint32_t>(
                    packed_lane_qweight[next_word_offset]);
            next_packed_word_1 =
                static_cast<uint32_t>(
                    packed_lane_qweight[next_word_offset + kMmaLanes]);
          }
        }

        MmaFragmentB fragment_b;
        MmaLaneDequant<Scalar>::run(
            packed_word_0,
            scale_0,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_0);
        MmaLaneDequant<Scalar>::run(
            packed_word_0 >> 8,
            scale_1,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_1);
        MmaLaneDequant<Scalar>::run(
            packed_word_1,
            scale_2,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_2);
        MmaLaneDequant<Scalar>::run(
            packed_word_1 >> 8,
            scale_3,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_3);

        if (k_step + 1 < kSteps) {
          fragment_a = next_fragment_a;
          packed_word_0 = next_packed_word_0;
          packed_word_1 = next_packed_word_1;
        }
      }
    } else {
#pragma unroll
      for (int k_step = 0; k_step < kHmmaBlockK / kMmaK; ++k_step) {
        MmaFragmentA fragment_a;
        const int global_k = group * kHmmaBlockK + k_step * kMmaK;
        load_mma_fragment_a_global_guard_m(
            fragment_a,
            input,
            size_m,
            size_k,
            global_k,
            lane);

        const int64_t word_offset =
            (((static_cast<int64_t>(packed_tile_n) * num_groups + group) *
                  (kHmmaBlockK / kMmaK) +
              k_step) *
                 kHmmaWarps +
             lane_n_tile) *
                kMmaLanes +
            lane;
        const uint32_t packed_word_0 =
            static_cast<uint32_t>(packed_lane_qweight[word_offset]);
        const uint32_t packed_word_1 =
            static_cast<uint32_t>(
                packed_lane_qweight[word_offset + kMmaLanes]);
        MmaFragmentB fragment_b;
        MmaLaneDequant<Scalar>::run(
            packed_word_0,
            scale_0,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_0);
        MmaLaneDequant<Scalar>::run(
            packed_word_0 >> 8,
            scale_1,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_1);
        MmaLaneDequant<Scalar>::run(
            packed_word_1,
            scale_2,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_2);
        MmaLaneDequant<Scalar>::run(
            packed_word_1 >> 8,
            scale_3,
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulator_3);
      }
    }
  }

  const int warp_partial_base =
      warp *
      kMmaLaneSplitKN32Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues;
  const int lane_partial_base = lane * kMmaLaneAccumulatorValues;
#pragma unroll
  for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
    partials[warp_partial_base + lane_partial_base + value] =
        accumulator_0.values[value];
    partials[
        warp_partial_base +
        kMmaLanes * kMmaLaneAccumulatorValues +
        lane_partial_base +
        value] = accumulator_1.values[value];
    partials[
        warp_partial_base +
        2 * kMmaLanes * kMmaLaneAccumulatorValues +
        lane_partial_base +
        value] = accumulator_2.values[value];
    partials[
        warp_partial_base +
        3 * kMmaLanes * kMmaLaneAccumulatorValues +
        lane_partial_base +
        value] = accumulator_3.values[value];
  }
  __syncthreads();

  if (warp == 0) {
    MmaFragmentC reduced_0 = {};
    MmaFragmentC reduced_1 = {};
    MmaFragmentC reduced_2 = {};
    MmaFragmentC reduced_3 = {};
#pragma unroll
    for (int partial_warp = 0; partial_warp < SplitKWarps; ++partial_warp) {
      const int partial_base =
          partial_warp *
          kMmaLaneSplitKN32Fragments *
          kMmaLanes *
          kMmaLaneAccumulatorValues;
#pragma unroll
      for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
        reduced_0.values[value] +=
            partials[partial_base + lane_partial_base + value];
        reduced_1.values[value] +=
            partials[
                partial_base +
                kMmaLanes * kMmaLaneAccumulatorValues +
                lane_partial_base +
                value];
        reduced_2.values[value] +=
            partials[
                partial_base +
                2 * kMmaLanes * kMmaLaneAccumulatorValues +
                lane_partial_base +
                value];
        reduced_3.values[value] +=
            partials[
                partial_base +
                3 * kMmaLanes * kMmaLaneAccumulatorValues +
                lane_partial_base +
                value];
      }
    }
    store_mma_fragment_guard_m(
        reduced_0,
        output,
        size_m,
        size_n,
        global_n,
        lane);
    store_mma_fragment_guard_m(
        reduced_1,
        output,
        size_m,
        size_n,
        global_n + kMmaN,
        lane);
    store_mma_fragment_guard_m(
        reduced_2,
        output,
        size_m,
        size_n,
        global_n + 2 * kMmaN,
        lane);
    store_mma_fragment_guard_m(
        reduced_3,
        output,
        size_m,
        size_n,
        global_n + 3 * kMmaN,
        lane);
  }
}

template <typename Scalar>
__device__ __forceinline__ void amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_body(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    float* __restrict__ partials) {
  const int lane = threadIdx.x & (kMmaLanes - 1);
  const int warp = threadIdx.x / kMmaLanes;
  const int quad = lane >> 2;
  const int tile_n = static_cast<int>(blockIdx.x);
  const int global_n = tile_n * kHmmaBlockN;

  MmaFragmentC accumulators[kMmaLaneSplitKN64Fragments] = {};

  for (int group = warp; group < num_groups; group += kMmaLaneSplitK24Warps) {
    const int64_t scale_offset =
        (static_cast<int64_t>(tile_n) * num_groups + group) *
        kHmmaBlockN;
    Scalar scales[kMmaLaneSplitKN64Fragments];
#pragma unroll
    for (int fragment = 0; fragment < kMmaLaneSplitKN64Fragments; ++fragment) {
      scales[fragment] =
          packed_scales[scale_offset + fragment * kMmaN + quad];
    }

    constexpr int kSteps = kHmmaBlockK / kMmaK;
    const int64_t group_word_base =
        (((static_cast<int64_t>(tile_n) * num_groups + group) *
              kSteps *
              kMmaLanes +
          lane) *
         kHmmaWarps);
    MmaFragmentA fragment_a;
    load_mma_fragment_a_global_guard_m(
        fragment_a,
        input,
        size_m,
        size_k,
        group * kHmmaBlockK,
        lane);
    uint4 packed_words =
        *reinterpret_cast<const uint4*>(
            packed_lane_qweight + group_word_base);

#pragma unroll
    for (int k_step = 0; k_step < kSteps; ++k_step) {
      MmaFragmentA next_fragment_a;
      uint4 next_packed_words = {};
      if (k_step + 1 < kSteps) {
        const int next_k_step = k_step + 1;
        load_mma_fragment_a_global_guard_m(
            next_fragment_a,
            input,
            size_m,
            size_k,
            group * kHmmaBlockK + next_k_step * kMmaK,
            lane);
        const int64_t next_word_offset =
            group_word_base +
            static_cast<int64_t>(next_k_step) *
                kMmaLanes *
                kHmmaWarps;
        next_packed_words =
            *reinterpret_cast<const uint4*>(
                packed_lane_qweight + next_word_offset);
      }

      const uint32_t words[kHmmaWarps] = {
          packed_words.x,
          packed_words.y,
          packed_words.z,
          packed_words.w,
      };
      MmaFragmentB fragment_b;
#pragma unroll
      for (int word = 0; word < kHmmaWarps; ++word) {
        MmaLaneDequant<Scalar>::run(
            words[word],
            scales[word * 2],
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulators[word * 2]);
        MmaLaneDequant<Scalar>::run(
            words[word] >> 8,
            scales[word * 2 + 1],
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulators[word * 2 + 1]);
      }

      if (k_step + 1 < kSteps) {
        fragment_a = next_fragment_a;
        packed_words = next_packed_words;
      }
    }
  }

  const int warp_partial_base =
      warp *
      kMmaLaneSplitKN64Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues;
  const int lane_partial_base = lane * kMmaLaneAccumulatorValues;
#pragma unroll
  for (int fragment = 0; fragment < kMmaLaneSplitKN64Fragments; ++fragment) {
#pragma unroll
    for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
      partials[
          warp_partial_base +
          fragment * kMmaLanes * kMmaLaneAccumulatorValues +
          lane_partial_base +
          value] = accumulators[fragment].values[value];
    }
  }
  __syncthreads();

  if (warp < kMmaLaneSplitKN64Fragments) {
    MmaFragmentC reduced = {};
#pragma unroll
    for (int partial_warp = 0; partial_warp < kMmaLaneSplitK24Warps; ++partial_warp) {
      const int partial_base =
          partial_warp *
              kMmaLaneSplitKN64Fragments *
              kMmaLanes *
              kMmaLaneAccumulatorValues +
          warp * kMmaLanes * kMmaLaneAccumulatorValues;
#pragma unroll
      for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
        reduced.values[value] +=
            partials[partial_base + lane_partial_base + value];
      }
    }
    store_mma_fragment_guard_m(
        reduced,
        output,
        size_m,
        size_n,
        global_n + warp * kMmaN,
        lane);
  }
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK12Threads)
void amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    float* __restrict__ scratch,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK12Warps *
      kMmaLaneSplitKN64Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];

  const int lane = threadIdx.x & (kMmaLanes - 1);
  const int warp = threadIdx.x / kMmaLanes;
  const int quad = lane >> 2;
  const int split = static_cast<int>(blockIdx.y);
  const int tile_n = static_cast<int>(blockIdx.x);
  const int global_n = tile_n * kHmmaBlockN;
  const int global_k_warp = split * kMmaLaneSplitK12Warps + warp;

  MmaFragmentC accumulators[kMmaLaneSplitKN64Fragments] = {};

  for (int group = global_k_warp; group < num_groups; group += kMmaLaneSplitK24Warps) {
    const int64_t scale_offset =
        (static_cast<int64_t>(tile_n) * num_groups + group) *
        kHmmaBlockN;
    Scalar scales[kMmaLaneSplitKN64Fragments];
#pragma unroll
    for (int fragment = 0; fragment < kMmaLaneSplitKN64Fragments; ++fragment) {
      scales[fragment] =
          packed_scales[scale_offset + fragment * kMmaN + quad];
    }

    constexpr int kSteps = kHmmaBlockK / kMmaK;
    const int64_t group_word_base =
        (((static_cast<int64_t>(tile_n) * num_groups + group) *
              kSteps *
              kMmaLanes +
          lane) *
         kHmmaWarps);
    MmaFragmentA fragment_a;
    load_mma_fragment_a_global_guard_m(
        fragment_a,
        input,
        size_m,
        size_k,
        group * kHmmaBlockK,
        lane);
    uint4 packed_words =
        *reinterpret_cast<const uint4*>(
            packed_lane_qweight + group_word_base);

#pragma unroll
    for (int k_step = 0; k_step < kSteps; ++k_step) {
      MmaFragmentA next_fragment_a;
      uint4 next_packed_words = {};
      if (k_step + 1 < kSteps) {
        const int next_k_step = k_step + 1;
        load_mma_fragment_a_global_guard_m(
            next_fragment_a,
            input,
            size_m,
            size_k,
            group * kHmmaBlockK + next_k_step * kMmaK,
            lane);
        const int64_t next_word_offset =
            group_word_base +
            static_cast<int64_t>(next_k_step) *
                kMmaLanes *
                kHmmaWarps;
        next_packed_words =
            *reinterpret_cast<const uint4*>(
                packed_lane_qweight + next_word_offset);
      }

      const uint32_t words[kHmmaWarps] = {
          packed_words.x,
          packed_words.y,
          packed_words.z,
          packed_words.w,
      };
      MmaFragmentB fragment_b;
#pragma unroll
      for (int word = 0; word < kHmmaWarps; ++word) {
        MmaLaneDequant<Scalar>::run(
            words[word],
            scales[word * 2],
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulators[word * 2]);
        MmaLaneDequant<Scalar>::run(
            words[word] >> 8,
            scales[word * 2 + 1],
            fragment_b);
        MmaInstruction<Scalar>::run(
            fragment_a,
            fragment_b,
            accumulators[word * 2 + 1]);
      }

      if (k_step + 1 < kSteps) {
        fragment_a = next_fragment_a;
        packed_words = next_packed_words;
      }
    }
  }

  const int warp_partial_base =
      warp *
      kMmaLaneSplitKN64Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues;
  const int lane_partial_base = lane * kMmaLaneAccumulatorValues;
#pragma unroll
  for (int fragment = 0; fragment < kMmaLaneSplitKN64Fragments; ++fragment) {
#pragma unroll
    for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
      partials[
          warp_partial_base +
          fragment * kMmaLanes * kMmaLaneAccumulatorValues +
          lane_partial_base +
          value] = accumulators[fragment].values[value];
    }
  }
  __syncthreads();

  if (warp < kMmaLaneSplitKN64Fragments) {
    MmaFragmentC reduced = {};
#pragma unroll
    for (int partial_warp = 0; partial_warp < kMmaLaneSplitK12Warps; ++partial_warp) {
      const int partial_base =
          partial_warp *
              kMmaLaneSplitKN64Fragments *
              kMmaLanes *
              kMmaLaneAccumulatorValues +
          warp * kMmaLanes * kMmaLaneAccumulatorValues;
#pragma unroll
      for (int value = 0; value < kMmaLaneAccumulatorValues; ++value) {
        reduced.values[value] +=
            partials[partial_base + lane_partial_base + value];
      }
    }

    const int thread_in_quad = lane & 3;
    const int column = global_n + warp * kMmaN + thread_in_quad * 2;
    const int64_t plane_stride = static_cast<int64_t>(size_m) * size_n;
    float* split_scratch = scratch + static_cast<int64_t>(split) * plane_stride;
    if (quad < size_m) {
      const int64_t index = static_cast<int64_t>(quad) * size_n + column;
      split_scratch[index] = reduced.values[0];
      split_scratch[index + 1] = reduced.values[1];
    }
    if (quad + 8 < size_m) {
      const int64_t index = static_cast<int64_t>(quad + 8) * size_n + column;
      split_scratch[index] = reduced.values[2];
      split_scratch[index + 1] = reduced.values[3];
    }
  }

  cooperative_groups::this_grid().sync();

  if (split == 0 && warp < kMmaLaneSplitKN64Fragments) {
    const int thread_in_quad = lane & 3;
    const int column = global_n + warp * kMmaN + thread_in_quad * 2;
    const int64_t plane_stride = static_cast<int64_t>(size_m) * size_n;
    if (quad < size_m) {
      const int64_t index = static_cast<int64_t>(quad) * size_n + column;
      output[index] = ScalarTraits<Scalar>::from_float(
          scratch[index] + scratch[plane_stride + index]);
      output[index + 1] = ScalarTraits<Scalar>::from_float(
          scratch[index + 1] + scratch[plane_stride + index + 1]);
    }
    if (quad + 8 < size_m) {
      const int64_t index = static_cast<int64_t>(quad + 8) * size_n + column;
      output[index] = ScalarTraits<Scalar>::from_float(
          scratch[index] + scratch[plane_stride + index]);
      output[index + 1] = ScalarTraits<Scalar>::from_float(
          scratch[index + 1] + scratch[plane_stride + index + 1]);
    }
  }
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK4Threads) void amplin_mma_lane_m16_n16_splitk4_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK4Warps *
      kMmaLaneSplitKFragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n16_splitk_body<Scalar, kMmaLaneSplitK4Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK8Threads) void amplin_mma_lane_m16_n16_splitk8_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK8Warps *
      kMmaLaneSplitKFragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n16_splitk_body<Scalar, kMmaLaneSplitK8Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK12Threads) void amplin_mma_lane_m16_n16_splitk12_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK12Warps *
      kMmaLaneSplitKFragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n16_splitk_body<Scalar, kMmaLaneSplitK12Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK12Threads) void amplin_mma_lane_m16_n32_splitk12_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK12Warps *
      kMmaLaneSplitKN32Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n32_splitk12_body<Scalar, kMmaLaneSplitK12Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK16Threads) void amplin_mma_lane_m16_n32_splitk16_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK16Warps *
      kMmaLaneSplitKN32Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n32_splitk12_body<Scalar, kMmaLaneSplitK16Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK12Threads) void amplin_mma_lane_m16_n32_splitk12_pipe2_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK12Warps *
      kMmaLaneSplitKN32Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n32_splitk12_body<
      Scalar,
      kMmaLaneSplitK12Warps,
      true>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK12Threads)
void amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK12Warps *
      kMmaLaneSplitKN32Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n32_splitk12_body<
      Scalar,
      kMmaLaneSplitK12Warps,
      true,
      true>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK24Threads)
void amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  extern __shared__ float partials[];
  amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_body(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK16Threads) void amplin_mma_lane_m16_n32_splitk16_pipe2_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK16Warps *
      kMmaLaneSplitKN32Fragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n32_splitk12_body<
      Scalar,
      kMmaLaneSplitK16Warps,
      true>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

template <typename Scalar>
__global__ __launch_bounds__(kMmaLaneSplitK16Threads) void amplin_mma_lane_m16_n16_splitk16_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_lane_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ float partials[
      kMmaLaneSplitK16Warps *
      kMmaLaneSplitKFragments *
      kMmaLanes *
      kMmaLaneAccumulatorValues];
  amplin_mma_lane_m16_n16_splitk_body<Scalar, kMmaLaneSplitK16Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      output,
      size_m,
      size_k,
      size_n,
      num_groups,
      partials);
}

__device__ __forceinline__ void cp_async_16(void* shared_destination, const void* global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(shared_address), "l"(global_source)
      : "memory");
#else
  const uint4 value = *reinterpret_cast<const uint4*>(global_source);
  *reinterpret_cast<uint4*>(shared_destination) = value;
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

template <typename Scalar>
__device__ __forceinline__ void copy_16_sync(
    Scalar* shared_destination,
    const Scalar* global_source) {
  const uint4 value = *reinterpret_cast<const uint4*>(global_source);
  *reinterpret_cast<uint4*>(shared_destination) = value;
}

template <typename Scalar>
__global__ __launch_bounds__(kV0Threads) void amplin_gptq_w4_group128_gemv_v0_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ qweight,
    const Scalar* __restrict__ scales,
    Scalar* __restrict__ output,
    int size_n) {
  __shared__ float paired_partials[kV0Warps * kSharedStride];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int column = lane & (kBlockN - 1);
  const int group = warp * 2 + (lane >> 4);
  const int global_n = blockIdx.x * kBlockN + column;

  float even_accumulator = 0.0f;
  float odd_accumulator = 0.0f;

#pragma unroll
  for (int packed_offset = 0; packed_offset < kPackedRowsPerGroup; ++packed_offset) {
    const int packed_k = group * kPackedRowsPerGroup + packed_offset;
    const uint32_t word = static_cast<uint32_t>(
        qweight[static_cast<int64_t>(packed_k) * size_n + global_n]);
    const int activation_k = group * kGroupSize + packed_offset * kPackFactor;

#pragma unroll
    for (int code_offset = 0; code_offset < kPackFactor; ++code_offset) {
      const int code = static_cast<int>((word >> (code_offset * 4)) & 0x0fu);
      const float activation = ScalarTraits<Scalar>::to_float(input[activation_k + code_offset]);
      const float centered_code = static_cast<float>(code - 8);
      if ((code_offset & 1) == 0) {
        even_accumulator = __fmaf_rn(activation, centered_code, even_accumulator);
      } else {
        odd_accumulator = __fmaf_rn(activation, centered_code, odd_accumulator);
      }
    }
  }

  const float scale = ScalarTraits<Scalar>::to_float(
      scales[static_cast<int64_t>(group) * size_n + global_n]);
  const float group_partial = (even_accumulator + odd_accumulator) * scale;
  const float adjacent_group = __shfl_down_sync(0xffffffffu, group_partial, kBlockN);

  if (lane < kBlockN) {
    paired_partials[warp * kSharedStride + column] = group_partial + adjacent_group;
  }
  __syncthreads();

  if (warp == 0 && lane < kBlockN) {
    float value = 0.0f;
#pragma unroll
    for (int paired_group = 0; paired_group < kV0Warps; ++paired_group) {
      value += paired_partials[paired_group * kSharedStride + column];
    }
    output[global_n] = ScalarTraits<Scalar>::from_float(value);
  }
}

template <typename Scalar>
__global__ __launch_bounds__(kGeneralMaxThreads) void amplin_gptq_w4_group128_group_stride_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ qweight,
    const Scalar* __restrict__ scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups,
    int blocks_n) {
  __shared__ float warp_partials[kGeneralMaxWarps * kSharedStride];

  const int activation_row = static_cast<int>(blockIdx.x) / blocks_n;
  const int column_block = static_cast<int>(blockIdx.x) - activation_row * blocks_n;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;
  const int column = lane & (kBlockN - 1);
  const int half_warp = lane >> 4;
  const int global_n = column_block * kBlockN + column;
  const int group_pairs = (num_groups + 1) / 2;
  const Scalar* row_input = input + static_cast<int64_t>(activation_row) * size_k;

  float warp_partial = 0.0f;
  for (int group_pair = warp; group_pair < group_pairs; group_pair += num_warps) {
    const int group = group_pair * 2 + half_warp;
    float group_partial = 0.0f;
    if (group < num_groups && global_n < size_n) {
      float even_accumulator = 0.0f;
      float odd_accumulator = 0.0f;

#pragma unroll
      for (int packed_offset = 0; packed_offset < kPackedRowsPerGroup; ++packed_offset) {
        const int packed_k = group * kPackedRowsPerGroup + packed_offset;
        const uint32_t word = static_cast<uint32_t>(
            qweight[static_cast<int64_t>(packed_k) * size_n + global_n]);
        const int activation_k = group * kGroupSize + packed_offset * kPackFactor;

#pragma unroll
        for (int code_offset = 0; code_offset < kPackFactor; ++code_offset) {
          const int code = static_cast<int>((word >> (code_offset * 4)) & 0x0fu);
          const float activation = ScalarTraits<Scalar>::to_float(row_input[activation_k + code_offset]);
          const float centered_code = static_cast<float>(code - 8);
          if ((code_offset & 1) == 0) {
            even_accumulator = __fmaf_rn(activation, centered_code, even_accumulator);
          } else {
            odd_accumulator = __fmaf_rn(activation, centered_code, odd_accumulator);
          }
        }
      }

      const float scale = ScalarTraits<Scalar>::to_float(
          scales[static_cast<int64_t>(group) * size_n + global_n]);
      group_partial = (even_accumulator + odd_accumulator) * scale;
    }

    const float adjacent_group = __shfl_down_sync(0xffffffffu, group_partial, kBlockN);
    if (lane < kBlockN) {
      warp_partial += group_partial + adjacent_group;
    }
  }

  if (lane < kBlockN) {
    warp_partials[warp * kSharedStride + column] = warp_partial;
  }
  __syncthreads();

  if (warp == 0 && lane < kBlockN && global_n < size_n) {
    float value = 0.0f;
#pragma unroll
    for (int partial_warp = 0; partial_warp < kGeneralMaxWarps; ++partial_warp) {
      if (partial_warp < num_warps) {
        value += warp_partials[partial_warp * kSharedStride + column];
      }
    }
    output[static_cast<int64_t>(activation_row) * size_n + global_n] =
        ScalarTraits<Scalar>::from_float(value);
  }
}

template <typename Scalar>
__global__ __launch_bounds__(kWideThreads) void amplin_gptq_w4_group128_gemv_k12288_wide_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ qweight,
    const Scalar* __restrict__ scales,
    Scalar* __restrict__ output,
    int size_n) {
  __shared__ float warp_partials[kWideWarps * kSharedStride];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int column = lane & (kBlockN - 1);
  const int half_warp = lane >> 4;
  const int global_n = static_cast<int>(blockIdx.x) * kBlockN + column;
  constexpr int group_pairs = kWideGroups / 2;

  float warp_partial = 0.0f;
#pragma unroll
  for (int group_pair = warp; group_pair < group_pairs; group_pair += kWideWarps) {
    const int group = group_pair * 2 + half_warp;
    float even_accumulator = 0.0f;
    float odd_accumulator = 0.0f;

#pragma unroll
    for (int packed_offset = 0; packed_offset < kPackedRowsPerGroup; ++packed_offset) {
      const int packed_k = group * kPackedRowsPerGroup + packed_offset;
      const uint32_t word = static_cast<uint32_t>(
          qweight[static_cast<int64_t>(packed_k) * size_n + global_n]);
      const int activation_k = group * kGroupSize + packed_offset * kPackFactor;

#pragma unroll
      for (int code_offset = 0; code_offset < kPackFactor; ++code_offset) {
        const int code = static_cast<int>((word >> (code_offset * 4)) & 0x0fu);
        const float activation = ScalarTraits<Scalar>::to_float(input[activation_k + code_offset]);
        const float centered_code = static_cast<float>(code - 8);
        if ((code_offset & 1) == 0) {
          even_accumulator = __fmaf_rn(activation, centered_code, even_accumulator);
        } else {
          odd_accumulator = __fmaf_rn(activation, centered_code, odd_accumulator);
        }
      }
    }

    const float scale = ScalarTraits<Scalar>::to_float(
        scales[static_cast<int64_t>(group) * size_n + global_n]);
    const float group_partial = (even_accumulator + odd_accumulator) * scale;
    const float adjacent_group = __shfl_down_sync(0xffffffffu, group_partial, kBlockN);
    if (lane < kBlockN) {
      warp_partial += group_partial + adjacent_group;
    }
  }

  if (lane < kBlockN) {
    warp_partials[warp * kSharedStride + column] = warp_partial;
  }
  __syncthreads();

  if (warp == 0 && lane < kBlockN) {
    float value = 0.0f;
#pragma unroll
    for (int partial_warp = 0; partial_warp < kWideWarps; ++partial_warp) {
      value += warp_partials[partial_warp * kSharedStride + column];
    }
    output[global_n] = ScalarTraits<Scalar>::from_float(value);
  }
}

template <typename Scalar, int RowsPerCta>
__global__ __launch_bounds__(kMultiRowThreads) void amplin_gptq_w4_group128_gemv_multirow_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ qweight,
    const Scalar* __restrict__ scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups,
    int blocks_n) {
  static_assert(RowsPerCta == 2 || RowsPerCta == 4);
  __shared__ float warp_partials[RowsPerCta * kMultiRowWarps * kSharedStride];

  const int row_tile = static_cast<int>(blockIdx.x) / blocks_n;
  const int column_block = static_cast<int>(blockIdx.x) - row_tile * blocks_n;
  const int row_base = row_tile * RowsPerCta;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int column = lane & (kBlockN - 1);
  const int half_warp = lane >> 4;
  const int global_n = column_block * kBlockN + column;
  const int group_pairs = num_groups / 2;
  const Scalar* row_inputs[RowsPerCta];
#pragma unroll
  for (int row = 0; row < RowsPerCta; ++row) {
    row_inputs[row] = input + static_cast<int64_t>(row_base + row) * size_k;
  }

  float row_warp_partials[RowsPerCta] = {};
  for (int group_pair = warp; group_pair < group_pairs; group_pair += kMultiRowWarps) {
    const int group = group_pair * 2 + half_warp;
    float even_accumulators[RowsPerCta] = {};
    float odd_accumulators[RowsPerCta] = {};

#pragma unroll
    for (int packed_offset = 0; packed_offset < kPackedRowsPerGroup; ++packed_offset) {
      const int packed_k = group * kPackedRowsPerGroup + packed_offset;
      const uint32_t word = static_cast<uint32_t>(
          qweight[static_cast<int64_t>(packed_k) * size_n + global_n]);
      const int activation_k = group * kGroupSize + packed_offset * kPackFactor;

#pragma unroll
      for (int code_offset = 0; code_offset < kPackFactor; ++code_offset) {
        const int code = static_cast<int>((word >> (code_offset * 4)) & 0x0fu);
        const float centered_code = static_cast<float>(code - 8);
#pragma unroll
        for (int row = 0; row < RowsPerCta; ++row) {
          const float activation =
              ScalarTraits<Scalar>::to_float(row_inputs[row][activation_k + code_offset]);
          if ((code_offset & 1) == 0) {
            even_accumulators[row] =
                __fmaf_rn(activation, centered_code, even_accumulators[row]);
          } else {
            odd_accumulators[row] =
                __fmaf_rn(activation, centered_code, odd_accumulators[row]);
          }
        }
      }
    }

    const float scale = ScalarTraits<Scalar>::to_float(
        scales[static_cast<int64_t>(group) * size_n + global_n]);
#pragma unroll
    for (int row = 0; row < RowsPerCta; ++row) {
      const float group_partial =
          (even_accumulators[row] + odd_accumulators[row]) * scale;
      const float adjacent_group =
          __shfl_down_sync(0xffffffffu, group_partial, kBlockN);
      if (lane < kBlockN) {
        row_warp_partials[row] += group_partial + adjacent_group;
      }
    }
  }

  if (lane < kBlockN) {
#pragma unroll
    for (int row = 0; row < RowsPerCta; ++row) {
      warp_partials[
          (row * kMultiRowWarps + warp) * kSharedStride + column] =
          row_warp_partials[row];
    }
  }
  __syncthreads();

  if (warp == 0 && lane < kBlockN) {
#pragma unroll
    for (int row = 0; row < RowsPerCta; ++row) {
      float value = 0.0f;
#pragma unroll
      for (int partial_warp = 0; partial_warp < kMultiRowWarps; ++partial_warp) {
        value += warp_partials[
            (row * kMultiRowWarps + partial_warp) * kSharedStride + column];
      }
      output[
          static_cast<int64_t>(row_base + row) * size_n + global_n] =
          ScalarTraits<Scalar>::from_float(value);
    }
  }
}

template <typename Scalar>
__global__ __launch_bounds__(kHmmaThreads) void amplin_gptq_w4_group128_gemm_hmma_v0_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups) {
  __shared__ __align__(32) Scalar shared_a[kHmmaBlockM * kHmmaBlockK];
  __shared__ __align__(32) Scalar shared_b[kHmmaBlockN * kHmmaBlockK];
  __shared__ __align__(32) float shared_c[kHmmaBlockM * kHmmaBlockN];

  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int tile_m = static_cast<int>(blockIdx.y);
  const int tile_n = static_cast<int>(blockIdx.x);
  const int global_m = tile_m * kHmmaBlockM;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

  for (int group = 0; group < num_groups; ++group) {
    for (int index = thread; index < kHmmaBlockM * kHmmaBlockK; index += kHmmaThreads) {
      const int row = index / kHmmaBlockK;
      const int group_k = index - row * kHmmaBlockK;
      shared_a[index] = input[
          static_cast<int64_t>(global_m + row) * size_k +
          group * kHmmaBlockK +
          group_k];
    }

    for (int word_index = thread; word_index < kHmmaBlockN * kHmmaPackedKWords; word_index += kHmmaThreads) {
      const int column = word_index / kHmmaPackedKWords;
      const int packed_k = word_index - column * kHmmaPackedKWords;
      const int64_t tile_offset =
          (static_cast<int64_t>(tile_n) * num_groups + group) *
          kHmmaBlockN *
          kHmmaPackedKWords;
      const uint32_t word = static_cast<uint32_t>(packed_qweight[tile_offset + word_index]);
      const float scale = ScalarTraits<Scalar>::to_float(
          packed_scales[
              (static_cast<int64_t>(tile_n) * num_groups + group) * kHmmaBlockN +
              column]);

#pragma unroll
      for (int code_offset = 0; code_offset < kPackFactor; ++code_offset) {
        const int code = static_cast<int>((word >> (code_offset * 4)) & 0x0fu);
        shared_b[
            column * kHmmaBlockK +
            packed_k * kPackFactor +
            code_offset] = ScalarTraits<Scalar>::from_float(static_cast<float>(code - 8) * scale);
      }
    }
    __syncthreads();

#pragma unroll
    for (int group_k = 0; group_k < kHmmaBlockK; group_k += 16) {
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          16,
          16,
          16,
          Scalar,
          nvcuda::wmma::row_major>
          a_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          16,
          16,
          16,
          Scalar,
          nvcuda::wmma::col_major>
          b_fragment;
      nvcuda::wmma::load_matrix_sync(a_fragment, shared_a + group_k, kHmmaBlockK);
      nvcuda::wmma::load_matrix_sync(
          b_fragment,
          shared_b + warp * 16 * kHmmaBlockK + group_k,
          kHmmaBlockK);
      nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    __syncthreads();
  }

  nvcuda::wmma::store_matrix_sync(
      shared_c + warp * 16,
      accumulator,
      kHmmaBlockN,
      nvcuda::wmma::mem_row_major);
  __syncthreads();

  for (int index = thread; index < kHmmaBlockM * kHmmaBlockN; index += kHmmaThreads) {
    const int row = index / kHmmaBlockN;
    const int column = index - row * kHmmaBlockN;
    output[
        static_cast<int64_t>(global_m + row) * size_n +
        tile_n * kHmmaBlockN +
        column] = ScalarTraits<Scalar>::from_float(shared_c[index]);
  }
}

template <typename Scalar, bool WideBStore, bool VectorA, bool AsyncA>
__global__ __launch_bounds__(kHmmaReuseThreads) void amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel(
    const Scalar* __restrict__ input,
    const int32_t* __restrict__ packed_qweight,
    const Scalar* __restrict__ packed_scales,
    Scalar* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups) {
  static_assert(!AsyncA || VectorA);
  __shared__ __align__(32) Scalar shared_a[kHmmaReuseBlockM * kHmmaBlockK];
  __shared__ __align__(32) Scalar shared_b[kHmmaBlockN * kHmmaBlockK];
  __shared__ __align__(32) float shared_c[kHmmaReuseWarps * kHmmaFragmentElements];

  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int warp_n = warp & 3;
  const int warp_m = (warp >> 2) * 32;
  const int tile_m = static_cast<int>(blockIdx.y);
  const int tile_n = static_cast<int>(blockIdx.x);
  const int global_m = tile_m * kHmmaReuseBlockM;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> accumulator_0;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> accumulator_1;
  nvcuda::wmma::fill_fragment(accumulator_0, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator_1, 0.0f);

  for (int group = 0; group < num_groups; ++group) {
    if constexpr (AsyncA) {
      for (
          int copy_index = thread;
          copy_index < kHmmaReuseACopies;
          copy_index += kHmmaReuseThreads) {
        const int row = copy_index / (kHmmaBlockK / kScalarsPerCpAsync);
        const int row_copy =
            copy_index - row * (kHmmaBlockK / kScalarsPerCpAsync);
        const int group_k = row_copy * kScalarsPerCpAsync;
        cp_async_16(
            shared_a + row * kHmmaBlockK + group_k,
            input +
                static_cast<int64_t>(global_m + row) * size_k +
                group * kHmmaBlockK +
                group_k);
      }
      cp_async_commit_group();
    } else if constexpr (VectorA) {
      for (
          int copy_index = thread;
          copy_index < kHmmaReuseACopies;
          copy_index += kHmmaReuseThreads) {
        const int row = copy_index / (kHmmaBlockK / kScalarsPerCpAsync);
        const int row_copy =
            copy_index - row * (kHmmaBlockK / kScalarsPerCpAsync);
        const int group_k = row_copy * kScalarsPerCpAsync;
        copy_16_sync(
            shared_a + row * kHmmaBlockK + group_k,
            input +
                static_cast<int64_t>(global_m + row) * size_k +
                group * kHmmaBlockK +
                group_k);
      }
    } else {
      for (int index = thread; index < kHmmaReuseBlockM * kHmmaBlockK; index += kHmmaReuseThreads) {
        const int row = index / kHmmaBlockK;
        const int group_k = index - row * kHmmaBlockK;
        shared_a[index] = input[
            static_cast<int64_t>(global_m + row) * size_k +
            group * kHmmaBlockK +
            group_k];
      }
    }

    for (
        int word_index = thread;
        word_index < kHmmaBlockN * kHmmaPackedKWords;
        word_index += kHmmaReuseThreads) {
      const int column = word_index / kHmmaPackedKWords;
      const int packed_k = word_index - column * kHmmaPackedKWords;
      const int64_t tile_offset =
          (static_cast<int64_t>(tile_n) * num_groups + group) *
          kHmmaBlockN *
          kHmmaPackedKWords;
      const uint32_t word = static_cast<uint32_t>(packed_qweight[tile_offset + word_index]);
      const float scale = ScalarTraits<Scalar>::to_float(
          packed_scales[
              (static_cast<int64_t>(tile_n) * num_groups + group) * kHmmaBlockN +
              column]);

      if constexpr (WideBStore) {
        store_dequantized_word_128(
            shared_b + column * kHmmaBlockK + packed_k * kPackFactor,
            word,
            scale);
      } else {
#pragma unroll
        for (int code_offset = 0; code_offset < kPackFactor; ++code_offset) {
          const int code = static_cast<int>((word >> (code_offset * 4)) & 0x0fu);
          shared_b[
              column * kHmmaBlockK +
              packed_k * kPackFactor +
              code_offset] = ScalarTraits<Scalar>::from_float(static_cast<float>(code - 8) * scale);
        }
      }
    }
    if constexpr (AsyncA) {
      cp_async_wait_all();
    }
    __syncthreads();

#pragma unroll
    for (int group_k = 0; group_k < kHmmaBlockK; group_k += 16) {
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          16,
          16,
          16,
          Scalar,
          nvcuda::wmma::row_major>
          a_fragment;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          16,
          16,
          16,
          Scalar,
          nvcuda::wmma::col_major>
          b_fragment;
      nvcuda::wmma::load_matrix_sync(
          b_fragment,
          shared_b + warp_n * 16 * kHmmaBlockK + group_k,
          kHmmaBlockK);
      nvcuda::wmma::load_matrix_sync(
          a_fragment,
          shared_a + warp_m * kHmmaBlockK + group_k,
          kHmmaBlockK);
      nvcuda::wmma::mma_sync(accumulator_0, a_fragment, b_fragment, accumulator_0);
      nvcuda::wmma::load_matrix_sync(
          a_fragment,
          shared_a + (warp_m + 16) * kHmmaBlockK + group_k,
          kHmmaBlockK);
      nvcuda::wmma::mma_sync(accumulator_1, a_fragment, b_fragment, accumulator_1);
    }
    __syncthreads();
  }

  float* warp_c = shared_c + warp * kHmmaFragmentElements;
  const int global_n = tile_n * kHmmaBlockN + warp_n * 16;
  nvcuda::wmma::store_matrix_sync(
      warp_c,
      accumulator_0,
      16,
      nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kHmmaFragmentElements; index += 32) {
    const int row = index / 16;
    const int column = index - row * 16;
    output[
        static_cast<int64_t>(global_m + warp_m + row) * size_n +
        global_n +
        column] = ScalarTraits<Scalar>::from_float(warp_c[index]);
  }
  __syncwarp();

  nvcuda::wmma::store_matrix_sync(
      warp_c,
      accumulator_1,
      16,
      nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kHmmaFragmentElements; index += 32) {
    const int row = index / 16;
    const int column = index - row * 16;
    output[
        static_cast<int64_t>(global_m + warp_m + 16 + row) * size_n +
        global_n +
        column] = ScalarTraits<Scalar>::from_float(warp_c[index]);
  }
}

template <typename Scalar>
void launch_amplin_gptq_w4_group128(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    int max_grid_x,
    cudaStream_t stream) {
  using TorchScalar = typename ScalarTraits<Scalar>::TorchScalar;
  if (size_m == 1 && size_k == kV0SizeK && size_n % kBlockN == 0) {
    const int blocks = size_n / kBlockN;
    amplin_gptq_w4_group128_gemv_v0_kernel<Scalar><<<blocks, kV0Threads, 0, stream>>>(
        reinterpret_cast<const Scalar*>(input.data_ptr<TorchScalar>()),
        qweight.data_ptr<int32_t>(),
        reinterpret_cast<const Scalar*>(scales.data_ptr<TorchScalar>()),
        reinterpret_cast<Scalar*>(output.data_ptr<TorchScalar>()),
        size_n);
    return;
  }

  if (size_m == 1 && size_k == kWideSizeK && size_n % kBlockN == 0) {
    const int blocks = size_n / kBlockN;
    amplin_gptq_w4_group128_gemv_k12288_wide_kernel<Scalar>
        <<<blocks, kWideThreads, 0, stream>>>(
            reinterpret_cast<const Scalar*>(input.data_ptr<TorchScalar>()),
            qweight.data_ptr<int32_t>(),
            reinterpret_cast<const Scalar*>(scales.data_ptr<TorchScalar>()),
            reinterpret_cast<Scalar*>(output.data_ptr<TorchScalar>()),
            size_n);
    return;
  }

  const int blocks_n = (size_n + kBlockN - 1) / kBlockN;
  const int64_t total_blocks = static_cast<int64_t>(size_m) * blocks_n;
  TORCH_CHECK(
      total_blocks <= max_grid_x,
      "Amplin activation-row/output-tile grid exceeds the selected CUDA device limit");
  const int group_pairs = (num_groups + 1) / 2;
  const int warps = std::min(kGeneralMaxWarps, group_pairs);
  const int threads = warps * 32;
  amplin_gptq_w4_group128_group_stride_kernel<Scalar><<<static_cast<int>(total_blocks), threads, 0, stream>>>(
      reinterpret_cast<const Scalar*>(input.data_ptr<TorchScalar>()),
      qweight.data_ptr<int32_t>(),
      reinterpret_cast<const Scalar*>(scales.data_ptr<TorchScalar>()),
      reinterpret_cast<Scalar*>(output.data_ptr<TorchScalar>()),
      size_k,
      size_n,
      num_groups,
      blocks_n);
}

template <typename Scalar>
void launch_amplin_gptq_w4_group128_k12288_wide(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    torch::Tensor& output,
    int size_n,
    cudaStream_t stream) {
  using TorchScalar = typename ScalarTraits<Scalar>::TorchScalar;
  const int blocks = size_n / kBlockN;
  amplin_gptq_w4_group128_gemv_k12288_wide_kernel<Scalar>
      <<<blocks, kWideThreads, 0, stream>>>(
          reinterpret_cast<const Scalar*>(input.data_ptr<TorchScalar>()),
          qweight.data_ptr<int32_t>(),
          reinterpret_cast<const Scalar*>(scales.data_ptr<TorchScalar>()),
          reinterpret_cast<Scalar*>(output.data_ptr<TorchScalar>()),
          size_n);
}

template <typename Scalar, int RowsPerCta>
void launch_amplin_gptq_w4_group128_multirow(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    torch::Tensor& output,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    int max_grid_x,
    cudaStream_t stream) {
  using TorchScalar = typename ScalarTraits<Scalar>::TorchScalar;
  const int blocks_n = size_n / kBlockN;
  const int64_t total_blocks =
      static_cast<int64_t>(size_m / RowsPerCta) * blocks_n;
  TORCH_CHECK(
      total_blocks <= max_grid_x,
      "Amplin multi-row grid exceeds the selected CUDA device limit");
  amplin_gptq_w4_group128_gemv_multirow_kernel<Scalar, RowsPerCta>
      <<<static_cast<int>(total_blocks), kMultiRowThreads, 0, stream>>>(
          reinterpret_cast<const Scalar*>(input.data_ptr<TorchScalar>()),
          qweight.data_ptr<int32_t>(),
          reinterpret_cast<const Scalar*>(scales.data_ptr<TorchScalar>()),
          reinterpret_cast<Scalar*>(output.data_ptr<TorchScalar>()),
          size_k,
          size_n,
          num_groups,
          blocks_n);
}

}  // namespace

torch::Tensor amplin_gptq_w4_group128_gemv_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "Amplin input must be CUDA");
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(), "Amplin weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == qweight.device() && input.device() == scales.device(),
      "Amplin tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin input must be FP16 or BF16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "Amplin qweight must be int32");
  TORCH_CHECK(scales.scalar_type() == input.scalar_type(), "Amplin scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 && qweight.dim() == 2 && scales.dim() == 2,
      "Amplin input must have at least two dimensions and qweight/scales must be 2D");
  TORCH_CHECK(
      input.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous(),
      "Amplin input, qweight, and scales must be contiguous");

  const int64_t size_k = input.size(-1);
  TORCH_CHECK(
      size_k > 0 && size_k % kGroupSize == 0,
      "Amplin K must be positive and divisible by group size 128");
  const int64_t size_n = qweight.size(1);
  TORCH_CHECK(size_n > 0, "Amplin N must be positive");
  TORCH_CHECK(
      qweight.size(0) == size_k / kPackFactor,
      "Amplin qweight must have canonical shape [K/8, N]");
  const int64_t num_groups = size_k / kGroupSize;
  TORCH_CHECK(
      scales.size(0) == num_groups && scales.size(1) == size_n,
      "Amplin scales must have shape [K/128, N]");
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(size_m > 0, "Amplin input must contain at least one activation row");
  TORCH_CHECK(
      size_m <= std::numeric_limits<int>::max() &&
          size_k <= std::numeric_limits<int>::max() &&
          size_n <= std::numeric_limits<int>::max() &&
          num_groups <= std::numeric_limits<int>::max(),
      "Amplin tensor dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin V0 requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = size_n;
  auto output = torch::empty(output_sizes, input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    launch_amplin_gptq_w4_group128<half>(
        input,
        qweight,
        scales,
        output,
        static_cast<int>(size_m),
        static_cast<int>(size_k),
        static_cast<int>(size_n),
        static_cast<int>(num_groups),
        properties->maxGridSize[0],
        stream);
  } else {
    launch_amplin_gptq_w4_group128<__nv_bfloat16>(
        input,
        qweight,
        scales,
        output,
        static_cast<int>(size_m),
        static_cast<int>(size_k),
        static_cast<int>(size_n),
        static_cast<int>(num_groups),
        properties->maxGridSize[0],
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor amplin_gptq_w4_group128_gemv_k12288_wide_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "Amplin K12288-wide input must be CUDA");
  TORCH_CHECK(
      qweight.is_cuda() && scales.is_cuda(),
      "Amplin K12288-wide weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == qweight.device() && input.device() == scales.device(),
      "Amplin K12288-wide tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin K12288-wide input must be FP16 or BF16");
  TORCH_CHECK(
      qweight.scalar_type() == at::kInt,
      "Amplin K12288-wide qweight must be int32");
  TORCH_CHECK(
      scales.scalar_type() == input.scalar_type(),
      "Amplin K12288-wide scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 && qweight.dim() == 2 && scales.dim() == 2,
      "Amplin K12288-wide input must have at least two dimensions and qweight/scales must be 2D");
  TORCH_CHECK(
      input.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous(),
      "Amplin K12288-wide input, qweight, and scales must be contiguous");

  const int64_t size_k = input.size(-1);
  TORCH_CHECK(size_k == kWideSizeK, "Amplin K12288-wide requires K=12288");
  const int64_t size_n = qweight.size(1);
  TORCH_CHECK(
      size_n > 0 && size_n % kBlockN == 0,
      "Amplin K12288-wide N must be positive and divisible by 16");
  TORCH_CHECK(
      qweight.size(0) == size_k / kPackFactor,
      "Amplin K12288-wide qweight must have canonical shape [K/8, N]");
  const int64_t num_groups = size_k / kGroupSize;
  TORCH_CHECK(
      scales.size(0) == num_groups && scales.size(1) == size_n,
      "Amplin K12288-wide scales must have shape [K/128, N]");
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(
      size_m == 1,
      "Amplin K12288-wide prototype requires exactly one flattened activation row");
  TORCH_CHECK(
      size_n <= std::numeric_limits<int>::max(),
      "Amplin K12288-wide N exceeds int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const int device = input.get_device();
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(device);
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin K12288-wide requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = size_n;
  auto output = torch::empty(output_sizes, input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  if (input.scalar_type() == at::kHalf) {
    launch_amplin_gptq_w4_group128_k12288_wide<half>(
        input,
        qweight,
        scales,
        output,
        static_cast<int>(size_n),
        stream);
  } else {
    launch_amplin_gptq_w4_group128_k12288_wide<__nv_bfloat16>(
        input,
        qweight,
        scales,
        output,
        static_cast<int>(size_n),
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor amplin_gptq_w4_group128_gemv_multirow_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "Amplin multi-row input must be CUDA");
  TORCH_CHECK(
      qweight.is_cuda() && scales.is_cuda(),
      "Amplin multi-row weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == qweight.device() && input.device() == scales.device(),
      "Amplin multi-row tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin multi-row input must be FP16 or BF16");
  TORCH_CHECK(
      qweight.scalar_type() == at::kInt,
      "Amplin multi-row qweight must be int32");
  TORCH_CHECK(
      scales.scalar_type() == input.scalar_type(),
      "Amplin multi-row scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 && qweight.dim() == 2 && scales.dim() == 2,
      "Amplin multi-row input must have at least two dimensions and qweight/scales must be 2D");
  TORCH_CHECK(
      input.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous(),
      "Amplin multi-row input, qweight, and scales must be contiguous");

  const int64_t size_k = input.size(-1);
  TORCH_CHECK(
      size_k == 3072 || size_k == 4096 || size_k == 12288,
      "Amplin multi-row prototype requires K=3072, 4096, or 12288");
  const int64_t size_n = qweight.size(1);
  TORCH_CHECK(
      size_n > 0 && size_n % kBlockN == 0,
      "Amplin multi-row N must be positive and divisible by 16");
  TORCH_CHECK(
      qweight.size(0) == size_k / kPackFactor,
      "Amplin multi-row qweight must have canonical shape [K/8, N]");
  const int64_t num_groups = size_k / kGroupSize;
  TORCH_CHECK(
      num_groups % 2 == 0,
      "Amplin multi-row requires an even K/128 group count");
  TORCH_CHECK(
      scales.size(0) == num_groups && scales.size(1) == size_n,
      "Amplin multi-row scales must have shape [K/128, N]");
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(
      size_m == 2 || (size_m >= kMultiRowMaxRows &&
                      size_m <= 16 &&
                      size_m % kMultiRowMaxRows == 0),
      "Amplin multi-row prototype requires flattened M=2, 4, 8, or 16");
  TORCH_CHECK(
      size_m <= std::numeric_limits<int>::max() &&
          size_k <= std::numeric_limits<int>::max() &&
          size_n <= std::numeric_limits<int>::max() &&
          num_groups <= std::numeric_limits<int>::max(),
      "Amplin multi-row tensor dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const int device = input.get_device();
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(device);
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin multi-row requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = size_n;
  auto output = torch::empty(output_sizes, input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  if (input.scalar_type() == at::kHalf) {
    if (size_m == 2) {
      launch_amplin_gptq_w4_group128_multirow<half, 2>(
          input,
          qweight,
          scales,
          output,
          static_cast<int>(size_m),
          static_cast<int>(size_k),
          static_cast<int>(size_n),
          static_cast<int>(num_groups),
          properties->maxGridSize[0],
          stream);
    } else {
      launch_amplin_gptq_w4_group128_multirow<half, 4>(
          input,
          qweight,
          scales,
          output,
          static_cast<int>(size_m),
          static_cast<int>(size_k),
          static_cast<int>(size_n),
          static_cast<int>(num_groups),
          properties->maxGridSize[0],
          stream);
    }
  } else if (size_m == 2) {
    launch_amplin_gptq_w4_group128_multirow<__nv_bfloat16, 2>(
        input,
        qweight,
        scales,
        output,
        static_cast<int>(size_m),
        static_cast<int>(size_k),
        static_cast<int>(size_n),
        static_cast<int>(num_groups),
        properties->maxGridSize[0],
        stream);
  } else {
    launch_amplin_gptq_w4_group128_multirow<__nv_bfloat16, 4>(
        input,
        qweight,
        scales,
        output,
        static_cast<int>(size_m),
        static_cast<int>(size_k),
        static_cast<int>(size_n),
        static_cast<int>(num_groups),
        properties->maxGridSize[0],
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

namespace {

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_impl(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n,
    HmmaLaunchMode launch_mode) {
  TORCH_CHECK(input.is_cuda(), "Amplin HMMA input must be CUDA");
  TORCH_CHECK(
      packed_qweight.is_cuda() && packed_scales.is_cuda(),
      "Amplin HMMA weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == packed_qweight.device() && input.device() == packed_scales.device(),
      "Amplin HMMA tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin HMMA input must be FP16 or BF16");
  TORCH_CHECK(packed_qweight.scalar_type() == at::kInt, "Amplin HMMA qweight must be int32");
  TORCH_CHECK(
      packed_scales.scalar_type() == input.scalar_type(),
      "Amplin HMMA scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 && packed_qweight.dim() == 4 && packed_scales.dim() == 3,
      "Amplin HMMA input must have at least two dimensions and packed weights must be 4D/3D");
  TORCH_CHECK(
      input.is_contiguous() && packed_qweight.is_contiguous() && packed_scales.is_contiguous(),
      "Amplin HMMA tensors must be contiguous");

  const int64_t packed_n_tiles = packed_qweight.size(0);
  const int64_t num_groups = packed_qweight.size(1);
  TORCH_CHECK(
      packed_n_tiles > 0 &&
          num_groups > 0 &&
          packed_qweight.size(2) == kHmmaBlockN &&
          packed_qweight.size(3) == kHmmaPackedKWords,
      "Amplin HMMA qweight must have shape [N/64, K/128, 64, 16]");
  TORCH_CHECK(
      packed_scales.size(0) == packed_n_tiles &&
          packed_scales.size(1) == num_groups &&
          packed_scales.size(2) == kHmmaBlockN,
      "Amplin HMMA scales must have shape [N/64, K/128, 64]");

  const int64_t size_k = input.size(-1);
  TORCH_CHECK(
      size_k == num_groups * kHmmaBlockK,
      "Amplin HMMA input K must match packed K/128 groups");
  TORCH_CHECK(
      logical_n > 0 &&
          logical_n % kHmmaBlockN == 0 &&
          logical_n <= packed_n_tiles * kHmmaBlockN,
      "Amplin HMMA logical N must be positive, divisible by 64, and fit the packed layout");
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(
      size_m > 0 && size_m % kHmmaBlockM == 0,
      "Amplin HMMA flattened M must be positive and divisible by 16");
  TORCH_CHECK(
      size_m <= std::numeric_limits<int>::max() &&
          size_k <= std::numeric_limits<int>::max() &&
          logical_n <= std::numeric_limits<int>::max() &&
          num_groups <= std::numeric_limits<int>::max(),
      "Amplin HMMA tensor dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin HMMA requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);
  const int64_t m64_grid_size =
      (logical_n / kHmmaBlockN) *
      (size_m / kHmmaReuseBlockM);
  const bool force_m64 =
      launch_mode == HmmaLaunchMode::kM64V1 ||
      launch_mode == HmmaLaunchMode::kM64V2 ||
      launch_mode == HmmaLaunchMode::kM64V2SyncA128 ||
      launch_mode == HmmaLaunchMode::kM64V3;
  TORCH_CHECK(
      !force_m64 || size_m % kHmmaReuseBlockM == 0,
      "Amplin HMMA M64 control requires flattened M divisible by 64");
  const bool use_m64_reuse =
      force_m64 ||
      (launch_mode == HmmaLaunchMode::kAuto &&
       size_m % kHmmaReuseBlockM == 0 &&
       m64_grid_size >= properties->multiProcessorCount);
  const bool use_wide_b_store = launch_mode != HmmaLaunchMode::kM64V1;
  const bool use_vector_a =
      launch_mode == HmmaLaunchMode::kAuto ||
      launch_mode == HmmaLaunchMode::kM64V2SyncA128 ||
      launch_mode == HmmaLaunchMode::kM64V3;
  const bool use_async_a =
      launch_mode == HmmaLaunchMode::kAuto ||
      launch_mode == HmmaLaunchMode::kM64V3;
  const int launch_block_m = use_m64_reuse ? kHmmaReuseBlockM : kHmmaBlockM;
  TORCH_CHECK(
      logical_n / kHmmaBlockN <= properties->maxGridSize[0] &&
          size_m / launch_block_m <= properties->maxGridSize[1],
      "Amplin HMMA grid exceeds the selected CUDA device limit");

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = logical_n;
  auto output = torch::empty(output_sizes, input.options());
  const dim3 grid(
      static_cast<unsigned int>(logical_n / kHmmaBlockN),
      static_cast<unsigned int>(size_m / launch_block_m));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    if (use_m64_reuse) {
      if (use_async_a) {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<half, true, true, true>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else if (use_vector_a) {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<half, true, true, false>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else if (use_wide_b_store) {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<half, true, false, false>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<half, false, false, false>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      }
    } else {
      amplin_gptq_w4_group128_gemm_hmma_v0_kernel<half><<<grid, kHmmaThreads, 0, stream>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          packed_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    }
  } else {
    if (use_m64_reuse) {
      if (use_async_a) {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<__nv_bfloat16, true, true, true>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else if (use_vector_a) {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<__nv_bfloat16, true, true, false>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else if (use_wide_b_store) {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<__nv_bfloat16, true, false, false>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else {
        amplin_gptq_w4_group128_gemm_hmma_m64_reuse_kernel<__nv_bfloat16, false, false, false>
            <<<grid, kHmmaReuseThreads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                packed_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      }
    } else {
      amplin_gptq_w4_group128_gemm_hmma_v0_kernel<__nv_bfloat16><<<grid, kHmmaThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
          packed_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_impl(
      input,
      packed_qweight,
      packed_scales,
      logical_n,
      HmmaLaunchMode::kAuto);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_v0_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_impl(
      input,
      packed_qweight,
      packed_scales,
      logical_n,
      HmmaLaunchMode::kV0);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v1_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_impl(
      input,
      packed_qweight,
      packed_scales,
      logical_n,
      HmmaLaunchMode::kM64V1);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v2_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_impl(
      input,
      packed_qweight,
      packed_scales,
      logical_n,
      HmmaLaunchMode::kM64V2);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v2_sync_a128_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_impl(
      input,
      packed_qweight,
      packed_scales,
      logical_n,
      HmmaLaunchMode::kM64V2SyncA128);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v3_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_impl(
      input,
      packed_qweight,
      packed_scales,
      logical_n,
      HmmaLaunchMode::kM64V3);
}

template <bool GlobalA>
torch::Tensor amplin_mma_lane_tile_cuda_impl(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "Amplin MMA-lane input must be CUDA");
  TORCH_CHECK(
      packed_lane_qweight.is_cuda() && scales.is_cuda(),
      "Amplin MMA-lane weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == packed_lane_qweight.device() && input.device() == scales.device(),
      "Amplin MMA-lane tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin MMA-lane input must be FP16 or BF16");
  TORCH_CHECK(
      packed_lane_qweight.scalar_type() == at::kInt,
      "Amplin MMA-lane qweight must be int32");
  TORCH_CHECK(
      scales.scalar_type() == input.scalar_type(),
      "Amplin MMA-lane scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() == 2 &&
          input.size(0) == kMmaM &&
          input.size(1) == kMmaK,
      "Amplin MMA-lane input must have shape [16, 16]");
  TORCH_CHECK(
      packed_lane_qweight.dim() == 1 &&
          packed_lane_qweight.numel() == kMmaLaneWords,
      "Amplin MMA-lane qweight must have shape [32]");
  TORCH_CHECK(
      scales.dim() == 1 && scales.numel() == kMmaLaneTileN,
      "Amplin MMA-lane scales must have shape [16]");
  TORCH_CHECK(
      input.is_contiguous() &&
          packed_lane_qweight.is_contiguous() &&
          scales.is_contiguous(),
      "Amplin MMA-lane tensors must be contiguous");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin MMA-lane prototype requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);

  auto output = torch::empty({kMmaM, kMmaLaneTileN}, input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    if constexpr (GlobalA) {
      amplin_mma_lane_tile_global_a_kernel<half><<<1, kMmaLanes, 0, stream>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()));
    } else {
      amplin_mma_lane_tile_kernel<half><<<1, kMmaLanes, 0, stream>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()));
    }
  } else {
    if constexpr (GlobalA) {
      amplin_mma_lane_tile_global_a_kernel<__nv_bfloat16><<<1, kMmaLanes, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    } else {
      amplin_mma_lane_tile_kernel<__nv_bfloat16><<<1, kMmaLanes, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor amplin_mma_lane_tile_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor scales) {
  return amplin_mma_lane_tile_cuda_impl<false>(
      input,
      packed_lane_qweight,
      scales);
}

torch::Tensor amplin_mma_lane_tile_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor scales) {
  return amplin_mma_lane_tile_cuda_impl<true>(
      input,
      packed_lane_qweight,
      scales);
}

template <bool GlobalA, int BlockM, int BlockN = kHmmaBlockN>
torch::Tensor amplin_mma_lane_cuda_impl(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  static_assert(
      BlockM == kHmmaReuseBlockM || BlockM == kMmaLaneM32BlockM);
  static_assert(BlockM != kMmaLaneM32BlockM || GlobalA);
  static_assert(BlockN == kHmmaBlockN || BlockN == kMmaLaneM32N32BlockN);
  static_assert(BlockN == kHmmaBlockN || BlockM == kMmaLaneM32BlockM);

  TORCH_CHECK(input.is_cuda(), "Amplin MMA-lane input must be CUDA");
  TORCH_CHECK(
      packed_lane_qweight.is_cuda() && packed_scales.is_cuda(),
      "Amplin MMA-lane weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == packed_lane_qweight.device() &&
          input.device() == packed_scales.device(),
      "Amplin MMA-lane tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin MMA-lane input must be FP16 or BF16");
  TORCH_CHECK(
      packed_lane_qweight.scalar_type() == at::kInt,
      "Amplin MMA-lane qweight must be int32");
  TORCH_CHECK(
      packed_scales.scalar_type() == input.scalar_type(),
      "Amplin MMA-lane scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 &&
          packed_lane_qweight.dim() == 5 &&
          packed_scales.dim() == 3,
      "Amplin MMA-lane input must have at least two dimensions and packed weights must be 5D/3D");
  TORCH_CHECK(
      input.is_contiguous() &&
          packed_lane_qweight.is_contiguous() &&
          packed_scales.is_contiguous(),
      "Amplin MMA-lane tensors must be contiguous");

  const int64_t packed_n_tiles = packed_lane_qweight.size(0);
  const int64_t num_groups = packed_lane_qweight.size(1);
  TORCH_CHECK(
      packed_n_tiles > 0 &&
          num_groups > 0 &&
          packed_lane_qweight.size(2) == kHmmaBlockK / kMmaK &&
          packed_lane_qweight.size(3) == kHmmaWarps &&
          packed_lane_qweight.size(4) == kMmaLanes,
      "Amplin MMA-lane qweight must have shape [N/64, K/128, 8, 4, 32]");
  TORCH_CHECK(
      packed_scales.size(0) == packed_n_tiles &&
          packed_scales.size(1) == num_groups &&
          packed_scales.size(2) == kHmmaBlockN,
      "Amplin MMA-lane scales must have shape [N/64, K/128, 64]");

  const int64_t size_k = input.size(-1);
  TORCH_CHECK(
      size_k == num_groups * kHmmaBlockK,
      "Amplin MMA-lane input K must match packed K/128 groups");
  if constexpr (BlockN == kMmaLaneM32N32BlockN) {
    TORCH_CHECK(
        logical_n > 0 &&
            logical_n % kMmaN == 0 &&
            logical_n <= packed_n_tiles * kHmmaBlockN,
        "Amplin MMA-lane N32 logical N must be positive, divisible by 8, and fit the packed layout");
  } else {
    TORCH_CHECK(
        logical_n > 0 &&
            logical_n % kHmmaBlockN == 0 &&
            logical_n <= packed_n_tiles * kHmmaBlockN,
        "Amplin MMA-lane logical N must be positive, divisible by 64, and fit the packed layout");
  }
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(
      size_m > 0 && size_m % BlockM == 0,
      "Amplin MMA-lane flattened M must be positive and divisible by ",
      BlockM);
  TORCH_CHECK(
      size_m <= std::numeric_limits<int>::max() &&
          size_k <= std::numeric_limits<int>::max() &&
          logical_n <= std::numeric_limits<int>::max() &&
          num_groups <= std::numeric_limits<int>::max(),
      "Amplin MMA-lane tensor dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin MMA-lane requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);
  const int64_t grid_n = (logical_n + BlockN - 1) / BlockN;
  TORCH_CHECK(
      grid_n <= properties->maxGridSize[0] &&
          size_m / BlockM <= properties->maxGridSize[1],
      "Amplin MMA-lane grid exceeds the selected CUDA device limit");

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = logical_n;
  auto output = torch::empty(output_sizes, input.options());
  const dim3 grid(
      static_cast<unsigned int>(grid_n),
      static_cast<unsigned int>(size_m / BlockM));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    if constexpr (BlockN == kMmaLaneM32N32BlockN) {
      if (logical_n % kMmaLaneM32N32BlockN == 0) {
        amplin_mma_lane_m32_n32_global_a_kernel<half, false>
            <<<grid, kMmaLaneM32N32Threads, 0, stream>>>(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                packed_lane_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else {
        amplin_mma_lane_m32_n32_global_a_kernel<half, true>
            <<<grid, kMmaLaneM32N32Threads, 0, stream>>>(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                packed_lane_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      }
    } else if constexpr (BlockM == kMmaLaneM32BlockM) {
      amplin_mma_lane_m32_global_a_kernel<half><<<grid, kMmaLaneM32Threads, 0, stream>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    } else if constexpr (GlobalA) {
      amplin_mma_lane_m64_global_a_kernel<half><<<grid, kHmmaReuseThreads, 0, stream>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    } else {
      amplin_mma_lane_m64_kernel<half><<<grid, kHmmaReuseThreads, 0, stream>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    }
  } else {
    if constexpr (BlockN == kMmaLaneM32N32BlockN) {
      if (logical_n % kMmaLaneM32N32BlockN == 0) {
        amplin_mma_lane_m32_n32_global_a_kernel<__nv_bfloat16, false>
            <<<grid, kMmaLaneM32N32Threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                packed_lane_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      } else {
        amplin_mma_lane_m32_n32_global_a_kernel<__nv_bfloat16, true>
            <<<grid, kMmaLaneM32N32Threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                packed_lane_qweight.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                static_cast<int>(size_k),
                static_cast<int>(logical_n),
                static_cast<int>(num_groups));
      }
    } else if constexpr (BlockM == kMmaLaneM32BlockM) {
      amplin_mma_lane_m32_global_a_kernel<__nv_bfloat16><<<grid, kMmaLaneM32Threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    } else if constexpr (GlobalA) {
      amplin_mma_lane_m64_global_a_kernel<__nv_bfloat16><<<grid, kHmmaReuseThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    } else {
      amplin_mma_lane_m64_kernel<__nv_bfloat16><<<grid, kHmmaReuseThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
          packed_lane_qweight.data_ptr<int32_t>(),
          reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          static_cast<int>(size_k),
          static_cast<int>(logical_n),
          static_cast<int>(num_groups));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor amplin_mma_lane_m64_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_cuda_impl<false, kHmmaReuseBlockM>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m64_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_cuda_impl<true, kHmmaReuseBlockM>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_cuda_impl<true, kMmaLaneM32BlockM>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_n32_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_cuda_impl<
      true,
      kMmaLaneM32BlockM,
      kMmaLaneM32N32BlockN>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

template <
    int SplitKWarps,
    bool OutputN32 = false,
    bool PipelineKSteps = false,
    bool InterleavedN32Words = false>
torch::Tensor amplin_mma_lane_m16_n16_cuda_impl(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  static_assert(!PipelineKSteps || OutputN32);
  static_assert(!InterleavedN32Words || (OutputN32 && PipelineKSteps));
  static_assert(!InterleavedN32Words || SplitKWarps == kMmaLaneSplitK12Warps);
  TORCH_CHECK(input.is_cuda(), "Amplin padded-M16 input must be CUDA");
  TORCH_CHECK(
      packed_lane_qweight.is_cuda() && packed_scales.is_cuda(),
      "Amplin padded-M16 weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == packed_lane_qweight.device() &&
          input.device() == packed_scales.device(),
      "Amplin padded-M16 tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin padded-M16 input must be FP16 or BF16");
  TORCH_CHECK(
      packed_lane_qweight.scalar_type() == at::kInt,
      "Amplin padded-M16 qweight must be int32");
  TORCH_CHECK(
      packed_scales.scalar_type() == input.scalar_type(),
      "Amplin padded-M16 scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 &&
          packed_lane_qweight.dim() == 5 &&
          packed_scales.dim() == 3,
      "Amplin padded-M16 input must have at least two dimensions and packed weights must be 5D/3D");
  TORCH_CHECK(
      input.is_contiguous() &&
          packed_lane_qweight.is_contiguous() &&
          packed_scales.is_contiguous(),
      "Amplin padded-M16 tensors must be contiguous");

  const int64_t packed_n_tiles = packed_lane_qweight.size(0);
  const int64_t num_groups = packed_lane_qweight.size(1);
  if constexpr (InterleavedN32Words) {
    TORCH_CHECK(
        packed_n_tiles > 0 &&
            packed_n_tiles % 2 == 0 &&
            num_groups > 0 &&
            packed_lane_qweight.size(2) == kHmmaBlockK / kMmaK &&
            packed_lane_qweight.size(3) == kMmaLanes &&
            packed_lane_qweight.size(4) == 2,
        "Amplin padded-M16 interleaved qweight must have shape [N/32, K/128, 8, 32, 2]");
  } else {
    TORCH_CHECK(
        packed_n_tiles > 0 &&
            num_groups > 0 &&
            packed_lane_qweight.size(2) == kHmmaBlockK / kMmaK &&
            packed_lane_qweight.size(3) == kHmmaWarps &&
            packed_lane_qweight.size(4) == kMmaLanes,
        "Amplin padded-M16 qweight must have shape [N/64, K/128, 8, 4, 32]");
  }
  TORCH_CHECK(
      packed_scales.size(0) * (InterleavedN32Words ? 2 : 1) == packed_n_tiles &&
          packed_scales.size(1) == num_groups &&
          packed_scales.size(2) == kHmmaBlockN,
      "Amplin padded-M16 scales must have shape [N/64, K/128, 64]");

  const int64_t size_k = input.size(-1);
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(
      size_k == num_groups * kHmmaBlockK,
      "Amplin padded-M16 input K must match packed K/128 groups");
  if constexpr (SplitKWarps != 0) {
    TORCH_CHECK(
        size_k == kWideSizeK && num_groups % SplitKWarps == 0,
        "Amplin padded-M16 split-K",
        SplitKWarps,
        " requires K=12288 and evenly divisible groups");
  }
  TORCH_CHECK(
      size_m == 2 || size_m == 4 || size_m == 8 || size_m == 16,
      "Amplin padded-M16 flattened M must be one of 2, 4, 8, or 16");
  TORCH_CHECK(
      logical_n > 0 &&
          logical_n % (OutputN32 ? 2 * kMmaLanePaddedBlockN : kMmaLanePaddedBlockN) == 0 &&
          logical_n <=
              packed_n_tiles *
                  (InterleavedN32Words ? 2 * kMmaLanePaddedBlockN : kHmmaBlockN),
      "Amplin padded-M16 logical N must be positive, divisible by ",
      OutputN32 ? 32 : 16,
      ", and fit the packed layout");
  TORCH_CHECK(
      size_k <= std::numeric_limits<int>::max() &&
          logical_n <= std::numeric_limits<int>::max() &&
          num_groups <= std::numeric_limits<int>::max(),
      "Amplin padded-M16 tensor dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin padded-M16 requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);
  const int64_t grid_n =
      logical_n / (OutputN32 ? 2 * kMmaLanePaddedBlockN : kMmaLanePaddedBlockN);
  TORCH_CHECK(
      grid_n <= properties->maxGridSize[0],
      "Amplin padded-M16 grid exceeds the selected CUDA device limit");

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = logical_n;
  auto output = torch::empty(output_sizes, input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    if constexpr (OutputN32) {
      if constexpr (SplitKWarps == kMmaLaneSplitK12Warps) {
        if constexpr (PipelineKSteps) {
          if constexpr (InterleavedN32Words) {
            amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_kernel<half>
                <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
                    reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                    packed_lane_qweight.data_ptr<int32_t>(),
                    reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                    static_cast<int>(size_m),
                    static_cast<int>(size_k),
                    static_cast<int>(logical_n),
                    static_cast<int>(num_groups));
          } else {
            amplin_mma_lane_m16_n32_splitk12_pipe2_kernel<half>
                <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
                    reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                    packed_lane_qweight.data_ptr<int32_t>(),
                    reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                    static_cast<int>(size_m),
                    static_cast<int>(size_k),
                    static_cast<int>(logical_n),
                    static_cast<int>(num_groups));
          }
        } else {
          amplin_mma_lane_m16_n32_splitk12_kernel<half>
              <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
                  reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                  packed_lane_qweight.data_ptr<int32_t>(),
                  reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                  reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                  static_cast<int>(size_m),
                  static_cast<int>(size_k),
                  static_cast<int>(logical_n),
                  static_cast<int>(num_groups));
        }
      } else {
        static_assert(SplitKWarps == kMmaLaneSplitK16Warps);
        if constexpr (PipelineKSteps) {
          amplin_mma_lane_m16_n32_splitk16_pipe2_kernel<half>
              <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK16Threads, 0, stream>>>(
                  reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                  packed_lane_qweight.data_ptr<int32_t>(),
                  reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                  reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                  static_cast<int>(size_m),
                  static_cast<int>(size_k),
                  static_cast<int>(logical_n),
                  static_cast<int>(num_groups));
        } else {
          amplin_mma_lane_m16_n32_splitk16_kernel<half>
              <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK16Threads, 0, stream>>>(
                  reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                  packed_lane_qweight.data_ptr<int32_t>(),
                  reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
                  reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                  static_cast<int>(size_m),
                  static_cast<int>(size_k),
                  static_cast<int>(logical_n),
                  static_cast<int>(num_groups));
        }
      }
    } else if constexpr (SplitKWarps == kMmaLaneSplitK4Warps) {
      amplin_mma_lane_m16_n16_splitk4_kernel<half>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK4Threads, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else if constexpr (SplitKWarps == kMmaLaneSplitK8Warps) {
      amplin_mma_lane_m16_n16_splitk8_kernel<half>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK8Threads, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else if constexpr (SplitKWarps == kMmaLaneSplitK12Warps) {
      amplin_mma_lane_m16_n16_splitk12_kernel<half>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else if constexpr (SplitKWarps == kMmaLaneSplitK16Warps) {
      amplin_mma_lane_m16_n16_splitk16_kernel<half>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK16Threads, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else {
      amplin_mma_lane_m16_n16_padded_kernel<half>
          <<<static_cast<unsigned int>(grid_n), kMmaLanePaddedThreads, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
              reinterpret_cast<half*>(output.data_ptr<at::Half>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    }
  } else {
    if constexpr (OutputN32) {
      if constexpr (SplitKWarps == kMmaLaneSplitK12Warps) {
        if constexpr (PipelineKSteps) {
          if constexpr (InterleavedN32Words) {
            amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_kernel<__nv_bfloat16>
                <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                    packed_lane_qweight.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                    static_cast<int>(size_m),
                    static_cast<int>(size_k),
                    static_cast<int>(logical_n),
                    static_cast<int>(num_groups));
          } else {
            amplin_mma_lane_m16_n32_splitk12_pipe2_kernel<__nv_bfloat16>
                <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                    packed_lane_qweight.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                    static_cast<int>(size_m),
                    static_cast<int>(size_k),
                    static_cast<int>(logical_n),
                    static_cast<int>(num_groups));
          }
        } else {
          amplin_mma_lane_m16_n32_splitk12_kernel<__nv_bfloat16>
              <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                  packed_lane_qweight.data_ptr<int32_t>(),
                  reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                  static_cast<int>(size_m),
                  static_cast<int>(size_k),
                  static_cast<int>(logical_n),
                  static_cast<int>(num_groups));
        }
      } else {
        static_assert(SplitKWarps == kMmaLaneSplitK16Warps);
        if constexpr (PipelineKSteps) {
          amplin_mma_lane_m16_n32_splitk16_pipe2_kernel<__nv_bfloat16>
              <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK16Threads, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                  packed_lane_qweight.data_ptr<int32_t>(),
                  reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                  static_cast<int>(size_m),
                  static_cast<int>(size_k),
                  static_cast<int>(logical_n),
                  static_cast<int>(num_groups));
        } else {
          amplin_mma_lane_m16_n32_splitk16_kernel<__nv_bfloat16>
              <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK16Threads, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                  packed_lane_qweight.data_ptr<int32_t>(),
                  reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
                  reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                  static_cast<int>(size_m),
                  static_cast<int>(size_k),
                  static_cast<int>(logical_n),
                  static_cast<int>(num_groups));
        }
      }
    } else if constexpr (SplitKWarps == kMmaLaneSplitK4Warps) {
      amplin_mma_lane_m16_n16_splitk4_kernel<__nv_bfloat16>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK4Threads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else if constexpr (SplitKWarps == kMmaLaneSplitK8Warps) {
      amplin_mma_lane_m16_n16_splitk8_kernel<__nv_bfloat16>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK8Threads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else if constexpr (SplitKWarps == kMmaLaneSplitK12Warps) {
      amplin_mma_lane_m16_n16_splitk12_kernel<__nv_bfloat16>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK12Threads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else if constexpr (SplitKWarps == kMmaLaneSplitK16Warps) {
      amplin_mma_lane_m16_n16_splitk16_kernel<__nv_bfloat16>
          <<<static_cast<unsigned int>(grid_n), kMmaLaneSplitK16Threads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    } else {
      amplin_mma_lane_m16_n16_padded_kernel<__nv_bfloat16>
          <<<static_cast<unsigned int>(grid_n), kMmaLanePaddedThreads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
              packed_lane_qweight.data_ptr<int32_t>(),
              reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
              reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
              static_cast<int>(size_m),
              static_cast<int>(size_k),
              static_cast<int>(logical_n),
              static_cast<int>(num_groups));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename Scalar>
void configure_amplin_mma_lane_m16_n64_splitk24_dynamic_shared(int device_index) {
  static thread_local int configured_device = -1;
  if (configured_device == device_index) {
    return;
  }
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_kernel<Scalar>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kMmaLaneSplitKN64SharedBytes));
  configured_device = device_index;
}

torch::Tensor amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  TORCH_CHECK(input.is_cuda(), "Amplin N64 split-K24 input must be CUDA");
  TORCH_CHECK(
      packed_lane_qweight.is_cuda() && packed_scales.is_cuda(),
      "Amplin N64 split-K24 weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == packed_lane_qweight.device() &&
          input.device() == packed_scales.device(),
      "Amplin N64 split-K24 tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin N64 split-K24 input must be FP16 or BF16");
  TORCH_CHECK(
      packed_lane_qweight.scalar_type() == at::kInt,
      "Amplin N64 split-K24 qweight must be int32");
  TORCH_CHECK(
      packed_scales.scalar_type() == input.scalar_type(),
      "Amplin N64 split-K24 scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 &&
          packed_lane_qweight.dim() == 5 &&
          packed_scales.dim() == 3,
      "Amplin N64 split-K24 input must have at least two dimensions and packed weights must be 5D/3D");
  TORCH_CHECK(
      input.is_contiguous() &&
          packed_lane_qweight.is_contiguous() &&
          packed_scales.is_contiguous(),
      "Amplin N64 split-K24 tensors must be contiguous");

  const int64_t packed_n_tiles = packed_lane_qweight.size(0);
  const int64_t num_groups = packed_lane_qweight.size(1);
  TORCH_CHECK(
      packed_n_tiles > 0 &&
          num_groups > 0 &&
          packed_lane_qweight.size(2) == kHmmaBlockK / kMmaK &&
          packed_lane_qweight.size(3) == kMmaLanes &&
          packed_lane_qweight.size(4) == kHmmaWarps,
      "Amplin N64 split-K24 qweight must have shape [N/64, K/128, 8, 32, 4]");
  TORCH_CHECK(
      packed_scales.size(0) == packed_n_tiles &&
          packed_scales.size(1) == num_groups &&
          packed_scales.size(2) == kHmmaBlockN,
      "Amplin N64 split-K24 scales must have shape [N/64, K/128, 64]");

  const int64_t size_k = input.size(-1);
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(
      size_k == kWideSizeK &&
          size_k == num_groups * kHmmaBlockK &&
          num_groups % kMmaLaneSplitK24Warps == 0,
      "Amplin N64 split-K24 requires K=12288 and evenly divisible groups");
  TORCH_CHECK(
      size_m == 2 || size_m == 4 || size_m == 8 || size_m == 16,
      "Amplin N64 split-K24 flattened M must be one of 2, 4, 8, or 16");
  TORCH_CHECK(
      logical_n > 0 &&
          logical_n % kHmmaBlockN == 0 &&
          logical_n <= packed_n_tiles * kHmmaBlockN,
      "Amplin N64 split-K24 logical N must be positive, divisible by 64, and fit the packed layout");
  TORCH_CHECK(
      size_k <= std::numeric_limits<int>::max() &&
          logical_n <= std::numeric_limits<int>::max() &&
          num_groups <= std::numeric_limits<int>::max(),
      "Amplin N64 split-K24 tensor dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin N64 split-K24 requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);
  TORCH_CHECK(
      properties->sharedMemPerBlockOptin >= kMmaLaneSplitKN64SharedBytes,
      "Amplin N64 split-K24 requires at least ",
      kMmaLaneSplitKN64SharedBytes,
      " bytes of opt-in shared memory per block");
  const int64_t grid_n = logical_n / kHmmaBlockN;
  TORCH_CHECK(
      grid_n <= properties->maxGridSize[0],
      "Amplin N64 split-K24 grid exceeds the selected CUDA device limit");

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = logical_n;
  auto output = torch::empty(output_sizes, input.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    configure_amplin_mma_lane_m16_n64_splitk24_dynamic_shared<half>(
        input.get_device());
    amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_kernel<half>
        <<<static_cast<unsigned int>(grid_n),
           kMmaLaneSplitK24Threads,
           kMmaLaneSplitKN64SharedBytes,
           stream>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            packed_lane_qweight.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            static_cast<int>(size_m),
            static_cast<int>(size_k),
            static_cast<int>(logical_n),
            static_cast<int>(num_groups));
  } else {
    configure_amplin_mma_lane_m16_n64_splitk24_dynamic_shared<__nv_bfloat16>(
        input.get_device());
    amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_kernel<__nv_bfloat16>
        <<<static_cast<unsigned int>(grid_n),
           kMmaLaneSplitK24Threads,
           kMmaLaneSplitKN64SharedBytes,
           stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            packed_lane_qweight.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<int>(size_m),
            static_cast<int>(size_k),
            static_cast<int>(logical_n),
            static_cast<int>(num_groups));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename Scalar>
void launch_amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved(
    const Scalar* input,
    const int32_t* packed_lane_qweight,
    const Scalar* packed_scales,
    Scalar* output,
    float* scratch,
    int size_m,
    int size_k,
    int size_n,
    int num_groups,
    int sm_count,
    cudaStream_t stream) {
  int active_blocks_per_sm = 0;
  cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_kernel<Scalar>,
      kMmaLaneSplitK12Threads,
      0);
  TORCH_CHECK(
      status == cudaSuccess,
      "Amplin N64 split-K12x2 failed to query cooperative occupancy: ",
      cudaGetErrorString(status));
  const int grid_blocks = (size_n / kHmmaBlockN) * 2;
  TORCH_CHECK(
      grid_blocks <= active_blocks_per_sm * sm_count,
      "Amplin N64 split-K12x2 cooperative grid requires ",
      grid_blocks,
      " resident CTAs but the selected device supports ",
      active_blocks_per_sm * sm_count);

  void* arguments[] = {
      &input,
      &packed_lane_qweight,
      &packed_scales,
      &output,
      &scratch,
      &size_m,
      &size_k,
      &size_n,
      &num_groups,
  };
  status = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(
          amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_kernel<Scalar>),
      dim3(static_cast<unsigned int>(size_n / kHmmaBlockN), 2, 1),
      dim3(kMmaLaneSplitK12Threads),
      arguments,
      0,
      stream);
  TORCH_CHECK(
      status == cudaSuccess,
      "Amplin N64 split-K12x2 cooperative launch failed: ",
      cudaGetErrorString(status));
}

torch::Tensor amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  TORCH_CHECK(input.is_cuda(), "Amplin N64 split-K12x2 input must be CUDA");
  TORCH_CHECK(
      packed_lane_qweight.is_cuda() && packed_scales.is_cuda(),
      "Amplin N64 split-K12x2 weight tensors must be CUDA");
  TORCH_CHECK(
      input.device() == packed_lane_qweight.device() &&
          input.device() == packed_scales.device(),
      "Amplin N64 split-K12x2 tensors must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "Amplin N64 split-K12x2 input must be FP16 or BF16");
  TORCH_CHECK(
      packed_lane_qweight.scalar_type() == at::kInt,
      "Amplin N64 split-K12x2 qweight must be int32");
  TORCH_CHECK(
      packed_scales.scalar_type() == input.scalar_type(),
      "Amplin N64 split-K12x2 scales dtype must match input dtype");
  TORCH_CHECK(
      input.dim() >= 2 &&
          packed_lane_qweight.dim() == 5 &&
          packed_scales.dim() == 3,
      "Amplin N64 split-K12x2 input must have at least two dimensions and packed weights must be 5D/3D");
  TORCH_CHECK(
      input.is_contiguous() &&
          packed_lane_qweight.is_contiguous() &&
          packed_scales.is_contiguous(),
      "Amplin N64 split-K12x2 tensors must be contiguous");

  const int64_t packed_n_tiles = packed_lane_qweight.size(0);
  const int64_t num_groups = packed_lane_qweight.size(1);
  TORCH_CHECK(
      packed_n_tiles > 0 &&
          num_groups > 0 &&
          packed_lane_qweight.size(2) == kHmmaBlockK / kMmaK &&
          packed_lane_qweight.size(3) == kMmaLanes &&
          packed_lane_qweight.size(4) == kHmmaWarps,
      "Amplin N64 split-K12x2 qweight must have shape [N/64, K/128, 8, 32, 4]");
  TORCH_CHECK(
      packed_scales.size(0) == packed_n_tiles &&
          packed_scales.size(1) == num_groups &&
          packed_scales.size(2) == kHmmaBlockN,
      "Amplin N64 split-K12x2 scales must have shape [N/64, K/128, 64]");

  const int64_t size_k = input.size(-1);
  const int64_t size_m = input.numel() / size_k;
  TORCH_CHECK(
      size_k == kWideSizeK &&
          size_k == num_groups * kHmmaBlockK &&
          num_groups % kMmaLaneSplitK24Warps == 0,
      "Amplin N64 split-K12x2 requires K=12288 and evenly divisible groups");
  TORCH_CHECK(
      size_m == 2 || size_m == 4 || size_m == 8 || size_m == 16,
      "Amplin N64 split-K12x2 flattened M must be one of 2, 4, 8, or 16");
  TORCH_CHECK(
      logical_n > 0 &&
          logical_n % kHmmaBlockN == 0 &&
          logical_n <= packed_n_tiles * kHmmaBlockN,
      "Amplin N64 split-K12x2 logical N must be positive, divisible by 64, and fit the packed layout");
  TORCH_CHECK(
      size_m <= std::numeric_limits<int>::max() &&
          size_k <= std::numeric_limits<int>::max() &&
          logical_n <= std::numeric_limits<int>::max() &&
          num_groups <= std::numeric_limits<int>::max(),
      "Amplin N64 split-K12x2 tensor dimensions exceed int32 kernel indexing limits");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 0,
      "Amplin N64 split-K12x2 requires CUDA compute capability 8.0, got ",
      properties->major,
      ".",
      properties->minor);
  TORCH_CHECK(
      properties->cooperativeLaunch != 0,
      "Amplin N64 split-K12x2 requires cooperative CUDA launch support");
  TORCH_CHECK(
      properties->sharedMemPerBlock >= kMmaLaneSplitKN64K12SharedBytes,
      "Amplin N64 split-K12x2 requires at least ",
      kMmaLaneSplitKN64K12SharedBytes,
      " bytes of shared memory per block");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaError_t status = cudaStreamIsCapturing(stream, &capture_status);
  TORCH_CHECK(
      status == cudaSuccess,
      "Amplin N64 split-K12x2 failed to query CUDA graph capture state: ",
      cudaGetErrorString(status));
  TORCH_CHECK(
      capture_status == cudaStreamCaptureStatusNone,
      "Amplin N64 split-K12x2 cooperative launch does not support CUDA graph capture");

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = logical_n;
  auto output = torch::empty(output_sizes, input.options());
  auto scratch = torch::empty(
      {2, size_m, logical_n},
      input.options().dtype(at::kFloat));

  if (input.scalar_type() == at::kHalf) {
    launch_amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        packed_lane_qweight.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(packed_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        scratch.data_ptr<float>(),
        static_cast<int>(size_m),
        static_cast<int>(size_k),
        static_cast<int>(logical_n),
        static_cast<int>(num_groups),
        properties->multiProcessorCount,
        stream);
  } else {
    launch_amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        packed_lane_qweight.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(packed_scales.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        scratch.data_ptr<float>(),
        static_cast<int>(size_m),
        static_cast<int>(size_k),
        static_cast<int>(logical_n),
        static_cast<int>(num_groups),
        properties->multiProcessorCount,
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor amplin_mma_lane_m16_n16_padded_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<0>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk4_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<kMmaLaneSplitK4Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk8_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<kMmaLaneSplitK8Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk12_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<kMmaLaneSplitK12Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk12_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<kMmaLaneSplitK12Warps, true>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk16_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<kMmaLaneSplitK16Warps, true>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk12_pipe2_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<
      kMmaLaneSplitK12Warps,
      true,
      true>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<
      kMmaLaneSplitK12Warps,
      true,
      true,
      true>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk16_pipe2_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<
      kMmaLaneSplitK16Warps,
      true,
      true>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk16_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_cuda_impl<kMmaLaneSplitK16Warps>(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}
