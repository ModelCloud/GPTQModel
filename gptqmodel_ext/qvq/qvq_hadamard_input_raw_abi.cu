// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_hadamard_input_raw_abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace {

constexpr int kTile = 256;
constexpr int kHighThreads = 64;

__device__ __forceinline__ int padded_index(int index) {
  return index + (index >> 5);
}

// This is the FP16 QVQ input contract: round the SU product, divide its
// FP32 value by FP16(sqrt(width)), round again, then round each ascending
// butterfly sum/difference. The first eight stages are tile-local.
template <int Width>
__global__ void hadamard_input_low(
    const half* __restrict__ input,
    const half* __restrict__ scale,
    half* __restrict__ workspace) {
  __shared__ half tile[kTile + kTile / 32];
  const int row = static_cast<int>(blockIdx.y);
  const int local = static_cast<int>(threadIdx.x);
  const int column = static_cast<int>(blockIdx.x) * kTile + local;
  const int64_t offset = static_cast<int64_t>(row) * Width + column;
  const float divisor = __half2float(__float2half_rn(sqrtf(float(Width))));
  const half scaled = __float2half_rn(
      __half2float(input[offset]) * __half2float(scale[column]));
  tile[padded_index(local)] =
      __float2half_rn(__half2float(scaled) / divisor);
  __syncthreads();

#pragma unroll
  for (int bit = 1; bit < kTile; bit <<= 1) {
    const int peer = local ^ bit;
    if (local < peer) {
      const float a = __half2float(tile[padded_index(local)]);
      const float b = __half2float(tile[padded_index(peer)]);
      tile[padded_index(local)] = __float2half_rn(a + b);
      tile[padded_index(peer)] = __float2half_rn(a - b);
    }
    __syncthreads();
  }
  workspace[offset] = tile[padded_index(local)];
}

// The remaining three/five stages combine the 8/32 independent tiles.
// One thread owns one intra-tile column across all tiles, so intermediate
// FP16-rounded values stay in registers until the final store.
template <int Width>
__global__ void hadamard_input_high(
    const half* workspace,
    half* output) {
  constexpr int Tiles = Width / kTile;
  const int row = static_cast<int>(blockIdx.y);
  const int local = static_cast<int>(blockIdx.x) * kHighThreads +
      static_cast<int>(threadIdx.x);
  half values[Tiles];
#pragma unroll
  for (int tile = 0; tile < Tiles; ++tile) {
    values[tile] = workspace[static_cast<int64_t>(row) * Width +
                             tile * kTile + local];
  }
#pragma unroll
  for (int bit = 1; bit < Tiles; bit <<= 1) {
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
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
  for (int tile = 0; tile < Tiles; ++tile) {
    output[static_cast<int64_t>(row) * Width + tile * kTile + local] =
        values[tile];
  }
}

// A full-row CTA can keep every FP16-rounded stage inside registers/shared
// memory, avoiding the global intermediate and second launch.
__device__ __forceinline__ half2 exact_half2_add(half2 first, half2 second) {
  const float2 a = __half22float2(first);
  const float2 b = __half22float2(second);
  return __floats2half2_rn(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ half2 exact_half2_sub(half2 first, half2 second) {
  const float2 a = __half22float2(first);
  const float2 b = __half22float2(second);
  return __floats2half2_rn(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ half swiglu_value(half gate, half up) {
  const float g = __half2float(gate);
  // StableHLO logistic(f16) lowers to f16 exp, f16 add, f16 divide.
  // Keep those materialization points even though this kernel fuses the
  // subsequent SwiGLU multiply with SU/Hadamard preparation.
  const half exponential = __float2half_rn(expf(-g));
  const half denominator = __float2half_rn(1.0f + __half2float(exponential));
  const half sigmoid = __float2half_rn(1.0f / __half2float(denominator));
  const half silu = __float2half_rn(g * __half2float(sigmoid));
  return __float2half_rn(__half2float(silu) * __half2float(up));
}

template <int Width, int Threads, bool SwiGlu = false>
__global__ void hadamard_input_single(
    const half* __restrict__ input,
    const half* __restrict__ up,
    const half* __restrict__ scale,
    half* __restrict__ output) {
  constexpr int Pairs = Width / 2;
  static_assert(Width == 2048 || Width == 8192);
  static_assert(Threads >= 128 && Threads <= 1024 && Threads % 32 == 0);
  __shared__ half2 tile[Pairs];
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const float divisor = __half2float(__float2half_rn(sqrtf(float(Width))));
#pragma unroll
  for (int item = 0; item < Pairs / Threads; ++item) {
    const int pair = lane + item * Threads;
    const int column0 = 2 * pair;
    const int64_t offset = static_cast<int64_t>(row) * Width + column0;
    half2 activation;
    if constexpr (SwiGlu) {
      const half2 gate_pair = *reinterpret_cast<const half2*>(input + offset);
      const half2 up_pair = *reinterpret_cast<const half2*>(up + offset);
      activation = __halves2half2(
          swiglu_value(__low2half(gate_pair), __low2half(up_pair)),
          swiglu_value(__high2half(gate_pair), __high2half(up_pair)));
    } else {
      activation = *reinterpret_cast<const half2*>(input + offset);
    }
    const half2 scaled = __hmul2(
        activation, *reinterpret_cast<const half2*>(scale + column0));
    const float2 scaled_values = __half22float2(scaled);
    const half value0 = __float2half_rn(scaled_values.x / divisor);
    const half value1 = __float2half_rn(scaled_values.y / divisor);
    half2 packed = __floats2half2_rn(
        __half2float(value0) + __half2float(value1),
        __half2float(value0) - __half2float(value1));
    // Pair-index bits 0..4 stay inside the warp. Every shuffle uses the
    // values after the previous FP16 rounding boundary.
#pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
      union Half2Bits {
        half2 value;
        unsigned bits;
      } self, peer;
      self.value = packed;
      peer.bits = __shfl_xor_sync(0xffffffffu, self.bits, bit);
      packed = (pair & bit) == 0
          ? exact_half2_add(packed, peer.value)
          : exact_half2_sub(peer.value, packed);
    }
    tile[pair] = packed;
  }
  __syncthreads();

  constexpr int SharedEnd = Width == 8192 ? Pairs / 8 : Pairs;
#pragma unroll
  for (int bit = 32; bit < SharedEnd; bit <<= 1) {
#pragma unroll
    for (int butterfly = lane; butterfly < Pairs / 2;
         butterfly += Threads) {
      const int pair = (butterfly & (bit - 1)) +
          ((butterfly & ~(bit - 1)) << 1);
      const int peer = pair + bit;
      const half2 first = tile[pair];
      const half2 second = tile[peer];
      tile[pair] = exact_half2_add(first, second);
      tile[peer] = exact_half2_sub(first, second);
    }
    __syncthreads();
  }

  if constexpr (Width == 8192) {
    // Each thread owns an eight-pair group for the final 512/1024/2048
    // pair-index butterflies. Keep the same ascending-stage FP16 rounds,
    // with no shared-memory handoff between these independently owned groups.
#pragma unroll
    for (int base = lane; base < Pairs / 8; base += Threads) {
      half2 values[8];
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        values[index] = tile[base + index * Pairs / 8];
      }
#pragma unroll
      for (int bit = 1; bit < 8; bit <<= 1) {
#pragma unroll
        for (int group = 0; group < 8; group += 2 * bit) {
#pragma unroll
          for (int offset = 0; offset < bit; ++offset) {
            const half2 first = values[group + offset];
            const half2 second = values[group + offset + bit];
            values[group + offset] = exact_half2_add(first, second);
            values[group + offset + bit] = exact_half2_sub(first, second);
          }
        }
      }
      const int64_t row_offset = static_cast<int64_t>(row) * Width;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        *reinterpret_cast<half2*>(
            output + row_offset + 2 * (base + index * Pairs / 8)) = values[index];
      }
    }
  } else {
#pragma unroll
    for (int item = 0; item < Pairs / Threads; ++item) {
      const int pair = lane + item * Threads;
      *reinterpret_cast<half2*>(
          output + static_cast<int64_t>(row) * Width + 2 * pair) = tile[pair];
    }
  }
}

int fail(char* error, uint64_t capacity, const char* message) {
  if (error != nullptr && capacity > 0) {
    std::snprintf(error, static_cast<size_t>(capacity), "%s", message);
  }
  return 1;
}

bool valid(const QvqHadamardInputRawConfig* config) {
  return config != nullptr &&
      config->abi_version == QVQ_HADAMARD_INPUT_RAW_ABI_VERSION &&
      config->struct_bytes == sizeof(QvqHadamardInputRawConfig) &&
      config->rows > 0 && config->rows <= 960 &&
      (config->width == 2048 || config->width == 8192);
}

}  // namespace

extern "C" {

uint32_t qvq_hadamard_input_raw_abi_version(void) {
  return QVQ_HADAMARD_INPUT_RAW_ABI_VERSION;
}

uint32_t qvq_hadamard_input_swiglu_raw_abi_version(void) {
  return QVQ_HADAMARD_INPUT_SWIGLU_RAW_ABI_VERSION;
}

uint64_t qvq_hadamard_input_raw_workspace_bytes(
    const QvqHadamardInputRawConfig* config) {
  return valid(config)
      ? static_cast<uint64_t>(config->rows) * config->width * sizeof(half)
      : 0;
}

int qvq_hadamard_input_raw_launch(
    const void* input_f16,
    const void* pre_scale_f16,
    void* output_f16,
    void* workspace_f16,
    uint64_t workspace_bytes,
    const QvqHadamardInputRawConfig* config,
    void* cuda_stream,
    char* error,
    uint64_t error_capacity) {
  if (!valid(config)) {
    return fail(error, error_capacity, "unsupported hadamard input geometry or ABI");
  }
  if (input_f16 == nullptr || pre_scale_f16 == nullptr || output_f16 == nullptr ||
      workspace_f16 == nullptr || cuda_stream == nullptr ||
      workspace_bytes < qvq_hadamard_input_raw_workspace_bytes(config)) {
    return fail(error, error_capacity, "missing or undersized hadamard input buffer");
  }
  auto stream = static_cast<cudaStream_t>(cuda_stream);
  if (config->width == 2048) {
    hadamard_input_single<2048, 256><<<config->rows, kTile, 0, stream>>>(
        static_cast<const half*>(input_f16),
        nullptr,
        static_cast<const half*>(pre_scale_f16),
        static_cast<half*>(output_f16));
  } else if (config->rows >= 256) {
    hadamard_input_single<8192, 512><<<config->rows, 512, 0, stream>>>(
        static_cast<const half*>(input_f16),
        nullptr,
        static_cast<const half*>(pre_scale_f16),
        static_cast<half*>(output_f16));
  } else if (config->rows >= 64) {
    hadamard_input_single<8192, 1024><<<config->rows, 1024, 0, stream>>>(
        static_cast<const half*>(input_f16),
        nullptr,
        static_cast<const half*>(pre_scale_f16),
        static_cast<half*>(output_f16));
  } else {
    // A full-row CTA underfills the H100 for tiny M. Retain the exact
    // two-pass tile path, including its caller-owned workspace contract.
    hadamard_input_low<8192><<<dim3(8192 / kTile, config->rows),
                                kTile, 0, stream>>>(
        static_cast<const half*>(input_f16),
        static_cast<const half*>(pre_scale_f16),
        static_cast<half*>(workspace_f16));
  }
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return fail(error, error_capacity, cudaGetErrorString(status));
  }
  if (config->width == 8192 && config->rows < 64) {
    hadamard_input_high<8192><<<dim3(kTile / kHighThreads, config->rows),
                                 kHighThreads, 0, stream>>>(
        static_cast<const half*>(workspace_f16),
        static_cast<half*>(output_f16));
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return fail(error, error_capacity, cudaGetErrorString(status));
    }
  }
  return 0;
}

int qvq_hadamard_input_swiglu_raw_launch(
    const void* gate_f16,
    const void* up_f16,
    const void* pre_scale_f16,
    void* output_f16,
    const QvqHadamardInputRawConfig* config,
    void* cuda_stream,
    char* error,
    uint64_t error_capacity) {
  if (!valid(config) || config->rows != 960 || config->width != 8192) {
    return fail(error, error_capacity, "unsupported SwiGLU Hadamard geometry");
  }
  if (gate_f16 == nullptr || up_f16 == nullptr || pre_scale_f16 == nullptr ||
      output_f16 == nullptr || cuda_stream == nullptr) {
    return fail(error, error_capacity, "missing SwiGLU Hadamard input");
  }
  hadamard_input_single<8192, 512, true><<<config->rows, 512, 0,
                                            static_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const half*>(gate_f16),
      static_cast<const half*>(up_f16),
      static_cast<const half*>(pre_scale_f16),
      static_cast<half*>(output_f16));
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess
      ? 0 : fail(error, error_capacity, cudaGetErrorString(status));
}

}  // extern "C"
