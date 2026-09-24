// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_hadamard_input_raw_abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace {

constexpr int kWidth = 8192;
constexpr int kTile = 256;
constexpr int kTiles = kWidth / kTile;
constexpr int kHighThreads = 64;

__device__ __forceinline__ int padded_index(int index) {
  return index + (index >> 5);
}

// This is the FP16 QVQ input contract: round the SU product, divide its
// FP32 value by FP16(sqrt(8192)), round again, then round each ascending
// butterfly sum/difference. The first eight stages are tile-local.
__global__ void hadamard_input_low(
    const half* __restrict__ input,
    const half* __restrict__ scale,
    half* __restrict__ workspace) {
  __shared__ half tile[kTile + kTile / 32];
  const int row = static_cast<int>(blockIdx.y);
  const int local = static_cast<int>(threadIdx.x);
  const int column = static_cast<int>(blockIdx.x) * kTile + local;
  const int64_t offset = static_cast<int64_t>(row) * kWidth + column;
  const float divisor = __half2float(__float2half_rn(sqrtf(float(kWidth))));
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

// The remaining five stages combine the 32 independent 256-column tiles.
// One thread owns one intra-tile column across all tiles, so intermediate
// FP16-rounded values stay in registers until the final store.
__global__ void hadamard_input_high(
    const half* workspace,
    half* output) {
  const int row = static_cast<int>(blockIdx.y);
  const int local = static_cast<int>(blockIdx.x) * kHighThreads +
      static_cast<int>(threadIdx.x);
  half values[kTiles];
#pragma unroll
  for (int tile = 0; tile < kTiles; ++tile) {
    values[tile] = workspace[static_cast<int64_t>(row) * kWidth +
                             tile * kTile + local];
  }
#pragma unroll
  for (int bit = 1; bit < kTiles; bit <<= 1) {
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
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
  for (int tile = 0; tile < kTiles; ++tile) {
    output[static_cast<int64_t>(row) * kWidth + tile * kTile + local] =
        values[tile];
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
      config->rows > 0 && config->rows <= 960 && config->width == kWidth;
}

}  // namespace

extern "C" {

uint32_t qvq_hadamard_input_raw_abi_version(void) {
  return QVQ_HADAMARD_INPUT_RAW_ABI_VERSION;
}

uint64_t qvq_hadamard_input_raw_workspace_bytes(
    const QvqHadamardInputRawConfig* config) {
  return valid(config)
      ? static_cast<uint64_t>(config->rows) * kWidth * sizeof(half)
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
  hadamard_input_low<<<dim3(kTiles, config->rows), kTile, 0, stream>>>(
      static_cast<const half*>(input_f16),
      static_cast<const half*>(pre_scale_f16),
      static_cast<half*>(workspace_f16));
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return fail(error, error_capacity, cudaGetErrorString(status));
  }
  hadamard_input_high<<<dim3(kTile / kHighThreads, config->rows),
                        kHighThreads, 0, stream>>>(
      static_cast<const half*>(workspace_f16),
      static_cast<half*>(output_f16));
  status = cudaGetLastError();
  return status == cudaSuccess
      ? 0 : fail(error, error_capacity, cudaGetErrorString(status));
}

}  // extern "C"
