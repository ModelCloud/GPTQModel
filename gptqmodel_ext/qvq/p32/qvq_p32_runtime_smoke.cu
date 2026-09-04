// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

bool close_enough(float actual, float expected) {
  const float tolerance = 2.0e-4f + 2.0e-3f * std::fabs(expected);
  return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
  if (qvq_p32_abi_version() != QVQ_P32_ABI_VERSION ||
      qvq_p32_kernel_version() != QVQ_P32_KERNEL_VERSION ||
      qvq_compiled_sm() != QVQ_P32_COMPILED_SM) {
    std::fprintf(stderr, "QVQ P32 runtime contract mismatch\n");
    return 1;
  }

  int device = 0;
  if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice") ||
      !qvq_device_is_supported(device)) {
    std::fprintf(stderr, "QVQ P32 runtime smoke requires an SM80 device\n");
    return 1;
  }

  constexpr int kM = 1;
  constexpr int kK = 32;
  constexpr int kN = 16;
  constexpr int kTransitionBits = 4;
  constexpr int kSplitCount = 2;
  constexpr int kWordsPerTile = 4 * kTransitionBits;
  constexpr int kTileCount =
      (kK / QVQ_P32_TILE_SIZE) * (kN / QVQ_P32_TILE_SIZE);

  std::vector<half> input(kM * kK);
  for (int index = 0; index < input.size(); ++index) {
    input[index] = __float2half((static_cast<float>(index % 11) - 5.0f) / 16.0f);
  }
  std::vector<uint32_t> trellis(kTileCount * kWordsPerTile);
  for (int index = 0; index < trellis.size(); ++index) {
    trellis[index] = 0x13579bdfu ^ (0x9e3779b9u * static_cast<uint32_t>(index + 1));
  }
  std::vector<half> levels(QVQ_P32_LEVEL_COUNT);
  for (int index = 0; index < levels.size(); ++index) {
    levels[index] = __float2half((static_cast<float>(index) - 127.5f) / 128.0f);
  }
  const std::vector<uint8_t> bank_ids = {0xa5u, 0x5au};
  const std::vector<uint8_t> bank_alt_id = {3u};

  void* device_input = nullptr;
  void* device_trellis = nullptr;
  void* device_levels = nullptr;
  void* device_bank_ids = nullptr;
  void* device_bank_alt_id = nullptr;
  float* device_native_output = nullptr;
  float* device_partials_output = nullptr;
  float* device_native_workspace = nullptr;
  float* device_partials_workspace = nullptr;
  cudaStream_t stream = nullptr;

  const size_t output_bytes = kM * kN * sizeof(float);
  const size_t workspace_bytes = kSplitCount * output_bytes;
  bool ok = check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate") &&
      check_cuda(cudaMalloc(&device_input, input.size() * sizeof(half)),
                 "cudaMalloc(input)") &&
      check_cuda(cudaMalloc(&device_trellis,
                            trellis.size() * sizeof(uint32_t)),
                 "cudaMalloc(trellis)") &&
      check_cuda(cudaMalloc(&device_levels, levels.size() * sizeof(half)),
                 "cudaMalloc(levels)") &&
      check_cuda(cudaMalloc(&device_bank_ids, bank_ids.size()),
                 "cudaMalloc(bank_ids)") &&
      check_cuda(cudaMalloc(&device_bank_alt_id, bank_alt_id.size()),
                 "cudaMalloc(bank_alt_id)") &&
      check_cuda(cudaMalloc(&device_native_output, output_bytes),
                 "cudaMalloc(native_output)") &&
      check_cuda(cudaMalloc(&device_partials_output, output_bytes),
                 "cudaMalloc(partials_output)") &&
      check_cuda(cudaMalloc(&device_native_workspace, workspace_bytes),
                 "cudaMalloc(native_workspace)") &&
      check_cuda(cudaMalloc(&device_partials_workspace, workspace_bytes),
                 "cudaMalloc(partials_workspace)");

  if (ok) {
    ok = check_cuda(cudaMemcpyAsync(device_input, input.data(),
                                    input.size() * sizeof(half),
                                    cudaMemcpyHostToDevice, stream),
                    "copy input") &&
        check_cuda(cudaMemcpyAsync(device_trellis, trellis.data(),
                                   trellis.size() * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "copy trellis") &&
        check_cuda(cudaMemcpyAsync(device_levels, levels.data(),
                                   levels.size() * sizeof(half),
                                   cudaMemcpyHostToDevice, stream),
                   "copy levels") &&
        check_cuda(cudaMemcpyAsync(device_bank_ids, bank_ids.data(),
                                   bank_ids.size(), cudaMemcpyHostToDevice,
                                   stream),
                   "copy bank_ids") &&
        check_cuda(cudaMemcpyAsync(device_bank_alt_id, bank_alt_id.data(),
                                   bank_alt_id.size(), cudaMemcpyHostToDevice,
                                   stream),
                   "copy bank_alt_id");
  }

  if (ok && qvq_p32_window(
                device_input, device_trellis, device_levels, device_bank_ids,
                device_bank_alt_id, device_native_output,
                device_native_workspace, kM, kK, kN, kTransitionBits,
                kSplitCount, QVQ_P32_VARIANT_SCALAR, 128, 2, 0,
                QVQ_P32_REDUCTION_NATIVE, stream) != 0) {
    std::fprintf(stderr, "native reduction failed: %s\n", qvq_last_error());
    ok = false;
  }
  if (ok && qvq_p32_window(
                device_input, device_trellis, device_levels, device_bank_ids,
                device_bank_alt_id, device_partials_output,
                device_partials_workspace, kM, kK, kN, kTransitionBits,
                kSplitCount, QVQ_P32_VARIANT_SCALAR, 128, 2, 0,
                QVQ_P32_REDUCTION_PARTIALS, stream) != 0) {
    std::fprintf(stderr, "partials reduction failed: %s\n", qvq_last_error());
    ok = false;
  }

  std::vector<float> native_output(kM * kN);
  std::vector<float> partials(kSplitCount * kM * kN);
  if (ok) {
    ok = check_cuda(cudaMemcpyAsync(native_output.data(), device_native_output,
                                    output_bytes, cudaMemcpyDeviceToHost,
                                    stream),
                    "copy native output") &&
        check_cuda(cudaMemcpyAsync(partials.data(), device_partials_workspace,
                                   workspace_bytes, cudaMemcpyDeviceToHost,
                                   stream),
                   "copy partials") &&
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

  if (ok) {
    for (int column = 0; column < kN; ++column) {
      const float expected = partials[column] + partials[kN + column];
      if (!close_enough(native_output[column], expected)) {
        std::fprintf(stderr,
                     "reduction mismatch at N=%d: native=%g partials=%g\n",
                     column, native_output[column], expected);
        ok = false;
        break;
      }
    }
  }

  if (stream != nullptr) cudaStreamSynchronize(stream);
  cudaFree(device_partials_workspace);
  cudaFree(device_native_workspace);
  cudaFree(device_partials_output);
  cudaFree(device_native_output);
  cudaFree(device_bank_alt_id);
  cudaFree(device_bank_ids);
  cudaFree(device_levels);
  cudaFree(device_trellis);
  cudaFree(device_input);
  if (stream != nullptr) cudaStreamDestroy(stream);

  if (!ok) return 1;
  std::printf("qvq_p32_runtime_smoke=PASS sm=%d split=%d reduction=native+partials\n",
              qvq_device_sm(device), kSplitCount);
  return 0;
}
