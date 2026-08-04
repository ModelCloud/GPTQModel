// SPDX-FileCopyrightText: 2026 AlpinDale and the dphnAI/sonar contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Swordfish: Blackwell (sm100/sm110) weight-quantized GEMM kernels.
// Host-only helpers for querying runtime GPU properties without assuming
// a fixed CUDA device index.

#pragma once

#ifndef __CUDA_ARCH__

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>

namespace swordfish {

// Return the SM count for the requested CUDA device, caching the result
// per device.  If device_index is -1 the current device is used.
inline int cached_device_sm_count(int device_index = -1) {
  if (device_index < 0) {
    cudaGetDevice(&device_index);
  }

  static std::mutex mutex;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lock(mutex);

  auto it = cache.find(device_index);
  if (it != cache.end()) {
    return it->second;
  }

  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_index);
  if (sms <= 0) sms = 1;
  cache[device_index] = sms;
  return sms;
}

// Return the per-SM active-block count for `kernel` on `device_index`,
// caching per device.  The caller must ensure the device guard is set if
// device_index == -1 (current device).
template <typename Kernel>
inline int cached_occupancy_for_device(int device_index, Kernel kernel,
                                       int block_size, int fallback) {
  if (device_index < 0) {
    cudaGetDevice(&device_index);
  }

  static std::mutex mutex;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lock(mutex);

  auto it = cache.find(device_index);
  if (it != cache.end()) {
    return it->second;
  }

  int ctas = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &ctas, kernel, block_size, 0);
  if (ctas <= 0) ctas = fallback;
  cache[device_index] = ctas;
  return ctas;
}

}  // namespace swordfish

#endif  // __CUDA_ARCH__
