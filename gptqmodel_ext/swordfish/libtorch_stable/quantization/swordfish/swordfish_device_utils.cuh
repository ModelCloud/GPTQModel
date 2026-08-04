// SPDX-FileCopyrightText: 2026 AlpinDale and the dphnAI/sonar contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Swordfish: Blackwell (sm100/sm110) weight-quantized GEMM kernels.
// Host-only helpers for querying runtime GPU properties without assuming
// a fixed CUDA device index.

#pragma once

#ifndef __CUDA_ARCH__

#include <cuda_runtime.h>
#include <functional>
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

struct OccupancyCacheKey {
  const void* kernel;
  int device_index;
  int block_size;

  bool operator==(const OccupancyCacheKey& other) const {
    return kernel == other.kernel && device_index == other.device_index &&
           block_size == other.block_size;
  }
};

struct OccupancyCacheKeyHash {
  std::size_t operator()(const OccupancyCacheKey& k) const noexcept {
    std::size_t h = std::hash<const void*>{}(k.kernel);
    h ^= std::hash<int>{}(k.device_index) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.block_size) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

// Return the per-SM active-block count for `kernel` on `device_index` and
// `block_size`, caching per (kernel, device, block_size).  The caller must set
// the CUDA device guard when device_index == -1.
template <typename Kernel>
inline int cached_occupancy_for_device(int device_index, Kernel kernel,
                                       int block_size, int fallback) {
  if (device_index < 0) {
    cudaGetDevice(&device_index);
  }

  static std::mutex mutex;
  static std::unordered_map<OccupancyCacheKey, int, OccupancyCacheKeyHash> cache;
  std::lock_guard<std::mutex> lock(mutex);

  OccupancyCacheKey key{
      reinterpret_cast<const void*>(kernel), device_index, block_size};
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  int ctas = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &ctas, kernel, block_size, 0);
  if (ctas <= 0) ctas = fallback;
  cache[key] = ctas;
  return ctas;
}

}  // namespace swordfish

#endif  // __CUDA_ARCH__
