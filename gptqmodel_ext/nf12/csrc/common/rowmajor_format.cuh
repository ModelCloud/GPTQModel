// SPDX-License-Identifier: Apache-2.0
// Portable row-major NF12 format helpers shared by the CUDA and HIP kernels.
// Only integer metadata, escape, and pair-decode logic lives here; launch
// geometry, cache hints, and wave intrinsics stay in the backend files.
#pragma once

#include <cstdint>

namespace nf12_rowmajor {

struct Pair {
  uint32_t lo;
  uint32_t hi;
};

__device__ __forceinline__ Pair decode_pair(
    uint16_t meta, uint32_t lows, const uint64_t* __restrict__ escapes,
    int64_t num_escapes, uint32_t base) {
  if (meta == 0u) {
    if (lows < static_cast<uint64_t>(num_escapes)) {
      const uint64_t value = __ldg(escapes + lows);
      return Pair{static_cast<uint32_t>(value),
                  static_cast<uint32_t>(value >> 32)};
    }
    return Pair{0u, 0u};
  }
  const uint32_t m0 = meta & 0xffu;
  const uint32_t m2 = meta >> 8;
  const uint32_t h0 = ((m0 & 0x87u) + base) & 0xffu;
  const uint32_t h1 = ((((m0 << 1) & 0x80u) | ((m0 >> 3) & 7u)) + base) & 0xffu;
  const uint32_t h2 = ((m2 & 0x87u) + base) & 0xffu;
  const uint32_t h3 = ((((m2 << 1) & 0x80u) | ((m2 >> 3) & 7u)) + base) & 0xffu;
  const uint32_t lo = ((h0 << 8) | (lows & 0xffu)) |
                      (((h1 << 8) | ((lows >> 8) & 0xffu)) << 16);
  const uint32_t hi = ((h2 << 8) | ((lows >> 16) & 0xffu)) |
                      (((h3 << 8) | ((lows >> 24) & 0xffu)) << 16);
  return Pair{lo, hi};
}

__device__ __forceinline__ int collision_count(uint32_t meta) {
  if (meta == 0u) return 0;
  const bool lower = (meta & 0x00ff00ffu) == 0u;
  const bool upper = (meta & 0xff00ff00u) == 0u;
  return static_cast<int>(lower) + static_cast<int>(upper);
}

__device__ __forceinline__ uint16_t selected_meta(uint32_t meta, int upper) {
  return static_cast<uint16_t>(
      ((meta >> (8 * upper)) & 0xffu) |
      (((meta >> (16 + 8 * upper)) & 0xffu) << 8));
}

__device__ __forceinline__ uint32_t selected_lows(uint32_t lo0, uint32_t lo1,
                                                  int upper) {
  return ((lo0 >> (16 * upper)) & 0xffffu) |
         (((lo1 >> (16 * upper)) & 0xffffu) << 16);
}

__device__ __forceinline__ uint64_t collision_escape(uint32_t lows,
                                                     uint32_t base) {
  const uint64_t high = static_cast<uint64_t>(base) << 8;
  return (high | (lows & 0xffu)) |
         ((high | ((lows >> 8) & 0xffu)) << 16) |
         ((high | ((lows >> 16) & 0xffu)) << 32) |
         ((high | ((lows >> 24) & 0xffu)) << 48);
}

}  // namespace nf12_rowmajor
