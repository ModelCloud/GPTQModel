// SPDX-License-Identifier: Apache-2.0
// Shared canonical -> row-major NF12 packing for the CUDA and HIP backends.
#pragma once

#include <cstdint>

#include "rowmajor_format.cuh"

#ifndef NF12_ROWMAJOR_SHFL_UP
#define NF12_ROWMAJOR_SHFL_UP(value, offset) \
  __shfl_up_sync(0xffffffffu, value, offset)
#endif

namespace nf12_rowmajor {

constexpr int kRowBlockBytes = 96;
constexpr int kRowMetaBytes = 32;
constexpr int kPackThreads = 128;

__global__ void count_collisions(const uint8_t* __restrict__ source,
                                 int32_t* __restrict__ counts) {
  const int64_t tile = blockIdx.x;
  const int tid = threadIdx.x;
  const uint32_t meta =
      reinterpret_cast<const uint32_t*>(source + tile * 1536)[tid];
  const int count = collision_count(meta);
  __shared__ int partial[kPackThreads];
  partial[tid] = count;
  __syncthreads();
  if (tid == 0) {
    int total = 0;
    for (int i = 0; i < kPackThreads; ++i) total += partial[i];
    counts[tile] = total;
  }
}

__global__ void pack(const uint8_t* __restrict__ source,
                     int64_t original_escapes, int64_t k_blocks,
                     uint32_t base, const int64_t* __restrict__ prefix,
                     uint8_t* __restrict__ dest,
                     uint64_t* __restrict__ dest_escapes,
                     int32_t* __restrict__ invalid_escape) {
  const int64_t tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid >> 2;
  const int atom = tid & 3;
  const uint8_t* src = source + tile * 1536;
  const uint32_t meta = reinterpret_cast<const uint32_t*>(src)[tid];
  const int plane = atom < 2 ? 1 : 2;
  const int lo_index = lane * 4 + 2 * (atom & 1);
  const uint32_t* lows = reinterpret_cast<const uint32_t*>(src + plane * 512);
  const uint32_t lo0 = lows[lo_index];
  const uint32_t lo1 = lows[lo_index + 1];
  const uint16_t lower_meta = selected_meta(meta, 0);
  const uint16_t upper_meta = selected_meta(meta, 1);
  const bool lower_collision = meta != 0u && lower_meta == 0u;
  const bool upper_collision = meta != 0u && upper_meta == 0u;

  int warp_prefix = collision_count(meta);
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    const int other = NF12_ROWMAJOR_SHFL_UP(warp_prefix, offset);
    if ((tid & 31) >= offset) warp_prefix += other;
  }
  __shared__ int warp_totals[4];
  if ((tid & 31) == 31) warp_totals[tid >> 5] = warp_prefix;
  __syncthreads();

  int prior_warps = 0;
#pragma unroll
  for (int warp = 0; warp < 4; ++warp) {
    if (warp < tid >> 5) prior_warps += warp_totals[warp];
  }
  const int count = collision_count(meta);
  const int64_t prior_tiles = tile != 0 ? prefix[tile - 1] : 0;
  const uint32_t first_collision = static_cast<uint32_t>(
      2 * original_escapes + prior_tiles + prior_warps + warp_prefix - count);
  const int64_t block = tile / k_blocks;
  const int64_t kb = tile - block * k_blocks;
  const int ri = lane >> 2;
  const int group = 4 * atom + (lane & 3);

  if (meta == 0u && lo0 >= original_escapes) atomicExch(invalid_escape, 1);

#pragma unroll
  for (int upper = 0; upper < 2; ++upper) {
    const uint16_t selected = upper ? upper_meta : lower_meta;
    const uint32_t normal_lows = selected_lows(lo0, lo1, upper);
    const bool collision = upper ? upper_collision : lower_collision;
    const uint32_t escape_index =
        first_collision + (upper ? static_cast<int>(lower_collision) : 0);
    const uint32_t out_lows =
        meta == 0u ? static_cast<uint32_t>(lo0 * 2 + upper)
                   : collision ? escape_index : normal_lows;
    const int64_t row = block * 16 + ri + upper * 8;
    uint8_t* dst = dest + (row * k_blocks + kb) * kRowBlockBytes;
    reinterpret_cast<uint16_t*>(dst)[group] = selected;
    reinterpret_cast<uint32_t*>(dst + kRowMetaBytes)[group] = out_lows;
    if (collision) {
      dest_escapes[escape_index] = collision_escape(normal_lows, base);
    }
  }
}

__global__ void split_escapes(const uint4* __restrict__ source,
                              int64_t count,
                              uint64_t* __restrict__ dest) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const uint4 bits = source[index];
  dest[2 * index] =
      static_cast<uint64_t>(bits.x) | (static_cast<uint64_t>(bits.z) << 32);
  dest[2 * index + 1] =
      static_cast<uint64_t>(bits.y) | (static_cast<uint64_t>(bits.w) << 32);
}

}  // namespace nf12_rowmajor
