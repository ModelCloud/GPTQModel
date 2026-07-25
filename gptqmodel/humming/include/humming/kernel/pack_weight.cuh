// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.


#include <humming/utils/all.cuh>

template <uint32_t kNumBits>
CUDA_INLINE void common_pack_weight(uint32_t *in_arr, uint32_t *out_arr) {
  constexpr uint32_t mask = (1 << kNumBits) - 1;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kNumBits; i++)
    out_arr[i] = 0;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 32; i++) {
    uint32_t index = i * kNumBits;
    uint32_t word_idx = index / 32;
    uint32_t bit_offset = index % 32;

    uint32_t val = in_arr[i] & mask;

    out_arr[word_idx] |= (val << bit_offset);

    if (bit_offset + kNumBits > 32) {
      uint32_t part1_bits = 32 - bit_offset;
      out_arr[word_idx + 1] |= (val >> part1_bits);
    }
  }
};

template <uint32_t kNumBits>
CUDA_INLINE void common_unpack_weight(uint32_t *in_arr, uint32_t *out_arr) {
  constexpr uint32_t mask = (1 << kNumBits) - 1;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 32; i++) {
    uint32_t index = i * kNumBits;
    uint32_t word_idx = index / 32;
    uint32_t bit_offset = index % 32;

    uint32_t part1_bits = MIN(32 - bit_offset, kNumBits);
    uint32_t part2_bits = kNumBits - part1_bits;

    uint32_t mask1 = (1 << part1_bits) - 1;
    out_arr[i] = (in_arr[index / 32] >> bit_offset) & mask1;

    if (part2_bits > 0) {
      uint32_t mask2 = (1 << part2_bits) - 1;
      out_arr[i] |= (in_arr[index / 32 + 1] & mask2) << part1_bits;
    }
  }
};

template <uint32_t kNumBits>
__global__ void pack_weight_kernel(const uint32_t *in_ptr, uint32_t *out_ptr, int64_t num_elements) {
  constexpr uint32_t iters = 1;
  constexpr uint32_t num_ints_per_iter = 32;
  uint32_t in_buffer[32];
  uint32_t out_buffer[kNumBits];

  PRAGMA_UNROLL_COUNT(32)
  for (uint32_t i = 0; i < iters; i++) {
    uint32_t in_offset = blockIdx.x * (iters * 32 * 32) + i * 32 * 32 + threadIdx.x * 32;
    uint32_t out_offset = blockIdx.x * (iters * 32 * kNumBits) + i * 32 * kNumBits + threadIdx.x * kNumBits;

    PRAGMA_UNROLL_COUNT(32)
    for (uint32_t i = 0; i < 32; i++) {
      if (in_offset + i >= num_elements) continue;
      in_buffer[i] = in_ptr[in_offset + i];
    }
    common_pack_weight<kNumBits>(in_buffer, out_buffer);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumBits; i++) {
      if (out_offset + i >= (num_elements * kNumBits / 32)) continue;
      out_ptr[out_offset + i] = out_buffer[i];
    }
  };
}


template <uint32_t kNumBits>
__global__ void unpack_weight_kernel(const uint32_t *in_ptr, uint32_t *out_ptr, int64_t num_elements) {
  constexpr uint32_t iters = 1;
  constexpr uint32_t num_ints_per_iter = 32;
  uint32_t in_buffer[kNumBits];
  uint32_t out_buffer[32];

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < iters; i++) {
    uint32_t in_offset = blockIdx.x * (iters * 32 * kNumBits) + i * 32 * kNumBits + threadIdx.x * kNumBits;
    uint32_t out_offset = blockIdx.x * (iters * 32 * 32) + i * 32 * 32 + threadIdx.x * 32;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumBits; i++) {
      if (in_offset + i >= (num_elements * kNumBits / 32)) continue;
      in_buffer[i] = in_ptr[in_offset + i];
    }
    common_unpack_weight<kNumBits>(in_buffer, out_buffer);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 32; i++) {
      if (out_offset + i >= num_elements) continue;
      out_ptr[out_offset + i] = out_buffer[i];
    }
  };
}
