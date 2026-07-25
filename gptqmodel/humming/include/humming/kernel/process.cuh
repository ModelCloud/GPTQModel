// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.


#include <humming/utils/all.cuh>

template <uint32_t kNumBitsB, uint32_t kNumBitsA>
CUDA_INLINE void humming_pack_weight(uint32_t *in_arr, uint32_t *out_arr, uint32_t interleave_mode) {
  constexpr uint32_t kNumBitsPaddedB = static_next_power_of_2(kNumBitsB);

  auto get_interleaved_index = [&](uint32_t i) {
    constexpr uint32_t N = 32 / kNumBitsPaddedB;
    constexpr uint32_t stride = 32 / kNumBitsA;

    uint32_t group_size = N / stride;
    uint32_t col = i % group_size;
    uint32_t row = i / group_size;
    return col * stride + row;
  };

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    PRAGMA_UNROLL
    for (uint32_t j = 0; j < kNumBitsPaddedB; j++) {
      uint32_t val = 0;
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < 32 / kNumBitsPaddedB; k++) {
        uint32_t new_k = get_interleaved_index(k);
        constexpr uint32_t mask1 = (1 << (kNumBitsB - 1));
        constexpr uint32_t mask2 = mask1 - 1;

        if (interleave_mode % 2 == 0) {
          uint32_t single_val = in_arr[i * 32 + j * 32 / kNumBitsPaddedB + k];
          val |= (single_val & mask2) << (k * kNumBitsPaddedB);
        } else {
          uint32_t single_val = in_arr[i * 32 + j * 32 / kNumBitsPaddedB + new_k];
          val |= (single_val & mask2) << (k * kNumBitsPaddedB);
        };

        if (interleave_mode / 2 == 0) {
          uint32_t single_val = in_arr[i * 32 + j * 32 / kNumBitsPaddedB + k];
          val |= (single_val & mask1) << (k * kNumBitsPaddedB);
        } else {
          uint32_t single_val = in_arr[i * 32 + j * 32 / kNumBitsPaddedB + new_k];
          val |= (single_val & mask1) << (k * kNumBitsPaddedB);
        };
      }

      out_arr[i * kNumBitsPaddedB + j] = val;
    }
  }

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    if constexpr (kNumBitsB == 3) {
      out_arr[3 * i + 0] = (out_arr[4 * i + 0] & 0x77777777) | ((out_arr[4 * i + 1] & 0x44444444) << 1);
      out_arr[3 * i + 1] = (out_arr[4 * i + 1] & 0x33333333) | ((out_arr[4 * i + 2] & 0x66666666) << 1);
      out_arr[3 * i + 2] = (out_arr[4 * i + 2] & 0x11111111) | ((out_arr[4 * i + 3] & 0x77777777) << 1);
    } else if constexpr (kNumBitsB == 5) {
      out_arr[5 * i + 0] = (out_arr[8 * i + 0] & 0x1F1F1F1F) | ((out_arr[8 * i + 1] & 0x1C1C1C1C) << 3);
      out_arr[5 * i + 1] = (out_arr[8 * i + 1] & 0x03030303) | ((out_arr[8 * i + 2] & 0x1F1F1F1F) << 2) | ((out_arr[8 * i + 3] & 0x10101010) << 3);
      out_arr[5 * i + 2] = (out_arr[8 * i + 3] & 0x0F0F0F0F) | ((out_arr[8 * i + 4] & 0x1E1E1E1E) << 3);
      out_arr[5 * i + 3] = (out_arr[8 * i + 4] & 0x01010101) | ((out_arr[8 * i + 5] & 0x1F1F1F1F) << 1) | ((out_arr[8 * i + 6] & 0x18181818) << 3);
      out_arr[5 * i + 4] = (out_arr[8 * i + 6] & 0x07070707) | ((out_arr[8 * i + 7] & 0x1F1F1F1F) << 3);
    } else if constexpr (kNumBitsB == 6) {
      out_arr[6 * i + 0] = (out_arr[8 * i + 0] & 0x3F3F3F3F) | ((out_arr[8 * i + 1] & 0x30303030) << 2);
      out_arr[6 * i + 1] = (out_arr[8 * i + 1] & 0x0F0F0F0F) | ((out_arr[8 * i + 2] & 0x3C3C3C3C) << 2);
      out_arr[6 * i + 2] = (out_arr[8 * i + 2] & 0x03030303) | ((out_arr[8 * i + 3] & 0x3F3F3F3F) << 2);
      out_arr[6 * i + 3] = (out_arr[8 * i + 4] & 0x3F3F3F3F) | ((out_arr[8 * i + 5] & 0x30303030) << 2);
      out_arr[6 * i + 4] = (out_arr[8 * i + 5] & 0x0F0F0F0F) | ((out_arr[8 * i + 6] & 0x3C3C3C3C) << 2);
      out_arr[6 * i + 5] = (out_arr[8 * i + 6] & 0x03030303) | ((out_arr[8 * i + 7] & 0x3F3F3F3F) << 2);
    } else if constexpr (kNumBitsB == 7) {
      out_arr[7 * i + 0] = (out_arr[8 * i + 0] & 0x7F7F7F7F) | ((out_arr[8 * i + 1] & 0x40404040) << 1);
      out_arr[7 * i + 1] = (out_arr[8 * i + 1] & 0x3F3F3F3F) | ((out_arr[8 * i + 2] & 0x60606060) << 1);
      out_arr[7 * i + 2] = (out_arr[8 * i + 2] & 0x1F1F1F1F) | ((out_arr[8 * i + 3] & 0x70707070) << 1);
      out_arr[7 * i + 3] = (out_arr[8 * i + 3] & 0x0F0F0F0F) | ((out_arr[8 * i + 4] & 0x78787878) << 1);
      out_arr[7 * i + 4] = (out_arr[8 * i + 4] & 0x07070707) | ((out_arr[8 * i + 5] & 0x7C7C7C7C) << 1);
      out_arr[7 * i + 5] = (out_arr[8 * i + 5] & 0x03030303) | ((out_arr[8 * i + 6] & 0x7E7E7E7E) << 1);
      out_arr[7 * i + 6] = (out_arr[8 * i + 6] & 0x01010101) | ((out_arr[8 * i + 7] & 0x7F7F7F7F) << 1);
    }
  }
};


template <uint32_t kNumBitsB, bool kPackedInput>
CUDA_INLINE uint32_t extract_packed_value(uint32_t *smem_row, uint32_t index) {

  if constexpr (!kPackedInput) return smem_row[index];

  uint32_t start_bits = index * kNumBitsB;
  uint32_t end_bits = (index + 1) * kNumBitsB;

  uint32_t start_int_index = start_bits / 32;
  uint32_t end_int_index = (end_bits - 1) / 32;
  uint32_t extract_value;
  if constexpr (32 % kNumBitsB == 0) {
    constexpr uint32_t extracted_mask = (1 << kNumBitsB) - 1;
    extract_value = smem_row[start_int_index] >> (start_bits % 32);
    extract_value = extract_value & extracted_mask;
  } else {
    uint32_t part2_bits = start_int_index == end_int_index ? 0 : end_bits % 32;
    uint32_t part2_mask = (1 << part2_bits) - 1;
    uint32_t part1_bits = kNumBitsB - part2_bits;
    uint32_t part1_mask = (1 << part1_bits) - 1;

    uint32_t extract_value1 = smem_row[start_int_index] >> (start_bits % 32);
    extract_value1 = extract_value1 & part1_mask;
    if (part2_bits > 0) {
      uint32_t extract_value2 = smem_row[end_int_index] & part2_mask;
      extract_value2 = extract_value2 << part1_bits;
      extract_value = extract_value1 | extract_value2;
    } else {
      extract_value = extract_value1;
    }
  }

  return extract_value;
}


template <
    uint32_t kNumBitsB, uint32_t kNumBitsA, bool kPackedInput,
    bool kShouldPreprocessForINT2FP, bool kShouldPreprocessWithZP,
    bool kShouldTransposeMiniBlock, uint32_t kGroupSizeZP,
    bool kUsePackedKLayout = false>
__global__ void weight_repack_nk(
    const uint32_t *in_ptr, uint32_t *out_ptr, const uint32_t *zp_ptr,
    uint32_t shape_n, uint32_t shape_k,
    uint32_t padded_shape_n, uint32_t padded_shape_k, uint32_t interleave_mode) {

  constexpr uint32_t kNumBitsInputB = kPackedInput ? kNumBitsB : 32;
  constexpr uint32_t smem_stride = 64 * kNumBitsInputB / 32;
  constexpr uint32_t zp_smem_stride = kShouldPreprocessWithZP ? CEIL_DIV(64, kGroupSizeZP) : 0;
  constexpr uint32_t zp_smem_num_rows = kPackedInput ? (64 * kNumBitsB / 32) : 64;
  uint32_t gmem_stride = CEIL_DIV(shape_k * kNumBitsInputB, 32);
  uint32_t zp_gmem_stride = kShouldPreprocessWithZP ? CEIL_DIV(shape_k, kGroupSizeZP) : 0;
  if constexpr (kShouldPreprocessWithZP) static_assert(kGroupSizeZP > 0);

  assert(padded_shape_n % 64 == 0);
  assert(padded_shape_k % (512 / kNumBitsA) == 0);
  assert(blockDim.x == 32);

  __shared__ uint32_t smem[64][smem_stride];
  __shared__ uint32_t zp_smem[zp_smem_num_rows][MAX(zp_smem_stride, 1)];
  uint32_t *smem_arr = reinterpret_cast<uint32_t *>(smem);
  constexpr uint32_t iters = sizeof(smem) / 4 / 32;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < iters; i++) {
    uint32_t smem_offset = i * 32 + threadIdx.x;
    uint32_t row = smem_offset / smem_stride + blockIdx.x * 64;
    uint32_t col = smem_offset % smem_stride + blockIdx.y * smem_stride;
    uint32_t glob_offset = row * gmem_stride + col + blockIdx.z * (gmem_stride * shape_n);

    if (row < shape_n && col < gmem_stride) {
      smem_arr[smem_offset] = in_ptr[glob_offset];
    } else {
      smem_arr[smem_offset] = 0;
    }
  };

  if constexpr (kShouldPreprocessWithZP) {
    uint32_t zp_max_row = kPackedInput ? (shape_n * kNumBitsB / 32) : shape_n;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(zp_smem_num_rows, 32); i++) {
      if (i * 32 + threadIdx.x < zp_smem_num_rows) {

        if constexpr (zp_smem_stride > 0) {
          PRAGMA_UNROLL
          for (uint32_t j = 0; j < zp_smem_stride; j++) {
            uint32_t row = blockIdx.x * zp_smem_num_rows + i * 32 + threadIdx.x;
            uint32_t col = (blockIdx.y * 64 + j * kGroupSizeZP) / kGroupSizeZP;
            uint32_t glob_offset = row * zp_gmem_stride + col + blockIdx.z * (zp_gmem_stride * (shape_n * kNumBitsB / 32));

            if (row < zp_max_row && col < zp_gmem_stride) {
              zp_smem[i * 32 + threadIdx.x][j] = zp_ptr[glob_offset];
            }
          }
        }
      }
    }
  };

  __syncthreads();

  // 4bit: [1][4][1][2][2][8]
  // 8bit: [2][2][2][2][2][4]
  // 16bit: [4][1][4][2][2][2]
  // Take 16bit activation as an example,
  // the packing block is (n, k) = (64, 16)
  // - INDEX1 (4) = 64 (total_k) / 16 (block_k)
  // - INDEX2 (1) = 64 (total_n) / 64 (block_n)
  // In each block, we pack 16x16 block
  // - INDEX3 (4) = 64 (block_n) / 16
  // In each 16x16 block, pack 8x8 block
  // - INDEX4 (2) = 16 (dim n) / 8
  // - INDEX5 (2) = 16 (dim k) / 8
  // In each 8x8, each thread process 2 elements
  // - INDEX6 (2) = 64 (total_elements) / 32 (num_threads)
  uint32_t tmp[kNumBitsA / 4][16 / kNumBitsA][kNumBitsA / 4][2][2][32 / kNumBitsA];

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 8; i++) {
    uint32_t row = i * 8 + threadIdx.x / 4;
    uint32_t zp_smem_row[MAX(zp_smem_stride, 1)];

    if constexpr (zp_smem_stride > 0) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < zp_smem_stride; j++) {
        if constexpr (!kPackedInput) {
          zp_smem_row[j] = zp_smem[row][j];
        } else {
          constexpr uint32_t extracted_mask = (1 << kNumBitsB) - 1;
          uint32_t start_bits = row * kNumBitsB;
          uint32_t end_bits = (row + 1) * kNumBitsB;
          uint32_t start_word = start_bits / 32;
          uint32_t end_word = (end_bits - 1) / 32;
          uint32_t val = zp_smem[start_word][j] >> (start_bits % 32);
          if (start_word != end_word) {
            val |= zp_smem[end_word][j] << (32 - (start_bits % 32));
          }
          zp_smem_row[j] = val & extracted_mask;
        }
      }
    }

    uint32_t *smem_row = smem[i * 8 + threadIdx.x / 4];
    PRAGMA_UNROLL
    for (uint32_t j = 0; j < kNumBitsA / 2; j++) {
      // 8/4/2 for 16/8/4 bits A
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < 32 / kNumBitsA; k++) {
        // 2/4/8 for 16/8/4 bits A
        uint32_t index = j * (128 / kNumBitsA) + threadIdx.x % 4 * (32 / kNumBitsA) + k;

        uint32_t extract_value = extract_packed_value<kNumBitsB, kPackedInput>(smem_row, index);

        if constexpr (kShouldPreprocessForINT2FP) {
          uint32_t zp_val;
          constexpr uint32_t extracted_mask = (1 << kNumBitsB) - 1;

          if constexpr (kShouldPreprocessWithZP) {
            zp_val = zp_smem_row[index / kGroupSizeZP];
          } else {
            zp_val = 1 << (kNumBitsB - 1);
          }

          extract_value = extract_value & extracted_mask;
          extract_value = extract_value >= zp_val ? extract_value - zp_val : extracted_mask - extract_value;
        }

        uint32_t i1 = j / 2;
        uint32_t i2 = (i * 8) / (kNumBitsA * 4);
        uint32_t i3 = (i * 8) % (kNumBitsA * 4) / 16;
        uint32_t i4 = i % 2;
        uint32_t i5 = j % 2;
        uint32_t i6 = k;
        if constexpr (kShouldTransposeMiniBlock) {
          tmp[i1][i2][i3][i5][i4][i6] = extract_value;
        } else {
          tmp[i1][i2][i3][i4][i5][i6] = extract_value;
        }
      }
    }
  }

  uint32_t *tmp2 = reinterpret_cast<uint32_t *>(tmp);
  constexpr uint32_t kNumBitsPaddedB = static_next_power_of_2(kNumBitsB);
  uint32_t out_arr[4 * kNumBitsPaddedB];

  humming_pack_weight<kNumBitsB, kNumBitsA>(reinterpret_cast<uint32_t *>(tmp), out_arr, interleave_mode);

  uint32_t out_stride = (256 / kNumBitsA) * padded_shape_n * kNumBitsB / 32;
  uint32_t col_offset = (256 / kNumBitsA) * (64 * blockIdx.x) * kNumBitsB / 32;
  uint32_t global_max_row = gridDim.z * padded_shape_k / (256 / kNumBitsA);

  constexpr uint32_t num_output_rows = kNumBitsA / 4;
  constexpr uint32_t num_ints_per_row = 16 * kNumBitsB / kNumBitsA;

  if constexpr (kUsePackedKLayout) {
    static_assert(kNumBitsA == 8);
    constexpr uint32_t hb = kNumBitsB / 2;
    uint32_t packed_out_stride = 64 * padded_shape_n * kNumBitsB / 32;
    uint32_t packed_max_row = gridDim.z * padded_shape_k / 64;
    uint32_t row = (blockIdx.y * 64 + blockIdx.z * padded_shape_k) / 64;
    if (row < packed_max_row) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < num_output_rows; i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < num_ints_per_row / kNumBitsB; j++) {
          PRAGMA_UNROLL
          for (uint32_t k = 0; k < kNumBitsB; k++) {
            uint32_t region = blockIdx.x * 4 + j * 2 + k / hb;
            uint32_t s = i * hb + k % hb;
            uint32_t col = region * (32 * kNumBitsB) + threadIdx.x * kNumBitsB + s;
            out_ptr[row * packed_out_stride + col] = out_arr[i * num_ints_per_row + j * kNumBitsB + k];
          }
        }
      }
    }
    return;
  }

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < num_output_rows; i++) {
    uint32_t row = (blockIdx.y * 64 + blockIdx.z * padded_shape_k) / (256 / kNumBitsA) + i;
    if (row >= global_max_row) continue;

    PRAGMA_UNROLL
    for (uint32_t j = 0; j < num_ints_per_row / kNumBitsB; j++) {
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < kNumBitsB; k++) {
        uint32_t col = col_offset + j * 32 * kNumBitsB + threadIdx.x * kNumBitsB + k;
        out_ptr[row * out_stride + col] = out_arr[i * num_ints_per_row + j * kNumBitsB + k];
      }
    }
  }
}
