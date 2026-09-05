// SPDX-License-Identifier: Apache-2.0
// Framework-neutral QVQ composite-Hadamard contract.
#ifndef GPTQMODEL_EXT_QVQ_P32_QVQ_P32_HADAMARD_H_
#define GPTQMODEL_EXT_QVQ_P32_QVQ_P32_HADAMARD_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return the canonical dense base order used after the power-of-two
// butterfly stages. Zero means this compact public table does not cover the
// requested width; one means a pure Walsh-Hadamard transform.
static inline int32_t qvq_p32_hadamard_base_order(int64_t width) {
  if (width <= 0) return 0;
  if (width % 40 == 0 && ((width / 40) & (width / 40 - 1)) == 0) return 40;
  if (width % 20 == 0 && ((width / 20) & (width / 20 - 1)) == 0) return 20;
  if (width % 12 == 0 && ((width / 12) & (width / 12 - 1)) == 0) return 12;
  if ((width & (width - 1)) == 0) return 1;
  return 0;
}

// Return one exact {-1,+1} coefficient from QVQ's canonical small base.
// Rows are bit-packed only to keep framework adapters from copying large
// literal matrices; graph compilers materialize these values as ordinary SSA
// constants before optimization.
static inline int32_t qvq_p32_hadamard_base_value(int32_t order, int32_t row,
                                                   int32_t column) {
  static const uint64_t had12[12] = {
      UINT64_C(0x1), UINT64_C(0xb8b), UINT64_C(0x717), UINT64_C(0xe2d),
      UINT64_C(0xc5b), UINT64_C(0x8b7), UINT64_C(0x16f), UINT64_C(0x2dd),
      UINT64_C(0x5b9), UINT64_C(0xb71), UINT64_C(0x6e3), UINT64_C(0xdc5),
  };
  static const uint64_t had20[20] = {
      UINT64_C(0x1), UINT64_C(0x9ea1b), UINT64_C(0x3d437), UINT64_C(0x7a86d),
      UINT64_C(0xf50d9), UINT64_C(0xea1b3), UINT64_C(0xd4367), UINT64_C(0xa86cf),
      UINT64_C(0x50d9f), UINT64_C(0xa1b3d), UINT64_C(0x4367b), UINT64_C(0x86cf5),
      UINT64_C(0xd9eb), UINT64_C(0x1b3d5), UINT64_C(0x367a9), UINT64_C(0x6cf51),
      UINT64_C(0xd9ea1), UINT64_C(0xb3d43), UINT64_C(0x67a87), UINT64_C(0xcf50d),
  };
  static const uint64_t had40[40] = {
      UINT64_C(0x100001), UINT64_C(0x9ea1b9ea1b), UINT64_C(0x3d4373d437),
      UINT64_C(0x7a86d7a86d), UINT64_C(0xf50d9f50d9), UINT64_C(0xea1b3ea1b3),
      UINT64_C(0xd4367d4367), UINT64_C(0xa86cfa86cf), UINT64_C(0x50d9f50d9f),
      UINT64_C(0xa1b3da1b3d), UINT64_C(0x4367b4367b), UINT64_C(0x86cf586cf5),
      UINT64_C(0xd9eb0d9eb), UINT64_C(0x1b3d51b3d5), UINT64_C(0x367a9367a9),
      UINT64_C(0x6cf516cf51), UINT64_C(0xd9ea1d9ea1), UINT64_C(0xb3d43b3d43),
      UINT64_C(0x67a8767a87), UINT64_C(0xcf50dcf50d), UINT64_C(0xffffe00001),
      UINT64_C(0x615e49ea1b), UINT64_C(0xc2bc83d437), UINT64_C(0x857927a86d),
      UINT64_C(0xaf26f50d9), UINT64_C(0x15e4cea1b3), UINT64_C(0x2bc98d4367),
      UINT64_C(0x57930a86cf), UINT64_C(0xaf26050d9f), UINT64_C(0x5e4c2a1b3d),
      UINT64_C(0xbc9844367b), UINT64_C(0x7930a86cf5), UINT64_C(0xf26140d9eb),
      UINT64_C(0xe4c2a1b3d5), UINT64_C(0xc9856367a9), UINT64_C(0x930ae6cf51),
      UINT64_C(0x2615ed9ea1), UINT64_C(0x4c2bcb3d43), UINT64_C(0x9857867a87),
      UINT64_C(0x30af2cf50d),
  };
  if (row < 0 || column < 0 || row >= order || column >= order) return 0;
  const uint64_t *rows = order == 12 ? had12 : order == 20 ? had20 :
                         order == 40 ? had40 : 0;
  if (order == 1) return 1;
  if (rows == 0) return 0;
  return ((rows[row] >> column) & UINT64_C(1)) != 0 ? 1 : -1;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GPTQMODEL_EXT_QVQ_P32_QVQ_P32_HADAMARD_H_
