// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "qvq_p32_abi.h"

#include <type_traits>

static_assert(QVQ_P32_OPERATION_VERSION == 1);
static_assert(QVQ_P32_ABI_VERSION == 1);
static_assert(QVQ_P32_KERNEL_VERSION == 8);
static_assert(QVQ_P32_COMPILED_SM == 80);
static_assert(QVQ_P32_TILE_SIZE == 16);
static_assert(QVQ_P32_LEVEL_COUNT == 256);
static_assert(QVQ_P32_TRANSITION_BITS_MIN == 4);
static_assert(QVQ_P32_TRANSITION_BITS_MAX == 7);
static_assert(QVQ_P32_SPLIT_COUNT_MAX == 128);
static_assert(QVQ_P32_STAGE_K_TILES_MIN == 1);
static_assert(QVQ_P32_STAGE_K_TILES_MAX == 4);
static_assert(QVQ_P32_SCALAR_M_MAX == 4);
static_assert(QVQ_P32_GROUPED_M_MAX == 16);
static_assert(QVQ_P32_GROUP_COUNT_MIN == 2);
static_assert(QVQ_P32_GROUP_COUNT_MAX == 3);

static_assert(QVQ_P32_VARIANT_SCALAR == 1);
static_assert(QVQ_P32_VARIANT_BLOCK == 2);
static_assert(QVQ_P32_REDUCTION_NATIVE == 1);
static_assert(QVQ_P32_REDUCTION_PARTIALS == 2);
static_assert(std::is_standard_layout_v<qvq_p32_config>);
static_assert(sizeof(qvq_p32_config) == 6 * sizeof(int));

int main() {
  const qvq_p32_config config = {
      1,
      QVQ_P32_VARIANT_SCALAR,
      128,
      QVQ_P32_STAGE_K_TILES_MIN,
      0,
      QVQ_P32_REDUCTION_NATIVE,
  };
  return config.split_count == 1 ? 0 : 1;
}
