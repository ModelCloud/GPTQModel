// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/base.cuh>

CUDA_INLINE
void prefetch_tensor_map(const void *desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile("prefetch.tensormap [%0];"
               :
               : "l"(gmem_int_desc)
               : "memory");
};

template <uint32_t kMultiCastSize = 1>
CUDA_INLINE void tma_load_1d(const void *desc_ptr, void *smem_ptr, void *mbar_ptr, uint32_t crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);

  if constexpr (kMultiCastSize == 1) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0)
                 : "memory");
  } else {
    constexpr uint16_t cast_mask = (1 << kMultiCastSize) - 1;
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%4}], [%2], %3;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "h"(cast_mask), "r"(crd0)
                 : "memory");
  }
};

template <uint32_t kMultiCastSize = 1>
CUDA_INLINE void tma_load_2d(const void *desc_ptr, void *smem_ptr, void *mbar_ptr, uint32_t crd0, uint32_t crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);

  if constexpr (kMultiCastSize == 1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1)
                 : "memory");
  } else {
    constexpr uint16_t cast_mask = (1 << kMultiCastSize) - 1;
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%4, %5}], [%2], %3;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "h"(cast_mask), "r"(crd0), "r"(crd1)
                 : "memory");
  }
}

template <uint32_t kMultiCastSize = 1>
CUDA_INLINE void tma_load_3d(const void *desc_ptr, void *smem_ptr, void *mbar_ptr, uint32_t crd0, uint32_t crd1, uint32_t crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);

  if constexpr (kMultiCastSize == 1) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1), "r"(crd2)
                 : "memory");
  } else {
    constexpr uint16_t cast_mask = (1 << kMultiCastSize) - 1;
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%4, %5, %6}], [%2], %3;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "h"(cast_mask), "r"(crd0), "r"(crd1), "r"(crd2)
                 : "memory");
  }
}

CUDA_INLINE void tma_store_2d(void *smem_ptr, const void *desc_ptr, uint32_t crd0, uint32_t crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);

  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
               " [%0, {%2, %3}], [%1];"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
               : "memory");
};

CUDA_INLINE void tma_reduce_add_2d(void *smem_ptr, const void *desc_ptr, uint32_t crd0, uint32_t crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);

  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"
               " [%0, {%2, %3}], [%1];"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
               : "memory");
};

CUDA_INLINE void tma_commit_mbarrier(void *mbar_ptr, uint32_t bytes) {
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(mbar_ptr);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(smem_int_ptr), "r"(bytes));
};

CUDA_INLINE void tma_commit_store_group() {
  asm volatile("cp.async.bulk.commit_group;\n");
};

CUDA_INLINE void tma_fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
};

template <uint32_t N, bool only_wait_read = false>
CUDA_INLINE void tma_wait_store_group() {
  if constexpr (only_wait_read) {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N));
  } else {
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N));
  }
};

template <uint32_t ord>
CUDA_INLINE void tensor_map_replace_global_dim(void *smem_desc_ptr, uint32_t value) {
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_desc_ptr);
  asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], %1, %2;\n"
               :
               : "r"(smem_int_ptr), "n"(ord), "r"(value)
               : "memory");
};

CUDA_INLINE void tensor_map_release_cta() {
  asm volatile("fence.proxy.tensormap::generic.release.cta;");
};

CUDA_INLINE void tensor_map_acquire_cta(const void *gmem_desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
  asm volatile("fence.proxy.tensormap::generic.acquire.cta [%0], 128;" ::"l"(gmem_int_desc)
               : "memory");
};
