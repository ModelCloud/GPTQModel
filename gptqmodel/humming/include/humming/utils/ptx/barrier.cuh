// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/base.cuh>


template <uint32_t kNumSyncThreads, uint32_t kNumThreads, uint32_t kBarrierId = 1>
CUDA_INLINE uint32_t sync_part_threads() {
  if constexpr (kNumSyncThreads == kNumThreads) {
    __syncthreads();
  } else {
    static_assert(kNumThreads >= kNumSyncThreads);
    static_assert(kNumSyncThreads > 0);
    asm volatile("bar.sync %0, %1;" ::"r"(kBarrierId), "r"(kNumSyncThreads));
  }
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
#if HUMMING_DEBUG_KERNEL
    uint64_t start_clock = debug_kernel_timer_start();
#endif
    int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
#if HUMMING_DEBUG_KERNEL
      debug_kernel_timeout_check(start_clock, "Humming Stream-K barrier timeout");
#endif
    } while (state != count);
  }
  sync_part_threads<kNumSyncThreads, kNumThreads>();
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_acquire2(int *lock, int count) {
  if (threadIdx.x == 0) {
#if HUMMING_DEBUG_KERNEL
    uint64_t start_clock = debug_kernel_timer_start();
#endif
    int state = 1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
#if HUMMING_DEBUG_KERNEL
      debug_kernel_timeout_check(start_clock, "Humming Stream-K barrier timeout");
#endif
    } while (state > count);
  }
  sync_part_threads<kNumSyncThreads, kNumThreads>();
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_release(int *lock, bool reset = false) {
  sync_part_threads<kNumSyncThreads, kNumThreads>();
  if (threadIdx.x == 0) {
    if (reset) {
      __stcg(&lock[0], 0);
    } else {
      int32_t val = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(val));
    }
  }
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_release2(int *lock, int32_t val) {
  sync_part_threads<kNumSyncThreads, kNumThreads>();
  if (threadIdx.x == 0) {
    if (val < 0) {
      asm volatile("fence.acq_rel.gpu;\n");
      __stcg(&lock[0], val);
    } else {
      int32_t val2 = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(val2));
    }
  }
}

CUDA_INLINE
void mbarrier_wait(void *barrier, bool phase_parity) {
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);
#if HUMMING_DEBUG_KERNEL
  uint64_t start_clock = debug_kernel_timer_start();
  uint32_t ready;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  do {
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "  selp.u32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(ready)
                 : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
                 : "memory");
    debug_kernel_timeout_check(start_clock, "Humming mbarrier timeout");
  } while (!ready);
#else
  do {
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "  selp.u32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(ready)
                 : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
                 : "memory");
    debug_kernel_timeout_check(start_clock, "Humming mbarrier timeout");
  } while (!ready);
#endif
#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  waitLoop:\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @p bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#else
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  waitLoop:\n"
               "  mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @p bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#endif
#endif
};


template <bool kUseCluster>
CUDA_INLINE void mbarrier_init_sync() {
  if constexpr (kUseCluster) {
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
  } else {
    __syncthreads();
  }
}


template <bool kUseCluster = false>
CUDA_INLINE void mbarrier_arrive(void *barrier) {
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);

  if constexpr (kUseCluster) {
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :
                 : "r"(smem_int_mbar)
                 : "memory");
  } else {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
                 :
                 : "r"(smem_int_mbar)
                 : "memory");
  }
};
