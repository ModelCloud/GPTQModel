// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/utils/base.cuh>

CUDA_INLINE void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::
                   : "memory");
}

CUDA_INLINE void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::
                   : "memory");
}

template <uint32_t N>
CUDA_INLINE void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N)
               : "memory");
}

CUDA_INLINE void warpgroup_fence_operand(uint32_t &reg) {
  asm volatile(""
               : "+r"(reg)::"memory");
}
