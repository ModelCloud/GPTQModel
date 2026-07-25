// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/mxmma.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>

#include <humming/datatype/dequant.cuh>


template <class MmaOpClass, uint32_t kRepeatCount, uint32_t kUnrollCount>
__global__ void tops_bench(uint32_t *out_ptr) {

  typename MmaOpClass::CRegisters regs_c;

  if constexpr (MmaOpClass::kMmaType == MmaType::WGMMA) {
    typename MmaOpClass::BRegisters regs_b;
    __shared__ alignas(1024) int4 smem[2048];
    uint64_t desc = make_wgmma_smem_desc<128>(cast_smem_ptr_to_uint(smem));
    PRAGMA_UNROLL_COUNT(kUnrollCount)
    for (uint32_t i = 0; i < kRepeatCount; i++) {
      wgmma_fence();
      MmaOpClass::fma(desc, regs_b, regs_c);
      wgmma_commit();
      wgmma_wait<0>();
    }
  } else if constexpr (MmaOpClass::kMmaType == MmaType::MXMMA) {
    typename MmaOpClass::ARegisters regs_a;
    typename MmaOpClass::BRegisters regs_b;
    PRAGMA_UNROLL_COUNT(kUnrollCount)
    for (uint32_t i = 0; i < kRepeatCount; i++)
      MmaOpClass::fma(regs_a, regs_b, 0u, 0u, regs_c, regs_c, 0, 0, 0, 0);
  } else {
    typename MmaOpClass::ARegisters regs_a;
    typename MmaOpClass::BRegisters regs_b;
    PRAGMA_UNROLL_COUNT(kUnrollCount)
    for (uint32_t i = 0; i < kRepeatCount; i++)
      MmaOpClass::fma(regs_a, regs_b, regs_c, regs_c);
  }

  if (blockIdx.x == 8192) out_ptr[0] = reinterpret_cast<uint32_t *>(&regs_c)[0];
};
