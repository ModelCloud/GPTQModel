// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 kernel computation.
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original implementation: https://github.com/dphnAI/nf12
// Copyright (c) 2026 AlpinDale and the nf12 contributors.
// Licensed under the Apache License, Version 2.0.
//
// Vendored into GPT-QModel (https://github.com/ModelCloud/GPTQModel) and
// modified for the NF12 method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include <nf12/format.cuh>

#include "internal.h"

namespace nf12::detail {

NF12_HD int64_t moe_expert_bytes(WeightSource source, int64_t n, int64_t k) {
  return n / 16 * (k / 64) * (source == WeightSource::kNF12 ? kSuperAtomBytes : kSuperAtomBytesBf16t);
}

#if defined(__CUDACC__)

__device__ __forceinline__ int64_t moe_in_row(const int32_t* __restrict__ gather, int64_t gather_div, int p) {
  return gather != nullptr ? static_cast<int64_t>(__ldg(gather + p)) / gather_div : p;
}

__device__ __forceinline__ int64_t moe_out_row(const int32_t* __restrict__ scatter, int p) {
  return scatter != nullptr ? static_cast<int64_t>(__ldg(scatter + p)) : p;
}

// Escape indices are local to this expert's slice. Offsets are absolute,
// so escape_offsets[0] need not be zero. num_escapes bounds prefetches.
struct MoeExpertCodec {
  uint32_t base_x4;
  const uint4* escapes;
  uint32_t num_escapes;
};

// One expert's escape slice from the absolute offset pair. The base is clamped
// to the table start so a corrupt negative offset cannot point before the
// allocation, and a decreasing pair yields an empty slice. The local index
// bound (lo0 < num_escapes) stays in the decode helpers, matching the HIP
// kernels; the loader owns the remaining upper-bound validation.
__device__ __forceinline__ MoeExpertCodec moe_escape_slice(const uint4* __restrict__ escapes,
                                                          const int64_t* __restrict__ escape_offsets, int e) {
  if (escapes == nullptr) return {0u, nullptr, 0u};
  const int64_t lo = __ldg(escape_offsets + e);
  const int64_t hi = __ldg(escape_offsets + e + 1);
  const int64_t base = lo > 0 ? lo : 0;
  return {0u, escapes + base, static_cast<uint32_t>(hi > base ? hi - base : 0)};
}

template <WeightSource S>
__device__ __forceinline__ MoeExpertCodec moe_expert_codec(const uint8_t* __restrict__ exp_base,
                                                           const uint4* escapes,
                                                           const int64_t* __restrict__ escape_offsets, int e) {
  if constexpr (S == WeightSource::kNF12) {
    // A null table means no expert has escapes.
    MoeExpertCodec slice = moe_escape_slice(escapes, escape_offsets, e);
    slice.base_x4 = exp_base_x4(__ldg(exp_base + e));
    return slice;
  } else {
    return {0u, nullptr, 0u};
  }
}

// Prefetch before the routing scan to avoid serializing decode loads behind it.
// This may precede griddepcontrol.wait because prefetch returns no data and
// L2 remains coherent with earlier writers.
template <WeightSource S>
__device__ __forceinline__ void moe_prefetch_codec(const uint8_t* exp_base, const uint4* escapes,
                                                   const int64_t* escape_offsets, int num_experts, int tid,
                                                   int nthreads) {
  if constexpr (S == WeightSource::kNF12) {
    const int base_lines = (num_experts + 127) / 128;
    const int lines = base_lines + (escapes != nullptr ? ((num_experts + 1) * 8 + 127) / 128 : 0);
    for (int i = tid; i < lines; i += nthreads) {
      const uint8_t* line = i < base_lines ? exp_base + i * 128
                                           : reinterpret_cast<const uint8_t*>(escape_offsets) + (i - base_lines) * 128;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
    }
  }
}

#endif  // __CUDACC__

}  // namespace nf12::detail
