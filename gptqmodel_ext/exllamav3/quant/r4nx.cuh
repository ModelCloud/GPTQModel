// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// R4NX codebook by ModelCloud.ai.
//
// R4NX reuses the Q4NX tile layout, kv-bit states and 64 zero-prefixed states
// exactly, and changes only the state -> weight map: the four int8 levels are
// the four bytes of `state * R4NX_MULTIPLIER`, scaled by R4NX_ALPHA. That is
// one 32-bit multiply and one byte permute per four weights instead of Q4NX's
// multiply plus per-nibble `((x >> 4) & 15) + (x & 15)` add.
#pragma once

#include <cstdint>

#include "q4nx.cuh"

constexpr int EXL3_R4NX_CODEBOOK = 4;
constexpr float EXL3_R4NX_ALPHA = 0.013531750173586456f;  // 1 / sqrt(5461.25)

constexpr bool r4nx_supports_step_bits(int64_t step_bits)
{
    return step_bits == 10 || step_bits == 12 || step_bits == 14 || step_bits == 16;
}

// Q4NX and R4NX share the packed layout and the NX decode path, so the kernels
// branch on the family rather than on the individual codebook.
constexpr bool exl3_is_nx_codebook(int cb)
{
    return cb == EXL3_Q4NX_CODEBOOK || cb == EXL3_R4NX_CODEBOOK;
}

template <int cb>
constexpr float exl3_nx_alpha()
{
    return cb == EXL3_R4NX_CODEBOOK ? EXL3_R4NX_ALPHA : EXL3_Q4NX_ALPHA;
}

constexpr bool exl3_nx_supports_step_bits(int cb, int64_t step_bits)
{
    return cb == EXL3_R4NX_CODEBOOK ? r4nx_supports_step_bits(step_bits) : q4nx_supports_step_bits(step_bits);
}

__device__ __forceinline__ uint32_t r4nx_bytes(uint32_t state)
{
    return state * 0x8095576Du;
}
