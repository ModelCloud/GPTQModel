// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// Q4NX codebook by AlexBrandes (Alex Brandes), from AlexBrandes/exllamav3.
#pragma once

#include <cstdint>

constexpr int EXL3_Q4NX_CODEBOOK = 3;
constexpr float EXL3_Q4NX_ALPHA = 0.1533929977694741f;  // 1 / sqrt(42.5)

constexpr bool q4nx_supports_step_bits(int64_t step_bits)
{
    return step_bits == 10 || step_bits == 12 || step_bits == 14;
}

__device__ __forceinline__ uint32_t q4nx_nibble_sums(uint32_t state)
{
    const uint32_t product = state * 0x83DCD12Du;
    return ((product >> 4) & 0x0f0f0f0fu) + (product & 0x0f0f0f0fu);
}
