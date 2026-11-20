// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// R4NX: an RDNA4-native NX-family trellis codebook.
//
// R4NX reuses the Q4NX trellis packing unchanged (same tile layout, same
// kv-bit states, the same 64 zero-prefixed states per tile) and replaces the
// level map with a single 32-bit multiply followed by signed-byte extraction:
//
//     product  = state * EXL3_R4NX_MULTIPLIER
//     level[c] = int8(product >> (8 * c)) * EXL3_R4NX_ALPHA     (c = 0..3)
//
// The levels are returned in EXL3 fragment order. On RDNA4 that is one
// v_mul_lo_u32 plus one v_perm_b32 per pair of weights; there is no add, no
// nibble split and no lookup table (Q4NX needs the multiply and a per-nibble
// add before its -15 bias).
//
// The multiplier is odd, so state -> product is a bijection modulo 2^32 and
// state -> four signed bytes is injective: every trellis state names exactly
// one four-weight fragment, and the codebook has no aliased states.
//
// Both constants come from the search documented in
// gptqmodel/exllamav3/r4nx.py, which is the reference implementation and the
// oracle for this file. 512 seeded odd multipliers were screened at all four
// rates by nearest-codepoint MSE on fixed N(0, 1) 4-tuples; the winner
// 0xBC707477 costs 0.008673 MSE/weight at 4.0 bpw on 65536 held-out tuples,
// against 0.008240 for the Q4NX nibble-sum map at the same bpw.
//
// EXL3_R4NX_ALPHA is 7 / 512, exactly representable in binary16. Every
// level * ALPHA product is an integer of at most 10 significant bits times a
// power of two, so it is exact in both binary32 and binary16: the half2
// multiply below cannot round differently from the FP32 reference multiply
// followed by a cast, which is what makes the HIP decode bit-exact against the
// torch oracle.
#pragma once

#include <cstdint>

constexpr int EXL3_R4NX_CODEBOOK = 4;
constexpr uint32_t EXL3_R4NX_MULTIPLIER = 0xBC707477u;
constexpr float EXL3_R4NX_ALPHA = 0.013671875f;  // 7 / 512

// Byte-gather selectors for __byte_perm: the two low bytes, then the two high
// bytes, each placed in its own 16-bit lane.
constexpr uint32_t EXL3_R4NX_PAIR_01 = 0x5140u;
constexpr uint32_t EXL3_R4NX_PAIR_23 = 0x5342u;

constexpr bool r4nx_supports_step_bits(int64_t step_bits)
{
    return step_bits == 10 || step_bits == 12 || step_bits == 14 || step_bits == 16;
}

// Integer part of the decode: one 32-bit multiply (v_mul_lo_u32).
__device__ __forceinline__ uint32_t r4nx_product(uint32_t state)
{
    return state * EXL3_R4NX_MULTIPLIER;
}

// Second integer op: gather two of the product's bytes, one per 16-bit lane
// (v_perm_b32 on RDNA4).
__device__ __forceinline__ uint32_t r4nx_byte_pair(uint32_t product, uint32_t selector)
{
    return __byte_perm(product, 0u, selector);
}
