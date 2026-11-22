// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel::sfpu {

constexpr uint32_t tile_vectors = 32;
constexpr auto u16 = sfpi::DataLayout::U16;

template <unsigned offset>
inline sfpi::vUInt decode(sfpi::vUInt lows, sfpi::vUInt meta, sfpi::vUInt base_bits) {
    const auto low = (lows >> (8 * (offset % 2))) & 0xffu;
    const auto delta = (meta >> (4 * offset)) & 7u;
    const auto sign = ((meta >> (4 * offset)) & 8u) << 12u;
    return ((delta << 8u) + base_bits) | sign | low;
}

void decode_face(uint32_t base) {
    const sfpi::vUInt base_bits = base << 8;
#pragma GCC unroll 8
    for (int vector = 0; vector < 8; ++vector) {
        const sfpi::vUInt lows01 = sfpi::dst_reg[vector].mode<u16>();
        const sfpi::vUInt lows23 = sfpi::dst_reg[tile_vectors + vector].mode<u16>();
        const sfpi::vUInt meta = sfpi::dst_reg[2 * tile_vectors + vector].mode<u16>();
        sfpi::dst_reg[vector].mode<u16>() = decode<0>(lows01, meta, base_bits);
        sfpi::dst_reg[tile_vectors + vector].mode<u16>() = decode<1>(lows01, meta, base_bits);
        sfpi::dst_reg[2 * tile_vectors + vector].mode<u16>() = decode<2>(lows23, meta, base_bits);
        sfpi::dst_reg[3 * tile_vectors + vector].mode<u16>() = decode<3>(lows23, meta, base_bits);
    }
}

}  // namespace ckernel::sfpu
#endif
