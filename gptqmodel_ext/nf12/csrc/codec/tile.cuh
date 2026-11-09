// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 / NF12-AD kernel computation.
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
// modified for the NF12-AD (AlpinDale) method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
// Lane (g,t) = (l>>2,l&3) owns 16-byte chunks t and t+4 in rows g and g+8.
// A quad transpose converts row-major chunks to the group registers in SPEC.md 4.1.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include <nf12/format.cuh>

namespace nf12::codec {

inline constexpr int kWarpsPerBlock = 8;
inline constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
inline constexpr uint32_t kFullMask = 0xffffffffu;

struct SuperAtom {
  int64_t nb, kb;
};
__device__ __forceinline__ SuperAtom super_atom(int64_t sa, int64_t k_blocks) {
  return SuperAtom{sa / k_blocks, sa % k_blocks};
}

// x[h][p]: 16-byte chunk (t + 4p) of row (g + 8h) of the super-atom.
using RowTile = uint4[2][2];
// r[j][i]: register R_i of the group of atom j held by this lane.
using AtomRegs = uint32_t[4][4];

// Element offset of this lane's chunk (h = 0, p = 0) within the matrix.
__device__ __forceinline__ int64_t row_tile_offset(SuperAtom s, int64_t ld, int lane) {
  return (s.nb * kSuperAtomRows + (lane >> 2)) * ld + s.kb * kSuperAtomCols + 8 * (lane & 3);
}

__device__ __forceinline__ void load_row_tile(const __nv_bfloat16* __restrict__ w, int64_t off, int64_t ld,
                                              RowTile& x) {
  const uint4* p = reinterpret_cast<const uint4*>(w + off);
  const int64_t row8 = ld;  // 8 rows of uint4 = 8 * ld * 2 bytes / 16
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int q = 0; q < 2; ++q) x[h][q] = __ldcs(p + h * row8 + 4 * q);
  }
}

__device__ __forceinline__ void store_row_tile(__nv_bfloat16* __restrict__ out, int64_t off, int64_t ld,
                                               const RowTile& x) {
  uint4* p = reinterpret_cast<uint4*>(out + off);
  const int64_t row8 = ld;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int q = 0; q < 2; ++q) __stcs(p + h * row8 + 4 * q, x[h][q]);
  }
}

// Swap lane bits 0-1 with the vector-component index using four shuffles.
// This transpose is its own inverse.
__device__ __forceinline__ void quad_transpose(uint4& v, int lane) {
  const bool b0 = lane & 1;
  uint32_t s0 = b0 ? v.x : v.y;
  uint32_t s1 = b0 ? v.z : v.w;
  s0 = __shfl_xor_sync(kFullMask, s0, 1);
  s1 = __shfl_xor_sync(kFullMask, s1, 1);
  if (b0) {
    v.x = s0;
    v.z = s1;
  } else {
    v.y = s0;
    v.w = s1;
  }
  const bool b1 = lane & 2;
  uint32_t u0 = b1 ? v.x : v.z;
  uint32_t u1 = b1 ? v.y : v.w;
  u0 = __shfl_xor_sync(kFullMask, u0, 2);
  u1 = __shfl_xor_sync(kFullMask, u1, 2);
  if (b1) {
    v.x = u0;
    v.y = u1;
  } else {
    v.z = u0;
    v.w = u1;
  }
}

// After quad_transpose, component c of x[h][p] is column pair (4 * (4p + c) + t)
// of row g + 8h. Atom j needs pairs 8j + t (R0/R1) and 8j + 4 + t (R2/R3),
// i.e. p = j >> 1 and c = 2 * (j & 1) + {0, 1}.
__device__ __forceinline__ void tile_to_atoms(const RowTile& x, AtomRegs& r) {
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    r[2 * p][0] = x[0][p].x;
    r[2 * p][2] = x[0][p].y;
    r[2 * p][1] = x[1][p].x;
    r[2 * p][3] = x[1][p].y;
    r[2 * p + 1][0] = x[0][p].z;
    r[2 * p + 1][2] = x[0][p].w;
    r[2 * p + 1][1] = x[1][p].z;
    r[2 * p + 1][3] = x[1][p].w;
  }
}

__device__ __forceinline__ void atoms_to_tile(const AtomRegs& r, RowTile& x) {
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    x[0][p] = make_uint4(r[2 * p][0], r[2 * p][2], r[2 * p + 1][0], r[2 * p + 1][2]);
    x[1][p] = make_uint4(r[2 * p][1], r[2 * p][3], r[2 * p + 1][1], r[2 * p + 1][3]);
  }
}

__device__ __forceinline__ void load_atoms(const __nv_bfloat16* __restrict__ w, int64_t ldw, SuperAtom s, int lane,
                                           AtomRegs& r) {
  RowTile x;
  load_row_tile(w, row_tile_offset(s, ldw, lane), ldw, x);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int p = 0; p < 2; ++p) quad_transpose(x[h][p], lane);
  }
  tile_to_atoms(x, r);
}

// High bytes of the group's slots, split by pair position: byte j of `a` is
// hi(s(2j)), byte j of `b` is hi(s(2j+1)), matching the meta byte order.
struct GroupHigh {
  uint32_t a, b;
};
__device__ __forceinline__ GroupHigh group_high(const uint32_t (&r)[4]) {
  const uint32_t p01 = __byte_perm(r[0], r[1], 0x7531);
  const uint32_t p23 = __byte_perm(r[2], r[3], 0x7531);
  return GroupHigh{__byte_perm(p01, p23, 0x6420), __byte_perm(p01, p23, 0x7531)};
}

// Setting each byte's high bit prevents borrowing when subtracting B <= 120.
// In-window bytes become [0x80,0x87], so bits 3-6 detect out-of-window values.
// A zero meta word requires an escape even when every value is in-window.
__device__ __forceinline__ bool encode_group_fast(const uint32_t (&r)[4], uint32_t base_x4, Group& out) {
  const GroupHigh hi = group_high(r);
  const uint32_t da = ((hi.a & 0x7f7f7f7fu) | 0x80808080u) - base_x4;
  const uint32_t db = ((hi.b & 0x7f7f7f7fu) | 0x80808080u) - base_x4;
  out.meta = (hi.a & 0x80808080u) | ((hi.b & 0x80808080u) >> 1) | ((db & 0x07070707u) << 3) | (da & 0x07070707u);
  out.lo0 = __byte_perm(r[0], r[1], 0x6420);
  out.lo1 = __byte_perm(r[2], r[3], 0x6420);
  return ((da | db) & 0x78787878u) == 0u && out.meta != 0u;
}

// With h = e>>1, all values fit for B in [hmax-7,hmin], clamped to [0,120].
// A sentinel collision requires B = hmin = hmax and all sign bits clear.
struct GroupStats {
  uint32_t hmin, hmax;
  bool all_positive;
};
__device__ __forceinline__ GroupStats group_stats(const uint32_t (&r)[4]) {
  const GroupHigh hi = group_high(r);
  const uint32_t ha = hi.a & 0x7f7f7f7fu;
  const uint32_t hb = hi.b & 0x7f7f7f7fu;
  uint32_t mn = __vminu4(ha, hb);
  uint32_t mx = __vmaxu4(ha, hb);
  mn = __vminu4(mn, mn >> 16);
  mx = __vmaxu4(mx, mx >> 16);
  mn = __vminu4(mn, mn >> 8);
  mx = __vmaxu4(mx, mx >> 8);
  return GroupStats{mn & 0xffu, mx & 0xffu, ((hi.a | hi.b) & 0x80808080u) == 0u};
}

}
