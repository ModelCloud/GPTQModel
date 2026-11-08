// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12-AD (NyoomFloat12) format + decode for AMD RDNA4 (gfx1200/gfx1200/gfx1250).
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original NF12 reference implementation and CUDA kernels:
//   https://github.com/dphnAI/nf12
//   Copyright (c) 2026 AlpinDale and the nf12 contributors.
//   Licensed under the Apache License, Version 2.0.
//
// This HIP port was written for GPT-QModel (ModelCloud.ai). It decodes the
// identical NF12-AD tiled byte stream that the CUDA kernels produce, so a
// checkpoint is portable between NVIDIA and AMD hosts. The bit layout is
// specified in gptqmodel_ext/nf12/docs/SPEC.md sections 2-4.
//
// PORTING NOTES (NVIDIA -> AMD), all verified on gfx1201 hardware:
//
//   * __byte_perm(x, y, sel) -> __builtin_amdgcn_perm(x, y, sel) is NOT a
//     transliteration. NVIDIA numbers a 4-bit nibble per output byte over the
//     concatenation [x.b0..x.b3, y.b0..y.b3]. AMD uses ONE FULL BYTE per
//     output byte, where 0..3 selects src1 (the SECOND argument) and 4..7
//     selects src0 (the FIRST). The helper prmt() below converts NVIDIA
//     nibble selectors to the AMD byte form, so the decode expressions can be
//     copied verbatim from include/nf12/format.cuh.
//   * __any_sync/warp vote -> __builtin_amdgcn_any / wave32 ballot.
//   * __ldcs / __stcs -> plain vector loads on RDNA4; there is no portable
//     cache-streaming hint that matters for these sizes.
//   * uint4 -> uint4 is available via hip_runtime.h, but alignment of the
//     packed buffer is only 16 bytes, so all vector loads stay 16-byte.
//   * __nv_bfloat16 -> __hip_bfloat16 (hip_bf16.h).
//
// The interface reads exactly as `uint4 planes[3]` per lane as the CUDA
// decoder does: lane l of a super-atom holds 16 bytes of each of the three
// 512-byte planes.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

namespace nf12_hip {

// SPEC.md section 4 geometry.
inline constexpr int kAtomRows = 16;
inline constexpr int kAtomCols = 16;
inline constexpr int kAtomsPerSuperAtom = 4;
inline constexpr int kSuperAtomRows = 16;
inline constexpr int kSuperAtomCols = 64;
inline constexpr int kSuperAtomBytes = 1536;
inline constexpr int kPlaneBytes = 512;
inline constexpr int kMaxExpBase = 120;

inline constexpr int kWarpSize = 32;
inline constexpr int kWarpsPerBlock = 8;
inline constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;
inline constexpr int kSuperAtomVecs = kSuperAtomBytes / 16;  // 96 uint4
inline constexpr int kPlaneVecs = kPlaneBytes / 16;          // 32 uint4

struct Group {
  uint32_t meta;
  uint32_t lo0;
  uint32_t lo1;
};

// Host + device: the launcher needs it on the host side too.
__host__ __device__ __forceinline__ uint32_t exp_base_x4(uint32_t exp_base) {
  return exp_base * 0x01010101u;
}

// ---------------------------------------------------------------------------
// NVIDIA __byte_perm -> AMD v_perm_b32
// ---------------------------------------------------------------------------
// `sel` uses NVIDIA nibble semantics: nibble i (bits 4i..4i+3) selects the
// source byte for output byte i, where 0..3 is x and 4..7 is y.
//
// AMD v_perm_b32 wants byte i of the selector (bits 8i..8i+7) to name a source
// byte, 0..3 -> src1, 4..7 -> src0. With NVIDIA x -> src0 and y -> src1:
//   nibble 0..3 (x.byte n)  -> AMD byte 4 + n
//   nibble 4..7 (y.byte n)  -> AMD byte n - 4
// This conversion is done at COMPILE TIME so every call site below lowers to a
// single v_perm_b32 with a literal selector.
constexpr uint32_t nv_sel_to_amd(uint32_t sel) {
  uint32_t out = 0;
  for (int i = 0; i < 4; ++i) {
    const uint32_t nib = (sel >> (4 * i)) & 0xFu;
    const uint32_t byte = (nib < 4u) ? (4u + nib) : (nib - 4u);
    out |= byte << (8 * i);
  }
  return out;
}

template <uint32_t kNvSel>
__device__ __forceinline__ uint32_t prmt(uint32_t x, uint32_t y) {
  return __builtin_amdgcn_perm(x, y, nv_sel_to_amd(kNvSel));
}

// ---------------------------------------------------------------------------
// Decode (SPEC.md 3 "Reference decode")
// ---------------------------------------------------------------------------
// Requires meta != 0 and base_x4 == exp_base * 0x01010101.
__device__ __forceinline__ void decode_bulk(const Group& g, uint32_t base_x4, uint32_t (&r)[4]) {
  const uint32_t m = g.meta;
  const uint32_t a = (m & 0x87878787u) + base_x4;
  const uint32_t b = (((m << 1) & 0x80808080u) | ((m >> 3) & 0x07070707u)) + base_x4;
  const uint32_t p0 = prmt<0x5140u>(a, b);  // hi(s0), hi(s1), hi(s2), hi(s3)
  const uint32_t p1 = prmt<0x7362u>(a, b);  // hi(s4) .. hi(s7)
  r[0] = prmt<0x5140u>(g.lo0, p0);
  r[1] = prmt<0x7362u>(g.lo0, p0);
  r[2] = prmt<0x5140u>(g.lo1, p1);
  r[3] = prmt<0x7362u>(g.lo1, p1);
}

// Every lane of the wave must reach this call; the escape branch is
// wave-uniform on the fast path.
__device__ __forceinline__ void decode_group(const Group& g, uint32_t base_x4,
                                             const uint4* __restrict__ escapes,
                                             uint32_t (&r)[4]) {
  decode_bulk(g, base_x4, r);
  if (g.meta == 0u) {
    const uint4 v = escapes[g.lo0];
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  }
}

// ---------------------------------------------------------------------------
// Tiling (SPEC.md 4.1 / 4.2)
// ---------------------------------------------------------------------------
// x[h][p]: 16-byte chunk (t + 4p) of row (g + 8h) of the super-atom.
using RowTile = uint4[2][2];
// r[j][i]: register R_i of the group of atom j held by this lane.
using AtomRegs = uint32_t[4][4];

struct SuperAtom {
  int64_t nb, kb;
};

__device__ __forceinline__ SuperAtom super_atom(int64_t sa, int64_t k_blocks) {
  return SuperAtom{sa / k_blocks, sa % k_blocks};
}

__device__ __forceinline__ int64_t row_tile_offset(SuperAtom s, int64_t ld, int lane) {
  return (s.nb * kSuperAtomRows + (lane >> 2)) * ld + s.kb * kSuperAtomCols + 8 * (lane & 3);
}

__device__ __forceinline__ void load_row_tile(const __hip_bfloat16* __restrict__ w,
                                              int64_t off, int64_t ld, RowTile& x) {
  const uint4* p = reinterpret_cast<const uint4*>(w + off);
  const int64_t row8 = ld;  // 8 rows of uint4
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int q = 0; q < 2; ++q) x[h][q] = p[h * row8 + 4 * q];
  }
}

__device__ __forceinline__ void store_row_tile(__hip_bfloat16* __restrict__ out,
                                               int64_t off, int64_t ld, const RowTile& x) {
  uint4* p = reinterpret_cast<uint4*>(out + off);
  const int64_t row8 = ld;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int q = 0; q < 2; ++q) p[h * row8 + 4 * q] = x[h][q];
  }
}

// Swap lane bits 0-1 with the vector-component index using four shuffles.
// Self-inverse. Wave32 on RDNA4, so xor masks 1 and 2 stay inside the wave.
__device__ __forceinline__ void quad_transpose(uint4& v, int lane) {
  const bool b0 = lane & 1;
  uint32_t s0 = b0 ? v.x : v.y;
  uint32_t s1 = b0 ? v.z : v.w;
  s0 = __shfl_xor(s0, 1);
  s1 = __shfl_xor(s1, 1);
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
  u0 = __shfl_xor(u0, 2);
  u1 = __shfl_xor(u1, 2);
  if (b1) {
    v.x = u0;
    v.y = u1;
  } else {
    v.z = u0;
    v.w = u1;
  }
}

// After quad_transpose, component c of x[h][p] is column pair (4*(4p+c)+t) of
// row g+8h. Atom j needs pairs 8j+t and 8j+4+t, i.e. p = j>>1 and
// c = 2*(j&1) + {0,1}.
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

}  // namespace nf12_hip
