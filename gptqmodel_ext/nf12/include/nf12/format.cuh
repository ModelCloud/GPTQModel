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
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define NF12_HD __host__ __device__ __forceinline__
#else
#define NF12_HD inline
#endif

namespace nf12 {

// Tiled layout in SPEC.md section 4.
inline constexpr int kAtomRows = 16;
inline constexpr int kAtomCols = 16;
inline constexpr int kAtomsPerSuperAtom = 4;
inline constexpr int kSuperAtomRows = 16;
inline constexpr int kSuperAtomCols = 64;
inline constexpr int kSuperAtomBytes = 1536;
inline constexpr int kSuperAtomBytesBf16t = 2048;
inline constexpr int kPlaneBytes = 512;
inline constexpr int kMaxExpBase = 120;

struct Group {
  uint32_t meta;
  uint32_t lo0;
  uint32_t lo1;
};

// Slot -> (row, col) offset within an atom for lane l (SPEC.md 4.1).
NF12_HD int slot_row(int lane, int slot) { return (lane >> 2) + ((slot >> 1) & 1) * 8; }
NF12_HD int slot_col(int lane, int slot) { return 2 * (lane & 3) + (slot & 1) + (slot >> 2) * 8; }

NF12_HD uint32_t exp_base_x4(uint32_t exp_base) { return exp_base * 0x01010101u; }

#if defined(__CUDACC__)

__device__ __forceinline__ uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t sel) {
  return __byte_perm(x, y, sel);
}

// Requires meta != 0 and base_x4 = exp_base * 0x01010101.
__device__ __forceinline__ void decode_bulk(const Group& g, uint32_t base_x4, uint32_t (&r)[4]) {
  const uint32_t m = g.meta;
  const uint32_t a = (m & 0x87878787u) + base_x4;
  const uint32_t b = (((m << 1) & 0x80808080u) | ((m >> 3) & 0x07070707u)) + base_x4;
  const uint32_t p0 = byte_perm(a, b, 0x5140);
  const uint32_t p1 = byte_perm(a, b, 0x7362);
  r[0] = byte_perm(g.lo0, p0, 0x5140);
  r[1] = byte_perm(g.lo0, p0, 0x7362);
  r[2] = byte_perm(g.lo1, p1, 0x5140);
  r[3] = byte_perm(g.lo1, p1, 0x7362);
}

// All 32 lanes must participate in the warp-uniform escape branch.
__device__ __forceinline__ void decode_group(const Group& g, uint32_t base_x4, const uint4* __restrict__ escapes,
                                             uint32_t (&r)[4]) {
  decode_bulk(g, base_x4, r);
  const bool esc = g.meta == 0u;
  if (__any_sync(0xffffffffu, esc)) {
    if (esc) {
      const uint4 v = __ldg(escapes + g.lo0);
      r[0] = v.x;
      r[1] = v.y;
      r[2] = v.z;
      r[3] = v.w;
    }
  }
}

// Tail-safe decode without a full-warp participation requirement.
__device__ __forceinline__ void decode_group_any_lane(const Group& g, uint32_t base_x4,
                                                      const uint4* __restrict__ escapes, uint32_t (&r)[4]) {
  decode_bulk(g, base_x4, r);
  if (g.meta == 0u) {
    const uint4 v = __ldg(escapes + g.lo0);
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  }
}

#endif

// Returns false for out-of-window groups or a sentinel collision (SPEC.md 3).
NF12_HD bool encode_group_in_window(const uint16_t (&slots)[8], uint32_t exp_base, Group& out) {
  uint32_t meta = 0, lo0 = 0, lo1 = 0;
  const uint32_t lo_e = 2u * exp_base;
  for (int i = 0; i < 8; ++i) {
    const uint32_t v = slots[i];
    const uint32_t e = (v >> 7) & 0xffu;
    if (e < lo_e || e > lo_e + 15u) return false;
    const uint32_t s = v >> 15;
    const uint32_t o1 = (e - lo_e) >> 1;
    const int j = i >> 1;
    const uint32_t bits = (i & 1) ? ((s << 6) | (o1 << 3)) : ((s << 7) | o1);
    meta |= bits << (8 * j);
    const uint32_t lo = v & 0xffu;
    if (i < 4)
      lo0 |= lo << (8 * i);
    else
      lo1 |= lo << (8 * (i - 4));
  }
  if (meta == 0u) return false;  // sentinel collision
  out = Group{meta, lo0, lo1};
  return true;
}

}
