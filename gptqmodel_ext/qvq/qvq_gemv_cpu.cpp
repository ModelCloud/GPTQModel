// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/Parallel.h>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <immintrin.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "pgc16_cpu_tables.h"

namespace qvq_cpu {
namespace {

constexpr uint32_t PGC16_MULTIPLIER = 40503;
constexpr uint32_t PGC16_INCREMENT = 17011;

alignas(64) float g_levels_float[256];
std::once_flag g_tables_once;

inline uint16_t pgc16_mix_state(uint16_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * PGC16_MULTIPLIER + PGC16_INCREMENT) & 0xFFFF;
  return static_cast<uint16_t>(mixed ^ (mixed >> 7));
}

void pgc16_build_tables() {
  for (int i = 0; i < 256; ++i) {
    c10::Half h = c10::Half(kPgc16LevelBits[i], c10::Half::from_bits());
    g_levels_float[i] = static_cast<float>(h);
  }
}

inline void pgc16_ensure_tables() { std::call_once(g_tables_once, pgc16_build_tables); }

// XOR masks for banked V2. Mirrors PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS.
static const uint16_t kV2B4BankMasks[13][4] = {
    /*  2 */ {0x0000, 0xA5A5, 0x5A5A, 0x3C3C},
    /*  3 */ {0x0000, 0xA5A5, 0x9696, 0x6969},
    /*  4 */ {0x0000, 0x5A5A, 0x3C3C, 0xC3C3},
    /*  5 */ {0x0000, 0x9696, 0x3C3C, 0xC3C3},
    /*  6 */ {0x0000, 0x6969, 0x5A5A, 0x3C3C},
    /*  7 */ {0x0000, 0xC3C3, 0x9696, 0x5A5A},
    /*  8 */ {0x0000, 0x0000, 0x0000, 0x0000},
    /*  9 */ {0x0000, 0x0000, 0x0000, 0x0000},
    /* 10 */ {0x0000, 0x0000, 0x0000, 0x0000},
    /* 11 */ {0x0000, 0x0000, 0x0000, 0x0000},
    /* 12 */ {0x0000, 0x0000, 0x0000, 0x0000},
    /* 13 */ {0x0000, 0x0000, 0x0000, 0x0000},
    /* 14 */ {0x0000, 0x0000, 0x0000, 0x0000},
};

inline bool cpu_has_avx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}

struct Plane {
  int width;
  int offset;
};

inline int planes_for_width(int bits, Plane* planes) {
  int count = 0;
  int remaining = bits;
  int offset = 0;
  while (remaining > 0) {
    int width = 1;
    while ((width << 1) <= remaining) width <<= 1;
    planes[count++] = {width, offset};
    remaining -= width;
    offset += width;
  }
  return count;
}

alignas(64) static const int32_t kLaneId[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

// Unpack one tile's 128 transition-bit code words from the planar int32 payload.
void unpack_tile_codes(const int32_t* tile_words, int E, uint16_t* codes) {
  Plane planes[5];
  int nplanes = planes_for_width(E, planes);
  std::memset(codes, 0, 128 * sizeof(uint16_t));
  for (int block = 0; block < 4; ++block) {
    int block_base = block * E;
    int word_offset = 0;
    for (int p = 0; p < nplanes; ++p) {
      int w = planes[p].width;
      int o = planes[p].offset;
      int pack_factor = 32 / w;
      uint32_t mask = (1u << w) - 1;
      for (int pw = 0; pw < w; ++pw) {
        uint32_t word = static_cast<uint32_t>(tile_words[block_base + word_offset + pw]);
        for (int k = 0; k < pack_factor; ++k) {
          int idx = block * 32 + pw * pack_factor + k;
          codes[idx] |= static_cast<uint16_t>(((word >> (k * w)) & mask) << o);
        }
      }
      word_offset += w;
    }
  }
}

__attribute__((always_inline)) __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void unpack_tile_codes_avx512(const int32_t* tile_words, int E, uint16_t* codes) {
  Plane planes[5];
  int nplanes = planes_for_width(E, planes);
  const __m512i lane_id = _mm512_load_si512(reinterpret_cast<const __m512i*>(kLaneId));
  for (int block = 0; block < 4; ++block) {
    __m512i block_codes = _mm512_setzero_si512();
    int block_base = block * E;
    int word_offset = 0;
    for (int p = 0; p < nplanes; ++p) {
      const int w = planes[p].width;
      const int o = planes[p].offset;
      const int log2_pf = 5 - __builtin_ctz(w);
      const int pf_mask = (1 << log2_pf) - 1;
      const int half_w = w / 2;
      const __mmask16 load_mask = static_cast<__mmask16>((1u << w) - 1u);
      const __m512i src = _mm512_maskz_loadu_epi32(
          load_mask, tile_words + block_base + word_offset);

      __m512i plane_codes = _mm512_setzero_si512();
      for (int h = 0; h < 2; ++h) {
        __m512i idx = _mm512_srli_epi32(lane_id, log2_pf);
        if (h == 1) {
          const __m512i half_offset = _mm512_set1_epi32(half_w);
          idx = _mm512_add_epi32(idx, half_offset);
        }

        int extra_shift = (h == 1 && w == 1) ? 16 : 0;
        __m512i shift = _mm512_and_epi32(lane_id, _mm512_set1_epi32(pf_mask));
        shift = _mm512_mullo_epi32(shift, _mm512_set1_epi32(w));
        if (extra_shift) {
          shift = _mm512_add_epi32(shift, _mm512_set1_epi32(extra_shift));
        }

        __m512i vals = _mm512_permutexvar_epi32(idx, src);
        vals = _mm512_srlv_epi32(vals, shift);
        vals = _mm512_and_epi32(vals, _mm512_set1_epi32((1u << w) - 1u));

        const __m256i vals16 = _mm512_cvtepi32_epi16(vals);
        if (h == 0) {
          plane_codes = _mm512_castsi256_si512(vals16);
        } else {
          plane_codes = _mm512_inserti64x4(plane_codes, vals16, 1);
        }
      }
      plane_codes = _mm512_slli_epi16(plane_codes, o);
      block_codes = _mm512_or_si512(block_codes, plane_codes);
      word_offset += w;
    }
    _mm512_storeu_si512(
        reinterpret_cast<__m512i*>(codes + block * 32), block_codes);
  }
}

template <int Width, int Offset>
__attribute__((always_inline))
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void unpack_plane_avx512(
    const int32_t* tile_words,
    int block_base,
    const __m512i lane_id,
    __m512i& block_codes) {
  constexpr int kLog2PackFactor = 5 - __builtin_ctz(Width);
  constexpr int kPackFactorMask = (1 << kLog2PackFactor) - 1;
  constexpr int kHalfWidth = Width / 2;
  constexpr __mmask16 kLoadMask = static_cast<__mmask16>((1u << Width) - 1u);
  constexpr uint32_t kValueMask = (1u << Width) - 1u;
  const __m512i src = _mm512_maskz_loadu_epi32(
      kLoadMask, tile_words + block_base + Offset);
  __m512i plane_codes = _mm512_setzero_si512();

  for (int half = 0; half < 2; ++half) {
    __m512i index = _mm512_srli_epi32(lane_id, kLog2PackFactor);
    if (half == 1) {
      index = _mm512_add_epi32(index, _mm512_set1_epi32(kHalfWidth));
    }
    __m512i shift = _mm512_and_epi32(lane_id, _mm512_set1_epi32(kPackFactorMask));
    shift = _mm512_mullo_epi32(shift, _mm512_set1_epi32(Width));
    if constexpr (Width == 1) {
      if (half == 1) {
        shift = _mm512_add_epi32(shift, _mm512_set1_epi32(16));
      }
    }
    __m512i values = _mm512_permutexvar_epi32(index, src);
    values = _mm512_srlv_epi32(values, shift);
    values = _mm512_and_epi32(values, _mm512_set1_epi32(kValueMask));
    const __m256i values16 = _mm512_cvtepi32_epi16(values);
    if (half == 0) {
      plane_codes = _mm512_castsi256_si512(values16);
    } else {
      plane_codes = _mm512_inserti64x4(plane_codes, values16, 1);
    }
  }
  block_codes = _mm512_or_si512(
      block_codes, _mm512_slli_epi16(plane_codes, Offset));
}

template <int Remaining, int Offset>
struct UnpackPlanesAvx512 {
  __attribute__((always_inline))
  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
  static inline void run(
      const int32_t* tile_words,
      int block_base,
      const __m512i lane_id,
      __m512i& block_codes) {
    constexpr int width =
        Remaining >= 16 ? 16 : Remaining >= 8 ? 8 : Remaining >= 4 ? 4 : Remaining >= 2 ? 2 : 1;
    unpack_plane_avx512<width, Offset>(tile_words, block_base, lane_id, block_codes);
    UnpackPlanesAvx512<Remaining - width, Offset + width>::run(
        tile_words, block_base, lane_id, block_codes);
  }
};

template <int Offset>
struct UnpackPlanesAvx512<0, Offset> {
  __attribute__((always_inline))
  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
  static inline void run(
      const int32_t*, int, const __m512i, __m512i&) {}
};

template <int E>
__attribute__((always_inline)) __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void unpack_tile_codes_avx512_specialized(
    const int32_t* tile_words,
    uint16_t* codes) {
  const __m512i lane_id = _mm512_load_si512(reinterpret_cast<const __m512i*>(kLaneId));
#pragma GCC unroll 4
  for (int block = 0; block < 4; ++block) {
    __m512i block_codes = _mm512_setzero_si512();
    UnpackPlanesAvx512<E, 0>::run(tile_words, block * E, lane_id, block_codes);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(codes + block * 32), block_codes);
  }
}

// Decode one tile into 256 row-major float weights using a 16-bit shift register.
// The 128 E-bit codes form a circular bitstream; each step's state is the 16-bit
// window ending at (s + 1) * E, equivalent to the gather-based decode_tile used
// before this optimization.
inline void decode_tile(
    int E,
    const uint16_t* codes,
    int segments_per_tile,
    int states_per_segment,
    const uint16_t* seg_masks,
    float* tile_weights) {
  uint16_t state = 0;
  if (E < 16) {
    // Prime the shift register with the last (16 - E) bits of the stream so the
    // first output corresponds to the original circular window.
    int n = 15 / E;
    for (int idx = 128 - n; idx < 128; ++idx) {
      state = static_cast<uint16_t>((static_cast<uint32_t>(state) << E) | static_cast<uint32_t>(codes[idx]));
    }
    uint16_t tail_mask = static_cast<uint16_t>((1u << (16 - E)) - 1u);
    state &= tail_mask;
  }
  for (int s = 0; s < 128; ++s) {
    state = static_cast<uint16_t>((static_cast<uint32_t>(state) << E) | static_cast<uint32_t>(codes[s]));
    uint16_t mask = seg_masks[s / states_per_segment];
    uint16_t mixed = pgc16_mix_state(state ^ mask);
    tile_weights[2 * s] = g_levels_float[mixed >> 8];
    tile_weights[2 * s + 1] = g_levels_float[mixed & 0xFF];
  }
}

// Build per-tile segment masks from the packed bank_ids byte.
void tile_segment_masks(
    uint8_t bank_byte,
    int E,
    int segments_per_tile,
    bool v2b2_p32,
    int bank_alt_id,
    uint16_t* seg_masks) {
  for (int seg = 0; seg < segments_per_tile; ++seg) {
    int selector = 0;
    if (v2b2_p32) {
      selector = (bank_byte >> seg) & 1;
    } else {
      selector = (bank_byte >> (seg * 2)) & 3;
    }
    if (selector == 0 || E < 2 || E > 7) {
      seg_masks[seg] = 0;
    } else {
      int idx = v2b2_p32 ? bank_alt_id : selector;
      if (idx < 0 || idx > 3) idx = 0;
      seg_masks[seg] = kV2B4BankMasks[E - 2][idx];
    }
  }
}

// Decode one tile into 256 row-major float weights using AVX-512 vector
// gathers.  The 128 E-bit codes form a circular bitstream; state_s is the
// 16-bit window ending at code s and is computed directly from the previous
// ceil(16/E) codes to break the serial table-lookup dependency chain.
__attribute__((always_inline)) __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void decode_tile_avx512(
    int E,
    const uint16_t* codes,
    int segments_per_tile,
    int states_per_segment,
    const uint16_t* seg_masks,
    float* tile_weights) {
  const __m512i perm0 = _mm512_setr_epi32(
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  const __m512i perm1 = _mm512_setr_epi32(
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

  const __m512i mult = _mm512_set1_epi32(PGC16_MULTIPLIER);
  const __m512i inc = _mm512_set1_epi32(PGC16_INCREMENT);

  int L = (E >= 16) ? 1 : ((16 + E - 1) / E);

  // Extended code buffer so every 16-step vector load can read the circular
  // tail without explicit wrap-around handling.  Max L is 8 (E == 2).
  uint16_t ext[128 + 8];
  for (int i = 0; i < 128 + L - 1; ++i) {
    int j = i - (L - 1);
    if (j < 0) j += 128;
    ext[i] = codes[j];
  }

  const __m512i and_mask = _mm512_set1_epi32(0xFFFF);
  for (int base = 0; base < 128; base += 16) {
    const uint16_t mask = seg_masks[base / states_per_segment];
    __m512i state = _mm512_setzero_si512();
    for (int i = 0; i < L; ++i) {
      const int start = base + L - 1 - i;
      const __m256i v16 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ext + start));
      __m512i v32 = _mm512_cvtepu16_epi32(v16);
      const int shift = i * E;
      if (shift) {
        v32 = _mm512_sll_epi32(v32, _mm_cvtsi32_si128(shift));
      }
      state = _mm512_or_si512(state, v32);
    }
    state = _mm512_and_si512(state, and_mask);
    state = _mm512_xor_si512(state, _mm512_set1_epi32(mask));

    // Inline the PGC16 bijection and gather from the 1 KiB half-precision
    // level table instead of a 512 KiB pre-mixed state table.
    __m512i mixed = _mm512_xor_epi32(state, _mm512_srli_epi32(state, 8));
    mixed = _mm512_add_epi32(_mm512_mullo_epi32(mixed, mult), inc);
    mixed = _mm512_and_si512(mixed, and_mask);
    mixed = _mm512_xor_epi32(mixed, _mm512_srli_epi32(mixed, 7));

    __m512i hi = _mm512_srli_epi32(mixed, 8);
    __m512i lo = _mm512_and_epi32(mixed, _mm512_set1_epi32(0xFF));
    const __m512 v0 = _mm512_i32gather_ps(hi, g_levels_float, 4);
    const __m512 v1 = _mm512_i32gather_ps(lo, g_levels_float, 4);
    const __m512 out0 = _mm512_permutex2var_ps(v0, perm0, v1);
    const __m512 out1 = _mm512_permutex2var_ps(v0, perm1, v1);
    _mm512_storeu_ps(tile_weights + 2 * base, out0);
    _mm512_storeu_ps(tile_weights + 2 * base + 16, out1);
  }
}

template <int E>
struct DecodeStateStepsAvx512 {
  template <int I, int L>
  __attribute__((always_inline))
  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
  static inline void run(
      const uint16_t* ext,
      int base,
      __m512i& state) {
    __m512i values = _mm512_cvtepu16_epi32(_mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(ext + base - I)));
    if constexpr (I == 0) {
      state = _mm512_or_si512(state, values);
    } else {
      state = _mm512_or_si512(state, _mm512_slli_epi32(values, I * E));
    }
    if constexpr (I + 1 < L) {
      DecodeStateStepsAvx512<E>::template run<I + 1, L>(ext, base, state);
    }
  }
};

template <int E>
__attribute__((always_inline)) __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void decode_tile_avx512_specialized(
    const uint16_t* codes,
    int segments_per_tile,
    int states_per_segment,
    const uint16_t* seg_masks,
    float* tile_weights) {
  constexpr int kStates = 128;
  constexpr int kL = (16 + E - 1) / E;
  constexpr int kExtendedCodes = kStates + kL - 1;
  const __m512i perm0 = _mm512_setr_epi32(
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  const __m512i perm1 = _mm512_setr_epi32(
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  const __m512i mult = _mm512_set1_epi32(PGC16_MULTIPLIER);
  const __m512i inc = _mm512_set1_epi32(PGC16_INCREMENT);
  const __m512i state_mask = _mm512_set1_epi32(0xFFFF);
  uint16_t ext[kExtendedCodes];

#pragma GCC unroll 8
  for (int i = 0; i < kExtendedCodes; ++i) {
    constexpr int kPrime = kL - 1;
    int index = i - kPrime;
    if (index < 0) {
      index += kStates;
    }
    ext[i] = codes[index];
  }

#pragma GCC unroll 8
  for (int base = 0; base < kStates; base += 16) {
    const uint16_t mask = seg_masks[base / states_per_segment];
    __m512i state = _mm512_setzero_si512();
    DecodeStateStepsAvx512<E>::template run<0, kL>(ext, base + kL - 1, state);
    state = _mm512_and_si512(state, state_mask);
    state = _mm512_xor_si512(state, _mm512_set1_epi32(mask));
    __m512i mixed = _mm512_xor_si512(state, _mm512_srli_epi32(state, 8));
    mixed = _mm512_add_epi32(_mm512_mullo_epi32(mixed, mult), inc);
    mixed = _mm512_and_si512(mixed, state_mask);
    mixed = _mm512_xor_si512(mixed, _mm512_srli_epi32(mixed, 7));
    const __m512i hi = _mm512_srli_epi32(mixed, 8);
    const __m512i lo = _mm512_and_si512(mixed, _mm512_set1_epi32(0xFF));
    const __m512 values0 = _mm512_i32gather_ps(hi, g_levels_float, 4);
    const __m512 values1 = _mm512_i32gather_ps(lo, g_levels_float, 4);
    _mm512_storeu_ps(
        tile_weights + 2 * base,
        _mm512_permutex2var_ps(values0, perm0, values1));
    _mm512_storeu_ps(
        tile_weights + 2 * base + 16,
        _mm512_permutex2var_ps(values0, perm1, values1));
  }
}

// One output tile accumulation (M == 1, AVX-512).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void accumulate_tile_m1_avx512(
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  __m512 acc = _mm512_setzero_ps();
  alignas(64) float tile_weights[256];
  for (int tr = 0; tr < I; ++tr) {
    int tile_idx = tr * O + tc;
    const int32_t* tile_words = trellis + tile_idx * (4 * E);
    alignas(64) uint16_t codes[128];
    unpack_tile_codes_avx512(tile_words, E, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ids) {
      tile_segment_masks(bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }
    decode_tile_avx512(E, codes, segments_per_tile, states_per_segment, seg_masks, tile_weights);
    const float* x_row = x + tr * 16;
    for (int i = 0; i < 16; ++i) {
      __m512 w = _mm512_loadu_ps(tile_weights + i * 16);
      __m512 xv = _mm512_set1_ps(x_row[i]);
      acc = _mm512_fmadd_ps(w, xv, acc);
    }
  }
  _mm512_storeu_ps(out + tc * 16, acc);
}

// Block of output tiles accumulation (M == 1, AVX-512).
// Processes a contiguous range of B output tiles with the input-tile loop on the
// outside.  This makes trellis/bank_id reads contiguous and exposes multiple
// independent tile decodes/FMAs for better instruction-level parallelism.
template <int B>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void accumulate_block_m1_avx512(
    int tc_start,
    int I,
    int O,
    const float* x,
    int K,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  static_assert(B >= 1 && B <= 8, "B must be 1..8");
  alignas(64) float tile_weights[B * 256];
  __m512 accv[B];
#pragma GCC unroll 8
  for (int t = 0; t < B; ++t) {
    accv[t] = _mm512_setzero_ps();
  }

  for (int tr = 0; tr < I; ++tr) {
    for (int t = 0; t < B; ++t) {
      int tc = tc_start + t;
      int tile_idx = tr * O + tc;
      const int32_t* tile_words = trellis + tile_idx * (4 * E);
      alignas(64) uint16_t codes[128];
      unpack_tile_codes_avx512(tile_words, E, codes);
      uint16_t seg_masks[8] = {0};
      if (bank_ids) {
        tile_segment_masks(bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
      }
      decode_tile_avx512(
          E,
          codes,
          segments_per_tile,
          states_per_segment,
          seg_masks,
          tile_weights + t * 256);
    }

    const float* x_row = x + tr * 16;
#pragma GCC unroll 8
    for (int t = 0; t < B; ++t) {
      __m512 a = accv[t];
      for (int i = 0; i < 16; ++i) {
        __m512 w = _mm512_loadu_ps(tile_weights + t * 256 + i * 16);
        __m512 xv = _mm512_set1_ps(x_row[i]);
        a = _mm512_fmadd_ps(w, xv, a);
      }
      accv[t] = a;
    }
  }

#pragma GCC unroll 8
  for (int t = 0; t < B; ++t) {
    _mm512_storeu_ps(out + (tc_start + t) * 16, accv[t]);
  }
}

template <int E>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void accumulate_tile_m1_avx512_specialized(
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    const int32_t* trellis,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  __m512 acc = _mm512_setzero_ps();
  alignas(64) float tile_weights[256];
  for (int tr = 0; tr < I; ++tr) {
    const int tile_idx = tr * O + tc;
    const int32_t* tile_words = trellis + tile_idx * (4 * E);
    alignas(64) uint16_t codes[128];
    unpack_tile_codes_avx512_specialized<E>(tile_words, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ids) {
      tile_segment_masks(
          bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }
    decode_tile_avx512_specialized<E>(
        codes, segments_per_tile, states_per_segment, seg_masks, tile_weights);
    const float* x_row = x + tr * 16;
    for (int i = 0; i < 16; ++i) {
      acc = _mm512_fmadd_ps(
          _mm512_loadu_ps(tile_weights + i * 16),
          _mm512_set1_ps(x_row[i]),
          acc);
    }
  }
  _mm512_storeu_ps(out + tc * 16, acc);
}

template <int B, int E>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void accumulate_block_m1_avx512_specialized(
    int tc_start,
    int I,
    int O,
    const float* x,
    int K,
    const int32_t* trellis,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  alignas(64) float tile_weights[B * 256];
  __m512 accv[B];
#pragma GCC unroll 8
  for (int t = 0; t < B; ++t) {
    accv[t] = _mm512_setzero_ps();
  }

  for (int tr = 0; tr < I; ++tr) {
#pragma GCC unroll 8
    for (int t = 0; t < B; ++t) {
      const int tc = tc_start + t;
      const int tile_idx = tr * O + tc;
      const int32_t* tile_words = trellis + tile_idx * (4 * E);
      alignas(64) uint16_t codes[128];
      unpack_tile_codes_avx512_specialized<E>(tile_words, codes);
      uint16_t seg_masks[8] = {0};
      if (bank_ids) {
        tile_segment_masks(
            bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
      }
      decode_tile_avx512_specialized<E>(
          codes,
          segments_per_tile,
          states_per_segment,
          seg_masks,
          tile_weights + t * 256);
    }

    const float* x_row = x + tr * 16;
#pragma GCC unroll 8
    for (int t = 0; t < B; ++t) {
      __m512 acc = accv[t];
#pragma GCC unroll 16
      for (int i = 0; i < 16; ++i) {
        acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(tile_weights + t * 256 + i * 16),
            _mm512_set1_ps(x_row[i]),
            acc);
      }
      accv[t] = acc;
    }
  }

#pragma GCC unroll 8
  for (int t = 0; t < B; ++t) {
    _mm512_storeu_ps(out + (tc_start + t) * 16, accv[t]);
  }
}

// One output tile accumulation (M > 1, AVX-512).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void accumulate_tile_mn_avx512(
    int64_t M,
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  // Use a float accumulator buffer; std::vector<__m512> is unsafe for over-aligned types.
  std::vector<float> acc(static_cast<size_t>(M) * 16, 0.0f);

  alignas(64) float tile_weights[256];
  for (int tr = 0; tr < I; ++tr) {
    int tile_idx = tr * O + tc;
    const int32_t* tile_words = trellis + tile_idx * (4 * E);
    alignas(64) uint16_t codes[128];
    unpack_tile_codes_avx512(tile_words, E, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ids) {
      tile_segment_masks(bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }
    decode_tile_avx512(E, codes, segments_per_tile, states_per_segment, seg_masks, tile_weights);

    for (int i = 0; i < 16; ++i) {
      __m512 w = _mm512_loadu_ps(tile_weights + i * 16);
      for (int64_t m = 0; m < M; ++m) {
        __m512 xv = _mm512_set1_ps(x[m * K + tr * 16 + i]);
        float* acc_base = acc.data() + m * 16;
        __m512 acc_v = _mm512_loadu_ps(acc_base);
        acc_v = _mm512_fmadd_ps(w, xv, acc_v);
        _mm512_storeu_ps(acc_base, acc_v);
      }
    }
  }

  for (int64_t m = 0; m < M; ++m) {
    _mm512_storeu_ps(out + m * N + tc * 16, _mm512_loadu_ps(acc.data() + m * 16));
  }
}

// Blocked M > 1 accumulation. A decoded tile is reused by a small register
// panel of input rows before the next tile is decoded.
template <int P, int B, int E>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void accumulate_block_mn_avx512_specialized(
    int64_t m_start,
    int active_rows,
    int tc_start,
    int I,
    int O,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  static_assert(P >= 1 && P <= 8, "P must be 1..8");
  static_assert(B >= 1 && B <= 8, "B must be 1..8");
  alignas(64) float tile_weights[B * 256];
  __m512 acc[P][B];
#pragma GCC unroll 8
  for (int p = 0; p < P; ++p) {
#pragma GCC unroll 8
    for (int t = 0; t < B; ++t) {
      acc[p][t] = _mm512_setzero_ps();
    }
  }

  for (int tr = 0; tr < I; ++tr) {
#pragma GCC unroll 8
    for (int t = 0; t < B; ++t) {
      const int tc = tc_start + t;
      const int tile_idx = tr * O + tc;
      const int32_t* tile_words = trellis + tile_idx * (4 * E);
      alignas(64) uint16_t codes[128];
      unpack_tile_codes_avx512_specialized<E>(tile_words, codes);
      uint16_t seg_masks[8] = {0};
      if (bank_ids) {
        tile_segment_masks(
            bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
      }
      decode_tile_avx512_specialized<E>(
          codes,
          segments_per_tile,
          states_per_segment,
          seg_masks,
          tile_weights + t * 256);
    }

#pragma GCC unroll 8
    for (int p = 0; p < P; ++p) {
      if (p < active_rows) {
        const float* x_row = x + static_cast<int64_t>(m_start + p) * K + tr * 16;
#pragma GCC unroll 8
        for (int t = 0; t < B; ++t) {
          __m512 value = acc[p][t];
#pragma GCC unroll 16
          for (int i = 0; i < 16; ++i) {
            value = _mm512_fmadd_ps(
                _mm512_loadu_ps(tile_weights + t * 256 + i * 16),
                _mm512_set1_ps(x_row[i]),
                value);
          }
          acc[p][t] = value;
        }
      }
    }
  }

#pragma GCC unroll 8
  for (int p = 0; p < P; ++p) {
    if (p < active_rows) {
#pragma GCC unroll 8
      for (int t = 0; t < B; ++t) {
        _mm512_storeu_ps(
            out + static_cast<int64_t>(m_start + p) * N + (tc_start + t) * 16,
            acc[p][t]);
      }
    }
  }
}

// Scalar fallback for one output tile.
void accumulate_tile_scalar(
    int64_t M,
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  alignas(64) float tile_weights[256];
  for (int64_t m = 0; m < M; ++m) {
    for (int j = 0; j < 16; ++j) {
      out[m * N + tc * 16 + j] = 0.0f;
    }
  }
  for (int tr = 0; tr < I; ++tr) {
    int tile_idx = tr * O + tc;
    const int32_t* tile_words = trellis + tile_idx * (4 * E);
    uint16_t codes[128];
    unpack_tile_codes(tile_words, E, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ids) {
      tile_segment_masks(bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }
    decode_tile(E, codes, segments_per_tile, states_per_segment, seg_masks, tile_weights);

    for (int i = 0; i < 16; ++i) {
      const float* w = tile_weights + i * 16;
      for (int64_t m = 0; m < M; ++m) {
        float xv = x[m * K + tr * 16 + i];
        for (int j = 0; j < 16; ++j) {
          out[m * N + tc * 16 + j] += w[j] * xv;
        }
      }
    }
  }
}

using TileFn = void (*)(
    int64_t M,
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out);

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void tile_avx512_dispatch(
    int64_t M,
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  if (M == 1) {
    accumulate_tile_m1_avx512(
        I, O, tc, x, K, trellis, E, bank_ids, segments_per_tile,
        states_per_segment, v2b2_p32, bank_alt_id, out);
  } else {
    accumulate_tile_mn_avx512(
        M, I, O, tc, x, K, N, trellis, E, bank_ids, segments_per_tile,
        states_per_segment, v2b2_p32, bank_alt_id, out);
  }
}

void tile_dispatch(
    int64_t M,
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  if (cpu_has_avx512()) {
    tile_avx512_dispatch(
        M, I, O, tc, x, K, N, trellis, E, bank_ids, segments_per_tile,
        states_per_segment, v2b2_p32, bank_alt_id, out);
  } else {
    accumulate_tile_scalar(
        M, I, O, tc, x, K, N, trellis, E, bank_ids, segments_per_tile,
        states_per_segment, v2b2_p32, bank_alt_id, out);
  }
}

void accumulate_block_m1_dispatch(
    int tc_start,
    int I,
    int O,
    const float* x,
    int K,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  switch (E) {
    case 2:
      accumulate_block_m1_avx512_specialized<4, 2>(
          tc_start, I, O, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 3:
      accumulate_block_m1_avx512_specialized<4, 3>(
          tc_start, I, O, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 4:
      accumulate_block_m1_avx512_specialized<4, 4>(
          tc_start, I, O, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 5:
      accumulate_block_m1_avx512_specialized<4, 5>(
          tc_start, I, O, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 6:
      accumulate_block_m1_avx512_specialized<4, 6>(
          tc_start, I, O, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 7:
      accumulate_block_m1_avx512_specialized<4, 7>(
          tc_start, I, O, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 8:
      accumulate_block_m1_avx512_specialized<4, 8>(
          tc_start, I, O, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    default:
      accumulate_block_m1_avx512<4>(
          tc_start, I, O, x, K, trellis, E, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
  }
}

void accumulate_tile_m1_dispatch(
    int I,
    int O,
    int tc,
    const float* x,
    int K,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  switch (E) {
    case 2:
      accumulate_tile_m1_avx512_specialized<2>(
          I, O, tc, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 3:
      accumulate_tile_m1_avx512_specialized<3>(
          I, O, tc, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 4:
      accumulate_tile_m1_avx512_specialized<4>(
          I, O, tc, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 5:
      accumulate_tile_m1_avx512_specialized<5>(
          I, O, tc, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 6:
      accumulate_tile_m1_avx512_specialized<6>(
          I, O, tc, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 7:
      accumulate_tile_m1_avx512_specialized<7>(
          I, O, tc, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    case 8:
      accumulate_tile_m1_avx512_specialized<8>(
          I, O, tc, x, K, trellis, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
    default:
      accumulate_tile_m1_avx512(
          I, O, tc, x, K, trellis, E, bank_ids, segments_per_tile, states_per_segment,
          v2b2_p32, bank_alt_id, out);
      break;
  }
}

template <int E>
void accumulate_mn_panel_block_dispatch(
    int64_t m_start,
    int active_rows,
    int tc_start,
    int block_width,
    int I,
    int O,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  switch (block_width) {
    case 1:
      accumulate_block_mn_avx512_specialized<4, 1, E>(
          m_start, active_rows, tc_start, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 2:
      accumulate_block_mn_avx512_specialized<4, 2, E>(
          m_start, active_rows, tc_start, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 3:
      accumulate_block_mn_avx512_specialized<4, 3, E>(
          m_start, active_rows, tc_start, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    default:
      accumulate_block_mn_avx512_specialized<4, 4, E>(
          m_start, active_rows, tc_start, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
  }
}

void accumulate_mn_panel_block_dispatch(
    int64_t m_start,
    int active_rows,
    int tc_start,
    int block_width,
    int I,
    int O,
    const float* x,
    int K,
    int N,
    const int32_t* trellis,
    int E,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  switch (E) {
    case 2:
      accumulate_mn_panel_block_dispatch<2>(
          m_start, active_rows, tc_start, block_width, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 3:
      accumulate_mn_panel_block_dispatch<3>(
          m_start, active_rows, tc_start, block_width, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 4:
      accumulate_mn_panel_block_dispatch<4>(
          m_start, active_rows, tc_start, block_width, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 5:
      accumulate_mn_panel_block_dispatch<5>(
          m_start, active_rows, tc_start, block_width, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 6:
      accumulate_mn_panel_block_dispatch<6>(
          m_start, active_rows, tc_start, block_width, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 7:
      accumulate_mn_panel_block_dispatch<7>(
          m_start, active_rows, tc_start, block_width, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    case 8:
      accumulate_mn_panel_block_dispatch<8>(
          m_start, active_rows, tc_start, block_width, I, O, x, K, N, trellis, bank_ids,
          segments_per_tile, states_per_segment, v2b2_p32, bank_alt_id, out);
      break;
    default:
      for (int t = 0; t < block_width; ++t) {
        accumulate_tile_mn_avx512(
            active_rows,
            I,
            O,
            tc_start + t,
            x + m_start * K,
            K,
            N,
            trellis,
            E,
            bank_ids,
            segments_per_tile,
            states_per_segment,
            v2b2_p32,
            bank_alt_id,
            out + m_start * N);
      }
      break;
  }
}

}  // namespace

torch::Tensor qvq_gemv_cpu(
    torch::Tensor x,
    torch::Tensor trellis,
    int64_t transition_bits,
    int64_t out_features,
    c10::optional<torch::Tensor> bank_ids,
    int64_t bank_alt_id,
    bool v2b4_p64,
    bool v2b2_p32) {
  TORCH_CHECK(x.device().is_cpu(), "qvq_gemv_cpu: x must be a CPU tensor");
  TORCH_CHECK(trellis.device().is_cpu(), "qvq_gemv_cpu: trellis must be a CPU tensor");
  TORCH_CHECK(x.dim() == 2, "qvq_gemv_cpu: x must be 2D [M, K]");
  TORCH_CHECK(trellis.dim() == 2, "qvq_gemv_cpu: trellis must be 2D [tile_count, words]");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
      "qvq_gemv_cpu: x must be float32, float16, or bfloat16");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "qvq_gemv_cpu: trellis must be int32");
  TORCH_CHECK(transition_bits >= 2 && transition_bits <= 16, "qvq_gemv_cpu: transition_bits must be 2..16");
  TORCH_CHECK(!(v2b4_p64 && v2b2_p32), "qvq_gemv_cpu: v2b4_p64 and v2b2_p32 are mutually exclusive");

  int64_t M = x.size(0);
  int64_t K = x.size(1);
  int64_t N = out_features;

  TORCH_CHECK(K % 16 == 0 && N % 16 == 0, "qvq_gemv_cpu: K and N must be multiples of 16");
  int I = static_cast<int>(K / 16);
  int O = static_cast<int>(N / 16);
  int E = static_cast<int>(transition_bits);
  int words_per_tile = 4 * E;

  int64_t tile_count = static_cast<int64_t>(I) * O;
  TORCH_CHECK(trellis.size(0) == tile_count, "qvq_gemv_cpu: trellis tile_count mismatch");
  TORCH_CHECK(trellis.size(1) == words_per_tile, "qvq_gemv_cpu: trellis words_per_tile mismatch");

  if (v2b4_p64 || v2b2_p32) {
    TORCH_CHECK(bank_ids.has_value(), "qvq_gemv_cpu: banked formats require bank_ids");
    TORCH_CHECK(
        bank_ids->dim() == 1 && bank_ids->size(0) == tile_count,
        "qvq_gemv_cpu: bank_ids must be [tile_count]");
    TORCH_CHECK(
        bank_ids->scalar_type() == at::kByte,
        "qvq_gemv_cpu: bank_ids must be uint8");
    TORCH_CHECK(bank_ids->device() == x.device(), "qvq_gemv_cpu: bank_ids must be on CPU");
    if (v2b2_p32) {
      TORCH_CHECK(bank_alt_id >= 1 && bank_alt_id <= 3, "qvq_gemv_cpu: v2b2_p32 bank_alt_id must be 1..3");
    }
  }

  int segments_per_tile = 1;
  int states_per_segment = 128;
  if (v2b2_p32) {
    segments_per_tile = 8;
    states_per_segment = 16;
  } else if (v2b4_p64) {
    segments_per_tile = 4;
    states_per_segment = 32;
  }

  pgc16_ensure_tables();

  auto x_f = x.to(at::kFloat).contiguous();
  auto out = at::zeros({M, N}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

  const float* x_ptr = x_f.data_ptr<float>();
  const int32_t* trellis_ptr = trellis.data_ptr<int32_t>();
  float* out_ptr = out.data_ptr<float>();
  const uint8_t* bank_ptr = nullptr;
  if (bank_ids.has_value() && bank_ids->numel() > 0) {
    bank_ptr = bank_ids->data_ptr<uint8_t>();
  }

  int64_t num_threads = at::get_num_threads();
  constexpr int kTileBlock = 4;

  if (M == 1 && cpu_has_avx512()) {
    int64_t grain = std::max<int64_t>(kTileBlock, O / num_threads);
    at::parallel_for(0, O, grain, [&](int64_t begin, int64_t end) {
      int tc = static_cast<int>(begin);
      while (tc + kTileBlock <= static_cast<int>(end)) {
        accumulate_block_m1_dispatch(
            tc,
            I,
            O,
            x_ptr,
            static_cast<int>(K),
            trellis_ptr,
            E,
            bank_ptr,
            segments_per_tile,
            states_per_segment,
            v2b2_p32,
            static_cast<int>(bank_alt_id),
            out_ptr);
        tc += kTileBlock;
      }
      while (tc < static_cast<int>(end)) {
        accumulate_tile_m1_dispatch(
            I,
            O,
            tc,
            x_ptr,
            static_cast<int>(K),
            trellis_ptr,
            E,
            bank_ptr,
            segments_per_tile,
            states_per_segment,
            v2b2_p32,
            static_cast<int>(bank_alt_id),
            out_ptr);
        ++tc;
      }
    });
  } else if (cpu_has_avx512()) {
    constexpr int kPanel = 4;
    constexpr int kTileBlock = 4;
    const int64_t output_blocks = (O + kTileBlock - 1) / kTileBlock;
    const int64_t grain = std::max<int64_t>(1, output_blocks / num_threads);
    at::parallel_for(0, output_blocks, grain, [&](int64_t begin, int64_t end) {
      for (int64_t block = begin; block < end; ++block) {
        const int tc = static_cast<int>(block * kTileBlock);
        const int block_width = std::min(kTileBlock, O - tc);
        for (int64_t m_start = 0; m_start < M; m_start += kPanel) {
          const int active_rows = static_cast<int>(std::min<int64_t>(kPanel, M - m_start));
          accumulate_mn_panel_block_dispatch(
              m_start,
              active_rows,
              tc,
              block_width,
              I,
              O,
              x_ptr,
              static_cast<int>(K),
              static_cast<int>(N),
              trellis_ptr,
              E,
              bank_ptr,
              segments_per_tile,
              states_per_segment,
              v2b2_p32,
              static_cast<int>(bank_alt_id),
              out_ptr);
        }
      }
    });
  } else {
    int64_t grain = std::max<int64_t>(1, O / num_threads);
    at::parallel_for(0, O, grain, [&](int64_t begin, int64_t end) {
      for (int64_t tc = begin; tc < end; ++tc) {
        tile_dispatch(
            M,
            I,
            O,
            static_cast<int>(tc),
            x_ptr,
            static_cast<int>(K),
            static_cast<int>(N),
            trellis_ptr,
            E,
            bank_ptr,
            segments_per_tile,
            states_per_segment,
            v2b2_p32,
            static_cast<int>(bank_alt_id),
            out_ptr);
      }
    });
  }

  return out;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void dequantize_tiles_avx512(
    int64_t begin,
    int64_t end,
    int O,
    int64_t N,
    int E,
    int words_per_tile,
    int segments_per_tile,
    int states_per_segment,
    const int32_t* trellis_ptr,
    const uint8_t* bank_ptr,
    bool v2b2_p32,
    int bank_alt_id,
    float* inner_ptr) {
  for (int64_t tile_idx = begin; tile_idx < end; ++tile_idx) {
    int tr = static_cast<int>(tile_idx / O);
    int tc = static_cast<int>(tile_idx % O);
    const int32_t* tile_words = trellis_ptr + tile_idx * words_per_tile;

    alignas(64) uint16_t codes[128];
    unpack_tile_codes_avx512(tile_words, E, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ptr) {
      tile_segment_masks(
          bank_ptr[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }

    alignas(64) float tile_weights[256];
    decode_tile_avx512(E, codes, segments_per_tile, states_per_segment, seg_masks, tile_weights);

    float* out_base = inner_ptr + static_cast<int64_t>(tr) * 16 * N + tc * 16;
    for (int i = 0; i < 16; ++i) {
      _mm512_storeu_ps(out_base + static_cast<int64_t>(i) * N, _mm512_loadu_ps(tile_weights + i * 16));
    }
  }
}

template <int E>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void dequantize_tiles_avx512_specialized(
    int64_t begin,
    int64_t end,
    int O,
    int64_t N,
    int segments_per_tile,
    int states_per_segment,
    const int32_t* trellis_ptr,
    const uint8_t* bank_ptr,
    bool v2b2_p32,
    int bank_alt_id,
    float* inner_ptr) {
  for (int64_t tile_idx = begin; tile_idx < end; ++tile_idx) {
    const int tr = static_cast<int>(tile_idx / O);
    const int tc = static_cast<int>(tile_idx % O);
    const int32_t* tile_words = trellis_ptr + tile_idx * (4 * E);
    alignas(64) uint16_t codes[128];
    unpack_tile_codes_avx512_specialized<E>(tile_words, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ptr) {
      tile_segment_masks(bank_ptr[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }
    alignas(64) float tile_weights[256];
    decode_tile_avx512_specialized<E>(
        codes, segments_per_tile, states_per_segment, seg_masks, tile_weights);
    float* out_base = inner_ptr + static_cast<int64_t>(tr) * 16 * N + tc * 16;
    for (int i = 0; i < 16; ++i) {
      _mm512_storeu_ps(out_base + static_cast<int64_t>(i) * N, _mm512_loadu_ps(tile_weights + i * 16));
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
static inline void dequantize_tiles_avx512_dispatch(
    int64_t begin,
    int64_t end,
    int O,
    int64_t N,
    int E,
    int words_per_tile,
    int segments_per_tile,
    int states_per_segment,
    const int32_t* trellis_ptr,
    const uint8_t* bank_ptr,
    bool v2b2_p32,
    int bank_alt_id,
    float* inner_ptr) {
  switch (E) {
    case 2:
      dequantize_tiles_avx512_specialized<2>(
          begin, end, O, N, segments_per_tile, states_per_segment, trellis_ptr, bank_ptr,
          v2b2_p32, bank_alt_id, inner_ptr);
      break;
    case 3:
      dequantize_tiles_avx512_specialized<3>(
          begin, end, O, N, segments_per_tile, states_per_segment, trellis_ptr, bank_ptr,
          v2b2_p32, bank_alt_id, inner_ptr);
      break;
    case 4:
      dequantize_tiles_avx512_specialized<4>(
          begin, end, O, N, segments_per_tile, states_per_segment, trellis_ptr, bank_ptr,
          v2b2_p32, bank_alt_id, inner_ptr);
      break;
    case 5:
      dequantize_tiles_avx512_specialized<5>(
          begin, end, O, N, segments_per_tile, states_per_segment, trellis_ptr, bank_ptr,
          v2b2_p32, bank_alt_id, inner_ptr);
      break;
    case 6:
      dequantize_tiles_avx512_specialized<6>(
          begin, end, O, N, segments_per_tile, states_per_segment, trellis_ptr, bank_ptr,
          v2b2_p32, bank_alt_id, inner_ptr);
      break;
    case 7:
      dequantize_tiles_avx512_specialized<7>(
          begin, end, O, N, segments_per_tile, states_per_segment, trellis_ptr, bank_ptr,
          v2b2_p32, bank_alt_id, inner_ptr);
      break;
    case 8:
      dequantize_tiles_avx512_specialized<8>(
          begin, end, O, N, segments_per_tile, states_per_segment, trellis_ptr, bank_ptr,
          v2b2_p32, bank_alt_id, inner_ptr);
      break;
    default:
      dequantize_tiles_avx512(
          begin, end, O, N, E, words_per_tile, segments_per_tile, states_per_segment,
          trellis_ptr, bank_ptr, v2b2_p32, bank_alt_id, inner_ptr);
      break;
  }
}

torch::Tensor qvq_inner_weight_cpu(
    torch::Tensor trellis,
    int64_t transition_bits,
    int64_t out_features,
    c10::optional<torch::Tensor> bank_ids,
    int64_t bank_alt_id,
    bool v2b4_p64,
    bool v2b2_p32) {
  TORCH_CHECK(trellis.device().is_cpu(), "qvq_inner_weight_cpu: trellis must be a CPU tensor");
  TORCH_CHECK(trellis.dim() == 2, "qvq_inner_weight_cpu: trellis must be 2D [tile_count, words]");
  TORCH_CHECK(trellis.scalar_type() == at::kInt, "qvq_inner_weight_cpu: trellis must be int32");
  TORCH_CHECK(
      transition_bits >= 2 && transition_bits <= 16,
      "qvq_inner_weight_cpu: transition_bits must be 2..16");
  TORCH_CHECK(!(v2b4_p64 && v2b2_p32), "qvq_inner_weight_cpu: v2b4_p64 and v2b2_p32 are mutually exclusive");

  int64_t N = out_features;
  TORCH_CHECK(N % 16 == 0, "qvq_inner_weight_cpu: out_features must be a multiple of 16");
  int O = static_cast<int>(N / 16);
  int E = static_cast<int>(transition_bits);
  int words_per_tile = 4 * E;

  int64_t tile_count = trellis.size(0);
  TORCH_CHECK(trellis.size(1) == words_per_tile, "qvq_inner_weight_cpu: trellis words_per_tile mismatch");
  TORCH_CHECK(tile_count % O == 0, "qvq_inner_weight_cpu: tile_count must be divisible by out_features/16");
  int I = static_cast<int>(tile_count / O);
  int64_t K = static_cast<int64_t>(I) * 16;

  if (v2b4_p64 || v2b2_p32) {
    TORCH_CHECK(bank_ids.has_value(), "qvq_inner_weight_cpu: banked formats require bank_ids");
    TORCH_CHECK(
        bank_ids->dim() == 1 && bank_ids->size(0) == tile_count,
        "qvq_inner_weight_cpu: bank_ids must be [tile_count]");
    TORCH_CHECK(
        bank_ids->scalar_type() == at::kByte, "qvq_inner_weight_cpu: bank_ids must be uint8");
    TORCH_CHECK(bank_ids->device() == trellis.device(), "qvq_inner_weight_cpu: bank_ids must be on CPU");
    if (v2b2_p32) {
      TORCH_CHECK(bank_alt_id >= 1 && bank_alt_id <= 3, "qvq_inner_weight_cpu: v2b2_p32 bank_alt_id must be 1..3");
    }
  }

  int segments_per_tile = 1;
  int states_per_segment = 128;
  if (v2b2_p32) {
    segments_per_tile = 8;
    states_per_segment = 16;
  } else if (v2b4_p64) {
    segments_per_tile = 4;
    states_per_segment = 32;
  }

  pgc16_ensure_tables();

  auto inner = at::empty({K, N}, at::TensorOptions().dtype(at::kFloat).device(trellis.device()));

  const int32_t* trellis_ptr = trellis.data_ptr<int32_t>();
  float* inner_ptr = inner.data_ptr<float>();
  const uint8_t* bank_ptr = nullptr;
  if (bank_ids.has_value() && bank_ids->numel() > 0) {
    bank_ptr = bank_ids->data_ptr<uint8_t>();
  }

  const bool has_avx512 = cpu_has_avx512();
  int64_t num_threads = at::get_num_threads();
  int64_t grain = std::max<int64_t>(1, tile_count / num_threads);

  at::parallel_for(0, tile_count, grain, [&](int64_t begin, int64_t end) {
    if (has_avx512) {
      dequantize_tiles_avx512_dispatch(
          begin,
          end,
          O,
          N,
          E,
          words_per_tile,
          segments_per_tile,
          states_per_segment,
          trellis_ptr,
          bank_ptr,
          v2b2_p32,
          static_cast<int>(bank_alt_id),
          inner_ptr);
    } else {
      alignas(64) float tile_weights[256];
      for (int64_t tile_idx = begin; tile_idx < end; ++tile_idx) {
        int tr = static_cast<int>(tile_idx / O);
        int tc = static_cast<int>(tile_idx % O);
        const int32_t* tile_words = trellis_ptr + tile_idx * words_per_tile;

        uint16_t seg_masks[8] = {0};
        if (bank_ptr) {
          tile_segment_masks(
              bank_ptr[tile_idx], E, segments_per_tile, v2b2_p32, static_cast<int>(bank_alt_id), seg_masks);
        }

        uint16_t codes[128];
        unpack_tile_codes(tile_words, E, codes);
        decode_tile(E, codes, segments_per_tile, states_per_segment, seg_masks, tile_weights);

        float* out_base = inner_ptr + static_cast<int64_t>(tr) * 16 * N + tc * 16;
        for (int i = 0; i < 16; ++i) {
          float* row = out_base + static_cast<int64_t>(i) * N;
          const float* w = tile_weights + i * 16;
          for (int j = 0; j < 16; ++j) {
            row[j] = w[j];
          }
        }
      }
    }
  });

  return inner;
}

}  // namespace qvq_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def(
      "gemv_cpu(Tensor x, Tensor trellis, int transition_bits, int out_features, Tensor? bank_ids=None, "
      "int bank_alt_id=0, bool v2b4_p64=False, bool v2b2_p32=False) -> Tensor");
  m.def(
      "inner_weight_cpu(Tensor trellis, int transition_bits, int out_features, Tensor? bank_ids=None, "
      "int bank_alt_id=0, bool v2b4_p64=False, bool v2b2_p32=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("gemv_cpu", qvq_cpu::qvq_gemv_cpu);
  m.impl("inner_weight_cpu", qvq_cpu::qvq_inner_weight_cpu);
}
