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

alignas(64) float g_state_values[65536][2];
std::once_flag g_tables_once;

inline uint16_t pgc16_mix_state(uint16_t state) {
  uint32_t mixed = state ^ (state >> 8);
  mixed = (mixed * PGC16_MULTIPLIER + PGC16_INCREMENT) & 0xFFFF;
  return static_cast<uint16_t>(mixed ^ (mixed >> 7));
}

void pgc16_build_tables() {
  for (uint32_t state = 0; state < 65536; ++state) {
    uint16_t mixed = pgc16_mix_state(static_cast<uint16_t>(state));
    uint16_t hi = mixed >> 8;
    uint16_t lo = mixed & 0xFF;
    c10::Half h0 = c10::Half(kPgc16LevelBits[hi], c10::Half::from_bits());
    c10::Half h1 = c10::Half(kPgc16LevelBits[lo], c10::Half::from_bits());
    g_state_values[state][0] = static_cast<float>(h0);
    g_state_values[state][1] = static_cast<float>(h1);
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

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void unpack_tile_codes_avx512(const int32_t* tile_words, int E, uint16_t* codes) {
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
    const float* vals = g_state_values[state ^ mask];
    tile_weights[2 * s] = vals[0];
    tile_weights[2 * s + 1] = vals[1];
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
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void decode_tile_avx512(
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

  const float* table0 = &g_state_values[0][0];
  const float* table1 = &g_state_values[0][1];

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

    const __m512 v0 = _mm512_i32gather_ps(state, table0, 8);
    const __m512 v1 = _mm512_i32gather_ps(state, table1, 8);
    const __m512 out0 = _mm512_permutex2var_ps(v0, perm0, v1);
    const __m512 out1 = _mm512_permutex2var_ps(v0, perm1, v1);
    _mm512_storeu_ps(tile_weights + 2 * base, out0);
    _mm512_storeu_ps(tile_weights + 2 * base + 16, out1);
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

  return out;
}

}  // namespace qvq_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def(
      "gemv_cpu(Tensor x, Tensor trellis, int transition_bits, int out_features, Tensor? bank_ids=None, "
      "int bank_alt_id=0, bool v2b4_p64=False, bool v2b2_p32=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("gemv_cpu", qvq_cpu::qvq_gemv_cpu);
}
