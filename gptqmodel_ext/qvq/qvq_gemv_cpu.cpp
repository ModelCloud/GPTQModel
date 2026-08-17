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

// Precomputed sliding-window state extraction.
struct StepWindow {
  uint8_t code_idx[16];
  uint16_t bit_mask[16];
  uint8_t state_shift[16];
};

void build_step_windows(int E, StepWindow* windows) {
  int total_bits = 128 * E;
  for (int step = 0; step < 128; ++step) {
    int start = ((step + 1) * E) - 16;
    start %= total_bits;
    if (start < 0) start += total_bits;
    for (int b = 0; b < 16; ++b) {
      int pos = start + b;
      if (pos >= total_bits) pos -= total_bits;
      int code_idx = pos / E;
      int bit_in_code = E - 1 - (pos % E);
      windows[step].code_idx[b] = static_cast<uint8_t>(code_idx);
      windows[step].bit_mask[b] = static_cast<uint16_t>(1u << bit_in_code);
      windows[step].state_shift[b] = static_cast<uint8_t>(15 - b);
    }
  }
}

// Decode one tile into 256 row-major float weights.
void decode_tile(
    const uint16_t* codes,
    const StepWindow* windows,
    int segments_per_tile,
    int states_per_segment,
    const uint16_t* seg_masks,
    float* tile_weights) {
  for (int s = 0; s < 128; ++s) {
    uint16_t state = 0;
    const StepWindow& w = windows[s];
    for (int b = 0; b < 16; ++b) {
      if (codes[w.code_idx[b]] & w.bit_mask[b]) {
        state |= static_cast<uint16_t>(1u << w.state_shift[b]);
      }
    }
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
    const StepWindow* windows,
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
    uint16_t codes[128];
    unpack_tile_codes(tile_words, E, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ids) {
      tile_segment_masks(bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }
    decode_tile(codes, windows, segments_per_tile, states_per_segment, seg_masks, tile_weights);
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
    const StepWindow* windows,
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
    uint16_t codes[128];
    unpack_tile_codes(tile_words, E, codes);
    uint16_t seg_masks[8] = {0};
    if (bank_ids) {
      tile_segment_masks(bank_ids[tile_idx], E, segments_per_tile, v2b2_p32, bank_alt_id, seg_masks);
    }
    decode_tile(codes, windows, segments_per_tile, states_per_segment, seg_masks, tile_weights);

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
    const StepWindow* windows,
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
    decode_tile(codes, windows, segments_per_tile, states_per_segment, seg_masks, tile_weights);

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
    const StepWindow* windows,
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
    const StepWindow* windows,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  if (M == 1) {
    accumulate_tile_m1_avx512(
        I, O, tc, x, K, trellis, E, windows, bank_ids, segments_per_tile,
        states_per_segment, v2b2_p32, bank_alt_id, out);
  } else {
    accumulate_tile_mn_avx512(
        M, I, O, tc, x, K, N, trellis, E, windows, bank_ids, segments_per_tile,
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
    const StepWindow* windows,
    const uint8_t* bank_ids,
    int segments_per_tile,
    int states_per_segment,
    bool v2b2_p32,
    int bank_alt_id,
    float* out) {
  if (cpu_has_avx512()) {
    tile_avx512_dispatch(
        M, I, O, tc, x, K, N, trellis, E, windows, bank_ids, segments_per_tile,
        states_per_segment, v2b2_p32, bank_alt_id, out);
  } else {
    accumulate_tile_scalar(
        M, I, O, tc, x, K, N, trellis, E, windows, bank_ids, segments_per_tile,
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

  StepWindow windows[128];
  build_step_windows(E, windows);

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
          windows,
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
