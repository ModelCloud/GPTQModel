// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Experimental MXFP4 (E2M1 weights + E8M0 per-32 scales) on-the-fly dequant
// + BF16/FP16/FP8-E4M3 matmul kernel for x86 CPUs with AVX-512.
//
// Storage stays at 4.25 bits/weight (MXFP4): qweight is (N, K/2) uint8 and
// scales is (N, K/32) uint8.  The kernel dequantizes each 32-element block to
// FP32 on the fly, FMAs against BF16/FP16/FP8 activations, and writes the result.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#define GPTQMODEL_MXFP4_X86 1
#else
#define GPTQMODEL_MXFP4_X86 0
#endif

// AVX512-FP16 intrinsics (__m512h, VFMADD*PH) need GCC >= 12 or Clang >= 14.
// The code is still gated at runtime on __builtin_cpu_supports("avx512fp16").
#if GPTQMODEL_MXFP4_X86 &&                                                    \
    ((defined(__clang__) && __clang_major__ >= 14) ||                         \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 12))
#define GPTQMODEL_MXFP4_HAS_FP16_ISA 1
#else
#define GPTQMODEL_MXFP4_HAS_FP16_ISA 0
#endif

// Tile shape for the AVX-512 BF16 FP8 path.  kNTile output columns x kMBlock
// activation rows are kept in registers; kNTile * kMBlock must stay <= 32 zmm.
#ifndef GPTQMODEL_MXFP4_NTILE
#define GPTQMODEL_MXFP4_NTILE 4
#endif
#ifndef GPTQMODEL_MXFP4_MBLOCK
#define GPTQMODEL_MXFP4_MBLOCK 8
#endif

namespace gptqmodel_mxfp4 {

namespace {

constexpr std::array<float, 16> kFp4Table = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// FP16 accumulators are folded into FP32 every kFp16FlushGroups K-groups (32
// weights each), bounding FP16 rounding drift over long reductions.  2048
// elements per fold measured as the knee on Emerald Rapids: at K=12288 it costs
// ~7% over unbounded FP16 accumulation while staying within 1 FP16 ULP of FP32
// accumulation, and folding disappears entirely when K <= 2048.
constexpr int64_t kFp16FlushGroups = 64;

// Variant 2 accumulates in FP16 for the whole reduction; everything else folds
// every `requested` K-groups, or every kFp16FlushGroups when unspecified.
inline int64_t fp16_flush_groups(int64_t variant, int64_t requested, int64_t K_groups) {
  if (requested > 0) {
    return std::min<int64_t>(requested, K_groups);
  }
  return variant == 2 ? K_groups : std::min<int64_t>(kFp16FlushGroups, K_groups);
}

// Variant 4 pins the narrow 4x4 tile so the M-block width can be measured.
inline bool fp16_wide_m(int64_t variant, int64_t M) {
  return variant != 4 && M >= 8;
}

// E8M0 scale: value = 2^(bits - 127).  bits=255 is NaN in torch, treat as 0.
inline float e8m0_scale(uint8_t bits) {
  if (bits == 255) {
    return 0.0f;
  }
  return std::ldexp(1.0f, static_cast<int>(bits) - 127);
}

#if GPTQMODEL_MXFP4_X86
__attribute__((constructor))
static void mxfp4_cpu_init_cpu_features() {
  __builtin_cpu_init();
}

inline bool mxfp4_cpu_supports_avx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl");
}

inline bool mxfp4_cpu_supports_bf16() {
  return mxfp4_cpu_supports_avx512() && __builtin_cpu_supports("avx512bf16");
}

inline bool mxfp4_cpu_supports_fp16() {
#if GPTQMODEL_MXFP4_HAS_FP16_ISA
  return mxfp4_cpu_supports_avx512() && __builtin_cpu_supports("avx512fp16");
#else
  return false;
#endif
}

// FP16 dequant is only exact/representable while |fp4 value| * 2^(bits-127)
// stays inside the normal FP16 range: max 6*2^13 < 65504 and min 0.5*2^-13
// above the smallest normal.  Scan the E8M0 bytes once and refuse the FP16
// path outside that window (also rejects the 255/NaN encoding).  Returns the
// largest scale byte seen, or -1 when any byte is outside the window.
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline int e8m0_scales_fp16_safe_avx512(const uint8_t* s_ptr, int64_t count) {
  __m512i vmin = _mm512_set1_epi8(static_cast<char>(0xFF));
  __m512i vmax = _mm512_setzero_si512();
  int64_t i = 0;
  for (; i + 64 <= count; i += 64) {
    const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s_ptr + i));
    vmin = _mm512_min_epu8(vmin, v);
    vmax = _mm512_max_epu8(vmax, v);
  }
  alignas(64) uint8_t buf_min[64];
  alignas(64) uint8_t buf_max[64];
  _mm512_store_si512(reinterpret_cast<__m512i*>(buf_min), vmin);
  _mm512_store_si512(reinterpret_cast<__m512i*>(buf_max), vmax);
  uint8_t lo = 0xFF;
  uint8_t hi = 0;
  for (int j = 0; j < 64; ++j) {
    lo = std::min<uint8_t>(lo, buf_min[j]);
    hi = std::max<uint8_t>(hi, buf_max[j]);
  }
  for (; i < count; ++i) {
    lo = std::min<uint8_t>(lo, s_ptr[i]);
    hi = std::max<uint8_t>(hi, s_ptr[i]);
  }
  if (lo < 114 || hi > 140) {
    return -1;
  }
  return static_cast<int>(hi);
}

// Largest |activation|, read straight off the FP16 bit patterns: clearing the
// sign bit leaves a monotone unsigned ordering for finite values, and inf/NaN
// (>= 0x7C00) come out as the maximum so they force the fallback below.
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline uint16_t max_abs_fp16_bits_avx512(const uint16_t* x, int64_t count) {
  const __m512i abs_mask = _mm512_set1_epi16(0x7FFF);
  __m512i vmax = _mm512_setzero_si512();
  int64_t i = 0;
  for (; i + 32 <= count; i += 32) {
    const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + i));
    vmax = _mm512_max_epu16(vmax, _mm512_and_si512(v, abs_mask));
  }
  alignas(64) uint16_t buf[32];
  _mm512_store_si512(reinterpret_cast<__m512i*>(buf), vmax);
  uint16_t hi = 0;
  for (int j = 0; j < 32; ++j) {
    hi = std::max<uint16_t>(hi, buf[j]);
  }
  for (; i < count; ++i) {
    hi = std::max<uint16_t>(hi, static_cast<uint16_t>(x[i] & 0x7FFF));
  }
  return hi;
}

// The FP16 micro-kernel multiplies and accumulates in FP16, so representable
// weights are not enough: `flush_groups` products of the largest weight and the
// largest activation must also stay under 65504, otherwise a lane could
// saturate to inf where the FP32-accumulating path returns a finite value.
// max|weight| is 6 * 2^(max_scale_byte - 127).
inline bool fp16_accum_cannot_overflow(int max_scale_byte, uint16_t max_abs_a_bits, int64_t flush_groups) {
  if (max_abs_a_bits >= 0x7C00) {  // inf/NaN activation
    return false;
  }
  const float a_max = static_cast<float>(c10::Half(max_abs_a_bits, c10::Half::from_bits()));
  const float w_max = 6.0f * std::ldexp(1.0f, max_scale_byte - 127);
  return static_cast<double>(a_max) * w_max * static_cast<double>(flush_groups) <= 65504.0;
}
#endif

#if GPTQMODEL_MXFP4_HAS_FP16_ISA
// Same nibble unpack as the BF16 path, but the 32-lane table holds FP16 values,
// so VPERMW yields 32 dequantized FP16 weights directly.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512fp16")))
inline __m512h load_fp4x32_to_fp16_h(const uint8_t* src, const __m512i tab_vec) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  const __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
  const __m128i part0 = _mm_unpacklo_epi8(lo, hi);
  const __m128i part1 = _mm_unpackhi_epi8(lo, hi);
  const __m256i idx8 = _mm256_set_m128i(part1, part0);
  const __m512i idx16 = _mm512_cvtepu8_epi16(idx8);
  return _mm512_castsi512_ph(_mm512_permutexvar_epi16(idx16, tab_vec));
}

// Widen the 32 FP16 lanes to two FP32 vectors and fold them into acc.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512fp16")))
inline __m512 fold_ph_into_ps(__m512 acc, __m512h v) {
  const __m512i bits = _mm512_castph_si512(v);
  const __m512 lo = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 0));
  const __m512 hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1));
  return _mm512_add_ps(acc, _mm512_add_ps(lo, hi));
}

// kNTile output columns x up to kMBlock activation rows, MXFP4 -> FP16 dequant
// + VFMADD*PH.  Each weight vector is dequantized once and reused across the
// whole M block, so a wider M block lowers the dequant-to-FMA ratio.
// Products accumulate in FP16 for `flush_groups` K-groups (32 weights each),
// then fold into FP32 accumulators.  flush_groups >= K_groups means pure FP16
// accumulation; flush_groups == 1 keeps full FP32 accumulation precision.
template <int kNTile, int kMBlock, typename OutT>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512fp16")))
inline void mxfp4_fp16_tile_allm(
    int64_t n0,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t K_packed,
    int64_t K_groups,
    int64_t flush_groups,
    const c10::Half* x_fp16,
    const uint8_t* q_ptr,
    const uint8_t* s_ptr,
    const uint16_t* fp16_table_ptr,
    OutT* out_ptr) {
  const uint8_t* q_rows[kNTile];
  const uint8_t* s_rows[kNTile];
  for (int c = 0; c < kNTile; ++c) {
    q_rows[c] = q_ptr + (n0 + c) * K_packed;
    s_rows[c] = s_ptr + (n0 + c) * K_groups;
  }

  for (int64_t m0 = 0; m0 < M; m0 += kMBlock) {
    const int64_t m_count = std::min<int64_t>(kMBlock, M - m0);
    __m512 facc[kNTile][kMBlock];
    for (int c = 0; c < kNTile; ++c) {
      for (int64_t b = 0; b < m_count; ++b) {
        facc[c][b] = _mm512_setzero_ps();
      }
    }

    for (int64_t g0 = 0; g0 < K_groups; g0 += flush_groups) {
      const int64_t g_end = std::min<int64_t>(g0 + flush_groups, K_groups);
      __m512h acc[kNTile][kMBlock];
      for (int c = 0; c < kNTile; ++c) {
        for (int64_t b = 0; b < m_count; ++b) {
          acc[c][b] = _mm512_setzero_ph();
        }
      }

      for (int64_t g = g0; g < g_end; ++g) {
        const int64_t p = g * 16;
        const int64_t x_off = m0 * K + g * 32;
        __m512h a[kMBlock];
        for (int64_t b = 0; b < m_count; ++b) {
          a[b] = _mm512_castsi512_ph(_mm512_loadu_si512(
              reinterpret_cast<const __m512i*>(x_fp16 + x_off + b * K)));
        }
        __m512h w[kNTile];
        for (int c = 0; c < kNTile; ++c) {
          const __m512i tab = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
              fp16_table_ptr + static_cast<size_t>(s_rows[c][g]) * 32));
          w[c] = load_fp4x32_to_fp16_h(q_rows[c] + p, tab);
        }
        for (int c = 0; c < kNTile; ++c) {
          for (int64_t b = 0; b < m_count; ++b) {
            acc[c][b] = _mm512_fmadd_ph(w[c], a[b], acc[c][b]);
          }
        }
      }

      for (int c = 0; c < kNTile; ++c) {
        for (int64_t b = 0; b < m_count; ++b) {
          facc[c][b] = fold_ph_into_ps(facc[c][b], acc[c][b]);
        }
      }
    }

    for (int64_t b = 0; b < m_count; ++b) {
      const int64_t out_base = (m0 + b) * N + n0;
      for (int c = 0; c < kNTile; ++c) {
        out_ptr[out_base + c] = OutT(_mm512_reduce_add_ps(facc[c][b]));
      }
    }
  }
}

// Scalar reference for the FP16 tables (N tail columns and non-AVX512 CPUs).
template <typename OutT>
inline void mxfp4_fp16_column_scalar(
    int64_t n,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t K_packed,
    int64_t K_groups,
    const c10::Half* x_fp16,
    const uint8_t* q_ptr,
    const uint8_t* s_ptr,
    const c10::Half (*fp16_table)[32],
    OutT* out_ptr) {
  const uint8_t* q_row = q_ptr + n * K_packed;
  const uint8_t* s_row = s_ptr + n * K_groups;
  for (int64_t m = 0; m < M; ++m) {
    const c10::Half* x_row = x_fp16 + m * K;
    float sum = 0.0f;
    for (int64_t g = 0; g < K_groups; ++g) {
      const c10::Half* wtab = fp16_table[s_row[g]];
      const int64_t k = g * 32;
      const int64_t p = g * 16;
      for (int i = 0; i < 16; ++i) {
        const uint8_t byte = q_row[p + i];
        sum += static_cast<float>(x_row[k + i * 2 + 0]) * static_cast<float>(wtab[byte & 0x0F]);
        sum += static_cast<float>(x_row[k + i * 2 + 1]) * static_cast<float>(wtab[(byte >> 4) & 0x0F]);
      }
    }
    out_ptr[m * N + n] = OutT(sum);
  }
}

// Drive the FP16 micro-kernel over all N tiles.  Small M blocks waste the
// dequantized weight vector, so batches of >= 8 rows use a 2x8 tile.
template <typename OutT>
inline void mxfp4_fp16_run(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t K_packed,
    int64_t K_groups,
    int64_t flush_groups,
    int64_t nthreads,
    const c10::Half* x_fp16,
    const uint8_t* q_ptr,
    const uint8_t* s_ptr,
    const c10::Half (*fp16_table)[32],
    bool wide_m,
    OutT* out_ptr) {
  const uint16_t* fp16_table_ptr = reinterpret_cast<const uint16_t*>(&fp16_table[0][0]);
  const int64_t kTile = wide_m ? 2 : 4;
  const int64_t n_tiles = N / kTile;
  const int64_t tiles_grain = std::max<int64_t>(1, n_tiles / nthreads);
  at::parallel_for(0, n_tiles, tiles_grain, [&](int64_t begin, int64_t end) {
    for (int64_t n_tile = begin; n_tile < end; ++n_tile) {
      if (wide_m) {
        mxfp4_fp16_tile_allm<2, 8>(n_tile * 2, M, N, K, K_packed, K_groups, flush_groups,
                                   x_fp16, q_ptr, s_ptr, fp16_table_ptr, out_ptr);
      } else {
        mxfp4_fp16_tile_allm<4, 4>(n_tile * 4, M, N, K, K_packed, K_groups, flush_groups,
                                   x_fp16, q_ptr, s_ptr, fp16_table_ptr, out_ptr);
      }
    }
  });
  for (int64_t n = n_tiles * kTile; n < N; ++n) {
    mxfp4_fp16_column_scalar(n, M, N, K, K_packed, K_groups, x_fp16, q_ptr, s_ptr, fp16_table, out_ptr);
  }
}
#endif

#if GPTQMODEL_MXFP4_X86

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m512 load_fp4x16_to_ps_avx512(const uint8_t* src, const float* table) {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
  const __m128i lo_nibbles = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
  const __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
  const __m128i interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
  const __m512i indices = _mm512_cvtepu8_epi32(interleaved);
  const __m512 tab = _mm512_load_ps(table);
  return _mm512_permutexvar_ps(indices, tab);
}

// GCC 11 does not expose _mm512_cvtpbh_ps, so implement the same bit-packing manually.
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m512 load_bf16x16_to_ps_avx512(const c10::BFloat16* src) {
  const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      reinterpret_cast<const uint16_t*>(src)));
  const __m512i extended = _mm512_cvtepu16_epi32(raw);
  return _mm512_castsi512_ps(_mm512_slli_epi32(extended, 16));
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m512 load_fp16x16_to_ps_avx512(const c10::Half* src) {
  const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      reinterpret_cast<const uint16_t*>(src)));
  return _mm512_cvtph_ps(raw);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m512 load_fp8x16_to_ps_avx512(const uint8_t* src, const float* fp8_table) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  return _mm512_i32gather_ps(_mm512_cvtepu8_epi32(raw), fp8_table, 4);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline void fp8_to_fp32_buffer(const uint8_t* src, float* dst, int64_t n, const float* fp8_to_fp32_table) {
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 v = load_fp8x16_to_ps_avx512(src + i, fp8_to_fp32_table);
    _mm512_storeu_ps(dst + i, v);
  }
  for (; i < n; ++i) {
    dst[i] = static_cast<float>(c10::Float8_e4m3fn(src[i], c10::Float8_e4m3fn::from_bits()));
  }
}

// One MXFP4 x activation dot product (BF16 or FP16 activations, FP32 accumulate).
// Kept as a target-attributed function rather than inline intrinsics in the
// dispatch lambda so the translation unit needs no global -mavx512* flags.
template <typename ActT>
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline float mxfp4_dot_avx512(
    const ActT* x_row,
    const uint8_t* q_row,
    const uint8_t* s_row,
    int64_t K_groups,
    const float (*scaled_table)[16]) {
  __m512 acc = _mm512_setzero_ps();
  for (int64_t g = 0; g < K_groups; ++g) {
    const float* tab = scaled_table[s_row[g]];
    const int64_t k = g * 32;
    const int64_t p = g * 16;

    const __m512 w0 = load_fp4x16_to_ps_avx512(q_row + p, tab);
    const __m512 w1 = load_fp4x16_to_ps_avx512(q_row + p + 8, tab);
    __m512 a0;
    __m512 a1;
    if constexpr (std::is_same_v<ActT, c10::BFloat16>) {
      a0 = load_bf16x16_to_ps_avx512(x_row + k);
      a1 = load_bf16x16_to_ps_avx512(x_row + k + 16);
    } else {
      a0 = load_fp16x16_to_ps_avx512(x_row + k);
      a1 = load_fp16x16_to_ps_avx512(x_row + k + 16);
    }
    acc = _mm512_fmadd_ps(w0, a0, acc);
    acc = _mm512_fmadd_ps(w1, a1, acc);
  }
  return _mm512_reduce_add_ps(acc);
}

template <int NT>
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline void mxfp4_fp8_tile(
    int64_t m,
    int64_t n0,
    int64_t N,
    int64_t K,
    int64_t K_packed,
    int64_t K_groups,
    const float* x_fp32,
    const uint8_t* q_ptr,
    const uint8_t* s_ptr,
    const float* scaled_table_fp8_ptr,
    c10::Float8_e4m3fn* out_ptr) {
  const float* x_row = x_fp32 + m * K;
  const uint8_t* q_rows[NT];
  const uint8_t* s_rows[NT];
  #pragma GCC unroll 8
  for (int t = 0; t < NT; ++t) {
    q_rows[t] = q_ptr + (n0 + t) * K_packed;
    s_rows[t] = s_ptr + (n0 + t) * K_groups;
  }
  __m512 acc[NT];
  #pragma GCC unroll 8
  for (int t = 0; t < NT; ++t) {
    acc[t] = _mm512_setzero_ps();
  }
  for (int64_t g = 0; g < K_groups; ++g) {
    const int64_t k = g * 32;
    const int64_t p = g * 16;
    const __m512 a0 = _mm512_loadu_ps(x_row + k);
    const __m512 a1 = _mm512_loadu_ps(x_row + k + 16);
    #pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) {
      const float* wtab = scaled_table_fp8_ptr + static_cast<size_t>(s_rows[t][g]) * 16;
      const __m512 w0 = load_fp4x16_to_ps_avx512(q_rows[t] + p, wtab);
      const __m512 w1 = load_fp4x16_to_ps_avx512(q_rows[t] + p + 8, wtab);
      acc[t] = _mm512_fmadd_ps(w0, a0, acc[t]);
      acc[t] = _mm512_fmadd_ps(w1, a1, acc[t]);
    }
  }
  const int64_t out_base = m * N + n0;
  #pragma GCC unroll 8
  for (int t = 0; t < NT; ++t) {
    out_ptr[out_base + t] = c10::Float8_e4m3fn(_mm512_reduce_add_ps(acc[t]));
  }
}
#endif

// Convert 32 packed 4-bit weights (16 bytes) into a vector of 32 BF16 values
// using a 32-lane table of 16 duplicated BF16 values for the current E8M0 scale.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))
inline __m512bh load_fp4x32_to_bf16_bh(const uint8_t* src, const __m512i tab_vec) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  const __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
  const __m128i part0 = _mm_unpacklo_epi8(lo, hi);  // lo0,hi0,...,lo7,hi7
  const __m128i part1 = _mm_unpackhi_epi8(lo, hi);  // lo8,hi8,...,lo15,hi15
  const __m256i idx8 = _mm256_set_m128i(part1, part0);
  const __m512i idx16 = _mm512_cvtepu8_epi16(idx8);
  return (__m512bh)_mm512_permutexvar_epi16(idx16, tab_vec);
}

// Dequantizes one kNTile-wide weight tile per K group and reuses it across all
// kMBlock activation rows, so the vpermw expansion is amortized over kMBlock
// VDPBF16PS instructions instead of one.  OutT selects the store dtype.
template <typename OutT, int kNTile, int kMBlock>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))
inline void mxfp4_tile_bf16_allm(
    int64_t n0,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t K_packed,
    int64_t K_groups,
    const c10::BFloat16* x_bf16,
    const uint8_t* q_ptr,
    const uint8_t* s_ptr,
    const uint16_t* bf16_table_ptr,
    OutT* out_ptr) {
  static_assert(kNTile * kMBlock <= 32, "accumulators must fit the 32 zmm registers");
  const uint8_t* q_rows[kNTile];
  const uint8_t* s_rows[kNTile];
  for (int c = 0; c < kNTile; ++c) {
    q_rows[c] = q_ptr + (n0 + c) * K_packed;
    s_rows[c] = s_ptr + (n0 + c) * K_groups;
  }

  for (int64_t m0 = 0; m0 < M; m0 += kMBlock) {
    const int64_t m_count = std::min<int64_t>(kMBlock, M - m0);
    __m512 acc[kNTile][kMBlock];
    for (int c = 0; c < kNTile; ++c) {
      for (int64_t b = 0; b < m_count; ++b) {
        acc[c][b] = _mm512_setzero_ps();
      }
    }

    for (int64_t g = 0; g < K_groups; ++g) {
      const int64_t p = g * 16;
      const int64_t x_off = m0 * K + g * 32;
      __m512bh a[kMBlock];
      for (int64_t b = 0; b < m_count; ++b) {
        a[b] = (__m512bh)_mm512_loadu_ps(
            reinterpret_cast<const float*>(x_bf16 + x_off + b * K));
      }

      for (int c = 0; c < kNTile; ++c) {
        const __m512i tab = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(bf16_table_ptr + static_cast<size_t>(s_rows[c][g]) * 32));
        const __m512bh w = load_fp4x32_to_bf16_bh(q_rows[c] + p, tab);
        for (int64_t b = 0; b < m_count; ++b) {
          acc[c][b] = _mm512_dpbf16_ps(acc[c][b], w, a[b]);
        }
      }
    }

    for (int64_t b = 0; b < m_count; ++b) {
      const int64_t m = m0 + b;
      const int64_t out_base = m * N + n0;
      for (int c = 0; c < kNTile; ++c) {
        out_ptr[out_base + c] = OutT(_mm512_reduce_add_ps(acc[c][b]));
      }
    }
  }
}

#if GPTQMODEL_MXFP4_X86
// GPTQMODEL_MXFP4_DISABLE_VNNI=1 forces the scalar reference path (fallback testing).
inline bool mxfp4_vnni_disabled_by_env() {
  static const bool disabled = [] {
    const char* v = std::getenv("GPTQMODEL_MXFP4_DISABLE_VNNI");
    return v != nullptr && v[0] == '1';
  }();
  return disabled;
}

inline bool mxfp4_cpu_supports_vnni() {
  return mxfp4_cpu_supports_avx512() && __builtin_cpu_supports("avx512vnni") &&
         !mxfp4_vnni_disabled_by_env();
}
#endif

// ---------------------------------------------------------------------------
// AVX-512 VNNI int8 path.
//
// The MXFP4 E2M1 code book scaled by 2 is exactly representable in int8:
//   {0, 1, 2, 3, 4, 6, 8, 12} and the negatives.  Adding a constant offset of
//   12 makes every weight an *unsigned* byte in [0, 24], which is what the u8
//   operand of VPDPBUSD needs.  Activations are quantized to signed int8 with a
//   per-(row, 32-element group) scale, matching the MXFP4 group granularity.
//
//   dot = sum_k (w_u8[k] - 12) * a_i8[k]
//       = vpdpbusd(w_u8, a_i8) - 12 * sum_k a_i8[k]
//
// The offset correction is identical for every output column, so it is a single
// broadcast subtract per (row, group).  The int32 group sum is then rescaled by
// (e8m0_scale / 2) * activation_scale into an FP32 accumulator.
//
// Weight bytes are pre-permuted (same 4.25 bits/weight MXFP4 footprint) into
// 16-column x 4-K tiles so one 64-byte VPDPBUSD operand covers 16 output
// columns x 4 K elements, with the activation supplied as a broadcast int32.
// ---------------------------------------------------------------------------

constexpr int64_t kVnniNTile = 16;
constexpr int kVnniOffset = 12;
// 256 packed bytes per (16 columns x 32 K) tile-group: 4 chunks of 64 bytes.
constexpr int64_t kVnniGroupBytes = 256;

inline uint8_t mxfp4_nibble_to_offset_u8(uint8_t nibble) {
  const int v = static_cast<int>(std::lrintf(kFp4Table[nibble] * 2.0f));
  return static_cast<uint8_t>(v + kVnniOffset);
}

#if GPTQMODEL_MXFP4_X86
// AVX-512 form of the activation quantizer below.  Bit-identical to the scalar
// version: _mm512_cvtps_epi32 and lrintf both round to nearest-even.
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline void mxfp4_quantize_activation_group_avx512(
    const uint8_t* x_u8,
    const float* fp8_to_fp32_table,
    int8_t* a_i8,
    float* a_scale,
    int32_t* a_corr) {
  const __m512 v0 = load_fp8x16_to_ps_avx512(x_u8, fp8_to_fp32_table);
  const __m512 v1 = load_fp8x16_to_ps_avx512(x_u8 + 16, fp8_to_fp32_table);
  // _mm512_and_ps needs AVX512DQ, so mask the sign bit through the integer domain.
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512 abs0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v0), abs_mask));
  const __m512 abs1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v1), abs_mask));
  const float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));
  if (!(max_abs > 0.0f)) {
    std::memset(a_i8, 0, 32);
    *a_scale = 0.0f;
    *a_corr = 0;
    return;
  }

  const __m512 inv = _mm512_set1_ps(127.0f / max_abs);
  const __m512i lo = _mm512_set1_epi32(-127);
  const __m512i hi = _mm512_set1_epi32(127);
  const __m512i q0 = _mm512_min_epi32(hi, _mm512_max_epi32(lo, _mm512_cvtps_epi32(_mm512_mul_ps(v0, inv))));
  const __m512i q1 = _mm512_min_epi32(hi, _mm512_max_epi32(lo, _mm512_cvtps_epi32(_mm512_mul_ps(v1, inv))));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(a_i8), _mm512_cvtepi32_epi8(q0));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(a_i8 + 16), _mm512_cvtepi32_epi8(q1));

  *a_scale = max_abs / 127.0f;
  *a_corr = kVnniOffset * (_mm512_reduce_add_epi32(q0) + _mm512_reduce_add_epi32(q1));
}
#endif

// Quantize one FP8 activation row group (32 values) to signed int8.
inline void mxfp4_quantize_activation_group(
    const uint8_t* x_u8,
    const float* fp8_to_fp32_table,
    int8_t* a_i8,
    float* a_scale,
    int32_t* a_corr) {
  float vals[32];
  float max_abs = 0.0f;
  for (int i = 0; i < 32; ++i) {
    const float v = fp8_to_fp32_table[x_u8[i]];
    vals[i] = v;
    max_abs = std::max(max_abs, std::fabs(v));
  }
  if (!(max_abs > 0.0f)) {
    for (int i = 0; i < 32; ++i) {
      a_i8[i] = 0;
    }
    *a_scale = 0.0f;
    *a_corr = 0;
    return;
  }
  const float scale = max_abs / 127.0f;
  const float inv_scale = 127.0f / max_abs;
  int32_t sum = 0;
  for (int i = 0; i < 32; ++i) {
    int q = static_cast<int>(std::lrintf(vals[i] * inv_scale));
    q = std::min(127, std::max(-127, q));
    a_i8[i] = static_cast<int8_t>(q);
    sum += q;
  }
  *a_scale = scale;
  *a_corr = kVnniOffset * sum;
}

#if GPTQMODEL_MXFP4_X86
template <int MB>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline void mxfp4_vnni_tile(
    int64_t n0,
    int64_t N,
    int64_t m0,
    int64_t K,
    int64_t K_groups,
    const int8_t* a_i8,
    const float* a_scale,
    const int32_t* a_corr,
    const uint8_t* q_tile,
    const uint8_t* s_tile,
    const float* wscale_half_table,
    c10::Float8_e4m3fn* out_ptr) {
  const __m512i code_tab = _mm512_broadcast_i32x4(_mm_setr_epi8(
      12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
  const __m512i nib_mask = _mm512_set1_epi8(0x0F);

  __m512 accf[MB];
  #pragma GCC unroll 8
  for (int b = 0; b < MB; ++b) {
    accf[b] = _mm512_setzero_ps();
  }

  for (int64_t g = 0; g < K_groups; ++g) {
    const uint8_t* q_grp = q_tile + g * kVnniGroupBytes;
    const __m512i sbits = _mm512_cvtepu8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(s_tile + g * kVnniNTile)));
    const __m512 wscale = _mm512_i32gather_ps(sbits, wscale_half_table, 4);

    __m512i acci[MB];
    #pragma GCC unroll 8
    for (int b = 0; b < MB; ++b) {
      acci[b] = _mm512_setzero_si512();
    }

    for (int c = 0; c < 4; ++c) {
      const __m512i raw = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(q_grp + c * 64));
      const __m512i w_lo = _mm512_shuffle_epi8(code_tab, _mm512_and_si512(raw, nib_mask));
      const __m512i w_hi = _mm512_shuffle_epi8(
          code_tab, _mm512_and_si512(_mm512_srli_epi16(raw, 4), nib_mask));
      #pragma GCC unroll 8
      for (int b = 0; b < MB; ++b) {
        const int32_t* a_row = reinterpret_cast<const int32_t*>(
            a_i8 + (m0 + b) * K + g * 32);
        acci[b] = _mm512_dpbusd_epi32(acci[b], w_lo, _mm512_set1_epi32(a_row[2 * c]));
        acci[b] = _mm512_dpbusd_epi32(acci[b], w_hi, _mm512_set1_epi32(a_row[2 * c + 1]));
      }
    }

    #pragma GCC unroll 8
    for (int b = 0; b < MB; ++b) {
      const int64_t ai = (m0 + b) * K_groups + g;
      const __m512i corr = _mm512_set1_epi32(a_corr[ai]);
      const __m512 v = _mm512_cvtepi32_ps(_mm512_sub_epi32(acci[b], corr));
      const __m512 cs = _mm512_mul_ps(wscale, _mm512_set1_ps(a_scale[ai]));
      accf[b] = _mm512_fmadd_ps(v, cs, accf[b]);
    }
  }

  const int64_t n_valid = std::min<int64_t>(kVnniNTile, N - n0);
  alignas(64) float tmp[kVnniNTile];
  for (int b = 0; b < MB; ++b) {
    _mm512_store_ps(tmp, accf[b]);
    c10::Float8_e4m3fn* out_row = out_ptr + (m0 + b) * N + n0;
    for (int64_t j = 0; j < n_valid; ++j) {
      out_row[j] = c10::Float8_e4m3fn(tmp[j]);
    }
  }
}
#endif

// Scalar reference for the VNNI packed layout (used when AVX-512 VNNI is absent).
// Carries no target attribute, and the translation unit is built without
// -mavx512*, so this cannot be auto-vectorized into VPDPBUSD -- which would
// fault on exactly the CPUs the fallback exists for.
inline void mxfp4_vnni_tile_scalar(
    int64_t n0,
    int64_t N,
    int64_t M,
    int64_t K,
    int64_t K_groups,
    const int8_t* a_i8,
    const float* a_scale,
    const int32_t* a_corr,
    const uint8_t* q_tile,
    const uint8_t* s_tile,
    const float* wscale_half_table,
    c10::Float8_e4m3fn* out_ptr) {
  const int64_t n_valid = std::min<int64_t>(kVnniNTile, N - n0);
  for (int64_t m = 0; m < M; ++m) {
    float acc[kVnniNTile] = {0.0f};
    for (int64_t g = 0; g < K_groups; ++g) {
      const uint8_t* q_grp = q_tile + g * kVnniGroupBytes;
      int32_t lane[kVnniNTile] = {0};
      for (int c = 0; c < 4; ++c) {
        for (int64_t L = 0; L < kVnniNTile; ++L) {
          for (int j = 0; j < 4; ++j) {
            const uint8_t byte = q_grp[c * 64 + L * 4 + j];
            const int w_lo = static_cast<int>(mxfp4_nibble_to_offset_u8(byte & 0x0F));
            const int w_hi = static_cast<int>(mxfp4_nibble_to_offset_u8((byte >> 4) & 0x0F));
            const int8_t* a_row = a_i8 + m * K + g * 32;
            lane[L] += w_lo * static_cast<int>(a_row[(2 * c) * 4 + j]);
            lane[L] += w_hi * static_cast<int>(a_row[(2 * c + 1) * 4 + j]);
          }
        }
      }
      const int64_t ai = m * K_groups + g;
      for (int64_t L = 0; L < kVnniNTile; ++L) {
        acc[L] += static_cast<float>(lane[L] - a_corr[ai]) *
                  (wscale_half_table[s_tile[g * kVnniNTile + L]] * a_scale[ai]);
      }
    }
    for (int64_t j = 0; j < n_valid; ++j) {
      out_ptr[m * N + n0 + j] = c10::Float8_e4m3fn(acc[j]);
    }
  }
}

inline void mxfp4_fp8_column_scalar(
    int64_t m,
    int64_t n,
    int64_t N,
    int64_t K,
    int64_t K_packed,
    int64_t K_groups,
    const float* x_fp32,
    const uint8_t* q_ptr,
    const uint8_t* s_ptr,
    const float* scaled_table_fp8_ptr,
    c10::Float8_e4m3fn* out_ptr) {
  const float* x_row = x_fp32 + m * K;
  const uint8_t* q_row = q_ptr + n * K_packed;
  const uint8_t* s_row = s_ptr + n * K_groups;
  float sum = 0.0f;
  for (int64_t g = 0; g < K_groups; ++g) {
    const float* wtab = scaled_table_fp8_ptr + static_cast<size_t>(s_row[g]) * 16;
    const int64_t k = g * 32;
    const int64_t p = g * 16;
    for (int i = 0; i < 16; ++i) {
      const uint8_t byte = q_row[p + i];
      const uint8_t lo = byte & 0x0F;
      const uint8_t hi = (byte >> 4) & 0x0F;
      sum += x_row[k + i * 2 + 0] * wtab[lo];
      sum += x_row[k + i * 2 + 1] * wtab[hi];
    }
  }
  out_ptr[m * N + n] = c10::Float8_e4m3fn(sum);
}

// total_macs is M*N*K: cost scales with the reduction length, not just with the
// number of output elements, so a decode-shaped GEMM (few outputs, long K) still
// deserves every core.
inline int64_t clamped_threads(int64_t requested, int64_t total_macs) {
  const int64_t hard_limit = 32;
  const int64_t available = at::get_num_threads();
  if (requested > 0) {
    return std::max<int64_t>(1, std::min<int64_t>(requested, std::min<int64_t>(available, hard_limit)));
  }
  // ~2M multiply-accumulates per thread is roughly 10-20 us of work, well above
  // the at::parallel_for dispatch overhead.
  const int64_t per_thread = 1 << 21;
  int64_t t = (total_macs + per_thread - 1) / per_thread;
  if (t < 1) {
    t = 1;
  }
  return std::min<int64_t>(t, std::min<int64_t>(available, hard_limit));
}

} // namespace

// Variant selects the compute path:
//   0 = auto, 1 = legacy FP32-accumulate / VDPBF16PS paths,
//   2 = native FP16 compute with unbounded FP16 accumulation,
//   3 = native FP16 compute with periodic FP32 folding (same as auto),
//   4 = native FP16 compute pinned to the narrow 4x4 tile.
at::Tensor mxfp4_linear_cpu(
    const at::Tensor& input,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    int64_t threads,
    int64_t variant,
    int64_t fp16_flush) {
  TORCH_CHECK(input.device().is_cpu(), "input must be CPU");
  TORCH_CHECK(qweight.device().is_cpu(), "qweight must be CPU");
  TORCH_CHECK(scales.device().is_cpu(), "scales must be CPU");
  TORCH_CHECK(input.dim() == 2, "input must be 2D (M, K)");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D (N, K/2)");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D (N, K/32)");

  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  const int64_t N = qweight.size(0);

  TORCH_CHECK(qweight.size(1) == K / 2, "qweight columns must be K/2");
  TORCH_CHECK(scales.size(1) == K / 32, "scales columns must be K/32");
  TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");

  const bool bf16 = input.scalar_type() == at::kBFloat16;
  const bool fp16 = input.scalar_type() == at::kHalf;
  const bool fp8 = input.scalar_type() == at::kFloat8_e4m3fn;
  TORCH_CHECK(bf16 || fp16 || fp8, "input must be BF16, FP16, or FP8_E4M3");

  at::Tensor output = at::empty({M, N}, input.options());

  const at::Tensor in = input.contiguous();
  const at::Tensor qw = qweight.contiguous();
  const at::Tensor sc = scales.contiguous();

  // Build per-E8M0-scale FP4 lookup table once.
  alignas(64) float scaled_table[256][16];
  for (int s = 0; s < 256; ++s) {
    const float scale = e8m0_scale(static_cast<uint8_t>(s));
    for (int i = 0; i < 16; ++i) {
      scaled_table[s][i] = kFp4Table[i] * scale;
    }
  }

  // Build FP8-E4M3 -> FP32 lookup table once (used for FP8 activation path).
  alignas(64) float fp8_to_fp32_table[256];
  for (int i = 0; i < 256; ++i) {
    fp8_to_fp32_table[i] = static_cast<float>(
        c10::Float8_e4m3fn(static_cast<uint8_t>(i), c10::Float8_e4m3fn::from_bits()));
  }

  // Build per-E8M0-scale FP4 table that is rounded to FP8-E4M3, then cast back
  // to FP32.  This gives an on-the-fly MXFP4 -> FP8 dequant path without storing
  // any FP8 weights (weights stay 4-bit + scales).
  alignas(64) float scaled_table_fp8[256][16];
  for (int s = 0; s < 256; ++s) {
    const float scale = e8m0_scale(static_cast<uint8_t>(s));
    for (int i = 0; i < 16; ++i) {
      scaled_table_fp8[s][i] =
          static_cast<float>(c10::Float8_e4m3fn(kFp4Table[i] * scale));
    }
  }

  // Build per-E8M0-scale FP4 table as FP16, duplicated to 32 lanes for VPERMW.
  // FP4 magnitudes (0.5..6) times a power of two are exact in FP16 as long as
  // the E8M0 exponent stays inside the range checked by e8m0_scales_fp16_safe.
  alignas(64) c10::Half scaled_table_fp16_512[256][32];
  for (int s = 0; s < 256; ++s) {
    const float scale = e8m0_scale(static_cast<uint8_t>(s));
    for (int i = 0; i < 16; ++i) {
      const c10::Half v = c10::Half(kFp4Table[i] * scale);
      scaled_table_fp16_512[s][i] = v;
      scaled_table_fp16_512[s][i + 16] = v;
    }
  }

  // Build per-E8M0-scale FP4 table as BF16, with each 16-value block duplicated
  // to 32 lanes for VPERMW/DPBF16PS lookup.  Values are rounded to FP8 first so
  // the BF16 path is bit-exact with the FP8-E4M3 reference (all FP8 values fit
  // exactly in BF16).  This stays in 4-bit + scale storage.
  alignas(64) c10::BFloat16 scaled_table_bf16_512[256][32];
  for (int s = 0; s < 256; ++s) {
    for (int i = 0; i < 16; ++i) {
      const c10::BFloat16 v = c10::BFloat16(scaled_table_fp8[s][i]);
      scaled_table_bf16_512[s][i] = v;
      scaled_table_bf16_512[s][i + 16] = v;
    }
  }

  // Same 32-lane layout without the FP8 rounding step.  Every MXFP4 value has
  // at most 3 mantissa bits, so these entries are exact in BF16 and the BF16
  // activation path keeps its current numerics.
  alignas(64) c10::BFloat16 scaled_table_bf16_exact_512[256][32];
  for (int s = 0; s < 256; ++s) {
    for (int i = 0; i < 16; ++i) {
      const c10::BFloat16 v = c10::BFloat16(scaled_table[s][i]);
      scaled_table_bf16_exact_512[s][i] = v;
      scaled_table_bf16_exact_512[s][i + 16] = v;
    }
  }

  const int64_t K_packed = K / 2;
  const int64_t K_groups = K / 32;

  const uint8_t* q_ptr = qw.const_data_ptr<uint8_t>();
  const uint8_t* s_ptr = sc.const_data_ptr<uint8_t>();

  const int64_t total = M * N;
  // Size the pool by MAC count, not by output elements: decode-shaped GEMMs
  // have a tiny M but each output still costs K multiply-adds.
  const int64_t nthreads = clamped_threads(threads, total * K);
  const int64_t grain = std::max<int64_t>(1, total / nthreads);

  if (bf16) {
    const c10::BFloat16* x_ptr = in.const_data_ptr<c10::BFloat16>();
    c10::BFloat16* out_ptr = output.data_ptr<c10::BFloat16>();

#if GPTQMODEL_MXFP4_X86
    if (mxfp4_cpu_supports_bf16()) {
      const uint16_t* bf16_table_ptr =
          reinterpret_cast<const uint16_t*>(&scaled_table_bf16_exact_512[0][0]);
      constexpr int kTile = GPTQMODEL_MXFP4_NTILE;
      constexpr int kMB = GPTQMODEL_MXFP4_MBLOCK;
      const int64_t n_tiles = (N + kTile - 1) / kTile;
      const int64_t tiles_grain = std::max<int64_t>(1, n_tiles / nthreads);

      at::parallel_for(0, n_tiles, tiles_grain, [&](int64_t begin, int64_t end) {
        for (int64_t n_tile = begin; n_tile < end; ++n_tile) {
          const int64_t n0 = n_tile * kTile;
          if (N - n0 >= kTile) {
            mxfp4_tile_bf16_allm<c10::BFloat16, kTile, kMB>(
                n0, M, N, K, K_packed, K_groups, x_ptr, q_ptr, s_ptr, bf16_table_ptr, out_ptr);
          } else {
            for (int64_t n = n0; n < N; ++n) {
              mxfp4_tile_bf16_allm<c10::BFloat16, 1, GPTQMODEL_MXFP4_MBLOCK>(
                  n, M, N, K, K_packed, K_groups, x_ptr, q_ptr, s_ptr, bf16_table_ptr, out_ptr);
            }
          }
        }
      });
      return output;
    }
#endif

    at::parallel_for(0, total, grain, [&](int64_t begin, int64_t end) {
#if GPTQMODEL_MXFP4_X86
      if (mxfp4_cpu_supports_avx512()) {
        for (int64_t idx = begin; idx < end; ++idx) {
          const int64_t m = idx / N;
          const int64_t n = idx % N;

          const float sum = mxfp4_dot_avx512(
              x_ptr + m * K, q_ptr + n * K_packed, s_ptr + n * K_groups, K_groups, scaled_table);
          out_ptr[idx] = c10::BFloat16(sum);
        }
        return;
      }
#endif
      // Scalar fallback.
      for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t m = idx / N;
        const int64_t n = idx % N;

        const c10::BFloat16* x_row = x_ptr + m * K;
        const uint8_t* q_row = q_ptr + n * K_packed;
        const uint8_t* s_row = s_ptr + n * K_groups;

        float sum = 0.0f;
        for (int64_t g = 0; g < K_groups; ++g) {
          const float scale = e8m0_scale(s_row[g]);
          const int64_t k = g * 32;
          const int64_t p = g * 16;
          for (int i = 0; i < 16; ++i) {
            const uint8_t byte = q_row[p + i];
            const uint8_t lo = byte & 0x0F;
            const uint8_t hi = (byte >> 4) & 0x0F;
            sum += static_cast<float>(x_row[k + i * 2 + 0]) * (kFp4Table[lo] * scale);
            sum += static_cast<float>(x_row[k + i * 2 + 1]) * (kFp4Table[hi] * scale);
          }
        }
        out_ptr[idx] = c10::BFloat16(sum);
      }
    });
  } else if (fp16) {
    const c10::Half* x_ptr = in.const_data_ptr<c10::Half>();
    c10::Half* out_ptr = output.data_ptr<c10::Half>();

#if GPTQMODEL_MXFP4_HAS_FP16_ISA
    if (variant != 1 && mxfp4_cpu_supports_fp16()) {
      // Native AVX512-FP16 path: MXFP4 -> FP16 dequant feeding VFMADD132PH,
      // which retires 32 multiply-adds per instruction like VDPBF16PS but
      // consumes FP16 activations without any widening.  Both the weight table
      // and the worst-case FP16 partial sum must be representable, otherwise
      // fall through to the FP32-accumulating path below.
      const int64_t flush_groups = fp16_flush_groups(variant, fp16_flush, K_groups);
      const int max_scale_byte = e8m0_scales_fp16_safe_avx512(s_ptr, N * K_groups);
      if (max_scale_byte >= 0 &&
          fp16_accum_cannot_overflow(
              max_scale_byte,
              max_abs_fp16_bits_avx512(reinterpret_cast<const uint16_t*>(x_ptr), M * K),
              flush_groups)) {
        mxfp4_fp16_run(M, N, K, K_packed, K_groups, flush_groups, nthreads, x_ptr, q_ptr, s_ptr,
                       scaled_table_fp16_512, fp16_wide_m(variant, M), out_ptr);
        return output;
      }
    }
#endif

    at::parallel_for(0, total, grain, [&](int64_t begin, int64_t end) {
#if GPTQMODEL_MXFP4_X86
      if (mxfp4_cpu_supports_avx512()) {
        for (int64_t idx = begin; idx < end; ++idx) {
          const int64_t m = idx / N;
          const int64_t n = idx % N;

          const float sum = mxfp4_dot_avx512(
              x_ptr + m * K, q_ptr + n * K_packed, s_ptr + n * K_groups, K_groups, scaled_table);
          out_ptr[idx] = c10::Half(sum);
        }
        return;
      }
#endif
      for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t m = idx / N;
        const int64_t n = idx % N;

        const c10::Half* x_row = x_ptr + m * K;
        const uint8_t* q_row = q_ptr + n * K_packed;
        const uint8_t* s_row = s_ptr + n * K_groups;

        float sum = 0.0f;
        for (int64_t g = 0; g < K_groups; ++g) {
          const float scale = e8m0_scale(s_row[g]);
          const int64_t k = g * 32;
          const int64_t p = g * 16;
          for (int i = 0; i < 16; ++i) {
            const uint8_t byte = q_row[p + i];
            const uint8_t lo = byte & 0x0F;
            const uint8_t hi = (byte >> 4) & 0x0F;
            sum += static_cast<float>(x_row[k + i * 2 + 0]) * (kFp4Table[lo] * scale);
            sum += static_cast<float>(x_row[k + i * 2 + 1]) * (kFp4Table[hi] * scale);
          }
        }
        out_ptr[idx] = c10::Half(sum);
      }
    });
  } else if (fp8) {
    const c10::Float8_e4m3fn* x_ptr_typed = in.const_data_ptr<c10::Float8_e4m3fn>();
    c10::Float8_e4m3fn* out_ptr_typed = output.data_ptr<c10::Float8_e4m3fn>();
    const uint8_t* x_u8 = reinterpret_cast<const uint8_t*>(x_ptr_typed);

#if GPTQMODEL_MXFP4_HAS_FP16_ISA
    if (variant == 2 || variant == 3 || variant == 4) {
      // Experimental: FP8 activations widened to FP16 once, then the same
      // VFMADD132PH micro-kernel used for FP16 inputs.  Included so the FP16
      // FMA and VDPBF16PS paths can be compared on identical inputs/outputs.
      const int64_t flush_groups = fp16_flush_groups(variant, fp16_flush, K_groups);
      const int max_scale_byte =
          mxfp4_cpu_supports_fp16() ? e8m0_scales_fp16_safe_avx512(s_ptr, N * K_groups) : -1;
      if (max_scale_byte >= 0) {
        at::Tensor x_fp16 = at::empty({M, K}, at::kHalf);
        c10::Half* x_fp16_ptr = x_fp16.data_ptr<c10::Half>();
        for (int64_t i = 0; i < M * K; ++i) {
          x_fp16_ptr[i] = c10::Half(fp8_to_fp32_table[x_u8[i]]);
        }
        if (fp16_accum_cannot_overflow(
                max_scale_byte,
                max_abs_fp16_bits_avx512(reinterpret_cast<const uint16_t*>(x_fp16_ptr), M * K),
                flush_groups)) {
          mxfp4_fp16_run(M, N, K, K_packed, K_groups, flush_groups, nthreads, x_fp16_ptr, q_ptr, s_ptr,
                         scaled_table_fp16_512, fp16_wide_m(variant, M), out_ptr_typed);
          return output;
        }
      }
    }
#endif

#if GPTQMODEL_MXFP4_X86
    if (variant != 1 && mxfp4_cpu_supports_bf16()) {
      // Fast path: FP8 activations are converted to BF16 once, then VDPBF16PS
      // performs 32 multiply-adds per instruction (2x the FP32 FMA throughput).
      at::Tensor x_bf16 = at::empty({M, K}, at::kBFloat16);
      c10::BFloat16* x_bf16_ptr = x_bf16.data_ptr<c10::BFloat16>();
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
          x_bf16_ptr[m * K + k] = c10::BFloat16(fp8_to_fp32_table[x_u8[m * K + k]]);
        }
      }
      const uint16_t* bf16_table_ptr = reinterpret_cast<const uint16_t*>(&scaled_table_bf16_512[0][0]);

      constexpr int kTile = GPTQMODEL_MXFP4_NTILE;
      constexpr int kMB = GPTQMODEL_MXFP4_MBLOCK;
      const int64_t n_tiles = (N + kTile - 1) / kTile;
      const int64_t tiles_grain = std::max<int64_t>(1, n_tiles / nthreads);

      at::parallel_for(0, n_tiles, tiles_grain, [&](int64_t begin, int64_t end) {
        for (int64_t n_tile = begin; n_tile < end; ++n_tile) {
          const int64_t n0 = n_tile * kTile;
          const int64_t n_end = std::min<int64_t>(n0 + kTile, N);
          if (n_end - n0 == kTile) {
            mxfp4_tile_bf16_allm<c10::Float8_e4m3fn, kTile, kMB>(n0, M, N, K, K_packed, K_groups, x_bf16_ptr, q_ptr, s_ptr, bf16_table_ptr, out_ptr_typed);
          } else {
            for (int64_t n = n0; n < n_end; ++n) {
              const uint8_t* q_row = q_ptr + n * K_packed;
              const uint8_t* s_row = s_ptr + n * K_groups;
              for (int64_t m = 0; m < M; ++m) {
                const c10::BFloat16* x_row = x_bf16_ptr + m * K;
                float sum = 0.0f;
                for (int64_t g = 0; g < K_groups; ++g) {
                  const c10::BFloat16* wtab = scaled_table_bf16_512[s_row[g]];
                  const int64_t k = g * 32;
                  const int64_t p = g * 16;
                  for (int i = 0; i < 16; ++i) {
                    const uint8_t byte = q_row[p + i];
                    const uint8_t lo = byte & 0x0F;
                    const uint8_t hi = (byte >> 4) & 0x0F;
                    sum += static_cast<float>(x_row[k + i * 2 + 0]) * static_cast<float>(wtab[lo]);
                    sum += static_cast<float>(x_row[k + i * 2 + 1]) * static_cast<float>(wtab[hi]);
                  }
                }
                out_ptr_typed[m * N + n] = c10::Float8_e4m3fn(sum);
              }
            }
          }
        }
      });
      return output;
    }
#endif

    // Pre-convert FP8 activations to FP32 once.  This lets the hot loop use
    // cheap FP32 loads and reuse each activation chunk across a 4-wide N tile.
    at::Tensor x_fp32 = at::empty({M, K}, at::kFloat);
    float* x_fp32_ptr = x_fp32.data_ptr<float>();
#if GPTQMODEL_MXFP4_X86
    if (mxfp4_cpu_supports_avx512()) {
      for (int64_t m = 0; m < M; ++m) {
        fp8_to_fp32_buffer(x_u8 + m * K, x_fp32_ptr + m * K, K, fp8_to_fp32_table);
      }
    } else {
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
          x_fp32_ptr[m * K + k] = static_cast<float>(x_ptr_typed[m * K + k]);
        }
      }
    }
#else
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t k = 0; k < K; ++k) {
        x_fp32_ptr[m * K + k] = static_cast<float>(x_ptr_typed[m * K + k]);
      }
    }
#endif

    const float* scaled_table_fp8_ptr = &scaled_table_fp8[0][0];
    constexpr int64_t kTile = 8;
    const int64_t n_tiles = (N + kTile - 1) / kTile;
    const int64_t total_tiles = M * n_tiles;
    const int64_t tiles_grain = std::max<int64_t>(1, total_tiles / nthreads);

    at::parallel_for(0, total_tiles, tiles_grain, [&](int64_t begin, int64_t end) {
#if GPTQMODEL_MXFP4_X86
      if (mxfp4_cpu_supports_avx512()) {
        for (int64_t tile = begin; tile < end; ++tile) {
          const int64_t m = tile / n_tiles;
          const int64_t n_tile = tile % n_tiles;
          const int64_t n0 = n_tile * kTile;
          const int64_t rem = N - n0;
          if (rem >= kTile) {
            mxfp4_fp8_tile<kTile>(m, n0, N, K, K_packed, K_groups, x_fp32_ptr, q_ptr, s_ptr, scaled_table_fp8_ptr, out_ptr_typed);
          } else if (rem >= 4) {
            mxfp4_fp8_tile<4>(m, n0, N, K, K_packed, K_groups, x_fp32_ptr, q_ptr, s_ptr, scaled_table_fp8_ptr, out_ptr_typed);
            for (int64_t n = n0 + 4; n < N; ++n) {
              mxfp4_fp8_tile<1>(m, n, N, K, K_packed, K_groups, x_fp32_ptr, q_ptr, s_ptr, scaled_table_fp8_ptr, out_ptr_typed);
            }
          } else {
            for (int64_t n = n0; n < N; ++n) {
              mxfp4_fp8_tile<1>(m, n, N, K, K_packed, K_groups, x_fp32_ptr, q_ptr, s_ptr, scaled_table_fp8_ptr, out_ptr_typed);
            }
          }
        }
        return;
      }
#endif
      for (int64_t tile = begin; tile < end; ++tile) {
        const int64_t m = tile / n_tiles;
        const int64_t n_tile = tile % n_tiles;
        const int64_t n0 = n_tile * kTile;
        for (int64_t n = n0; n < std::min<int64_t>(n0 + kTile, N); ++n) {
          mxfp4_fp8_column_scalar(m, n, N, K, K_packed, K_groups, x_fp32_ptr, q_ptr, s_ptr, scaled_table_fp8_ptr, out_ptr_typed);
        }
      }
    });
  }

  return output;
}

// Permute MXFP4 weights into the VNNI 16-column x 4-K tile layout.  Storage
// stays MXFP4-sized (two 4-bit codes per byte + one E8M0 byte per 32 weights);
// only the byte order changes, so this is a one-time load-time transform.
std::vector<at::Tensor> mxfp4_prepack_vnni(
    const at::Tensor& qweight,
    const at::Tensor& scales) {
  TORCH_CHECK(qweight.device().is_cpu() && scales.device().is_cpu(), "tensors must be CPU");
  TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2, "qweight/scales must be 2D");
  TORCH_CHECK(qweight.scalar_type() == at::kByte, "qweight must be uint8");
  TORCH_CHECK(scales.scalar_type() == at::kByte, "scales must be uint8");

  const at::Tensor qw = qweight.contiguous();
  const at::Tensor sc = scales.contiguous();

  const int64_t N = qw.size(0);
  const int64_t K_packed = qw.size(1);
  const int64_t K = K_packed * 2;
  const int64_t K_groups = K / 32;
  TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");
  TORCH_CHECK(sc.size(0) == N && sc.size(1) == K_groups, "scales shape mismatch");

  const int64_t n_tiles = (N + kVnniNTile - 1) / kVnniNTile;
  at::Tensor qpack = at::zeros({n_tiles, K_groups, kVnniGroupBytes}, qw.options());
  at::Tensor spack = at::full({n_tiles, K_groups, kVnniNTile}, 127, sc.options());

  const uint8_t* q_ptr = qw.const_data_ptr<uint8_t>();
  const uint8_t* s_ptr = sc.const_data_ptr<uint8_t>();
  uint8_t* qp = qpack.data_ptr<uint8_t>();
  uint8_t* sp = spack.data_ptr<uint8_t>();

  at::parallel_for(0, n_tiles, 1, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const int64_t n0 = t * kVnniNTile;
      for (int64_t g = 0; g < K_groups; ++g) {
        uint8_t* q_grp = qp + (t * K_groups + g) * kVnniGroupBytes;
        uint8_t* s_grp = sp + (t * K_groups + g) * kVnniNTile;
        for (int64_t L = 0; L < kVnniNTile; ++L) {
          const int64_t n = n0 + L;
          if (n >= N) {
            continue;
          }
          s_grp[L] = s_ptr[n * K_groups + g];
          const uint8_t* q_row = q_ptr + n * K_packed;
          for (int c = 0; c < 4; ++c) {
            for (int j = 0; j < 4; ++j) {
              const int64_t k_lo = g * 32 + c * 8 + j;
              const int64_t k_hi = k_lo + 4;
              const uint8_t nib_lo = (k_lo & 1) ? (q_row[k_lo >> 1] >> 4) : (q_row[k_lo >> 1] & 0x0F);
              const uint8_t nib_hi = (k_hi & 1) ? (q_row[k_hi >> 1] >> 4) : (q_row[k_hi >> 1] & 0x0F);
              q_grp[c * 64 + L * 4 + j] =
                  static_cast<uint8_t>((nib_lo & 0x0F) | ((nib_hi & 0x0F) << 4));
            }
          }
        }
      }
    }
  });

  return {qpack, spack};
}

at::Tensor mxfp4_linear_cpu_vnni(
    const at::Tensor& input,
    const at::Tensor& qpack,
    const at::Tensor& spack,
    int64_t N,
    int64_t threads) {
  TORCH_CHECK(input.device().is_cpu(), "input must be CPU");
  TORCH_CHECK(input.dim() == 2, "input must be 2D (M, K)");
  TORCH_CHECK(input.scalar_type() == at::kFloat8_e4m3fn, "VNNI path requires FP8_E4M3 input");
  TORCH_CHECK(qpack.dim() == 3 && spack.dim() == 3, "packed tensors must be 3D");

  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  const int64_t K_groups = K / 32;
  const int64_t n_tiles = qpack.size(0);
  TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");
  TORCH_CHECK(qpack.size(1) == K_groups && qpack.size(2) == kVnniGroupBytes, "qpack shape mismatch");
  TORCH_CHECK(spack.size(0) == n_tiles && spack.size(1) == K_groups, "spack shape mismatch");
  TORCH_CHECK(N > 0 && N <= n_tiles * kVnniNTile, "N does not match packed tiles");

  const at::Tensor in = input.contiguous();
  const at::Tensor qp = qpack.contiguous();
  const at::Tensor sp = spack.contiguous();

  at::Tensor output = at::empty({M, N}, input.options());

  alignas(64) float fp8_to_fp32_table[256];
  for (int i = 0; i < 256; ++i) {
    fp8_to_fp32_table[i] = static_cast<float>(
        c10::Float8_e4m3fn(static_cast<uint8_t>(i), c10::Float8_e4m3fn::from_bits()));
  }
  // Weight codes are stored pre-multiplied by 2, so fold the 1/2 into the scale.
  alignas(64) float wscale_half_table[256];
  for (int i = 0; i < 256; ++i) {
    wscale_half_table[i] = e8m0_scale(static_cast<uint8_t>(i)) * 0.5f;
  }

  const uint8_t* x_u8 = reinterpret_cast<const uint8_t*>(in.const_data_ptr<c10::Float8_e4m3fn>());
  at::Tensor a_i8_t = at::empty({M, K}, at::kChar);
  at::Tensor a_scale_t = at::empty({M, K_groups}, at::kFloat);
  at::Tensor a_corr_t = at::empty({M, K_groups}, at::kInt);
  int8_t* a_i8 = a_i8_t.data_ptr<int8_t>();
  float* a_scale = a_scale_t.data_ptr<float>();
  int32_t* a_corr = a_corr_t.data_ptr<int32_t>();

#if GPTQMODEL_MXFP4_X86
  const bool quant_avx512 = mxfp4_cpu_supports_avx512();
#else
  constexpr bool quant_avx512 = false;
#endif
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t g = 0; g < K_groups; ++g) {
      const uint8_t* src = x_u8 + m * K + g * 32;
      int8_t* dst = a_i8 + m * K + g * 32;
      float* sc_out = a_scale + m * K_groups + g;
      int32_t* corr_out = a_corr + m * K_groups + g;
#if GPTQMODEL_MXFP4_X86
      if (quant_avx512) {
        mxfp4_quantize_activation_group_avx512(src, fp8_to_fp32_table, dst, sc_out, corr_out);
        continue;
      }
#else
      (void)quant_avx512;
#endif
      mxfp4_quantize_activation_group(src, fp8_to_fp32_table, dst, sc_out, corr_out);
    }
  }

  const uint8_t* qp_ptr = qp.const_data_ptr<uint8_t>();
  const uint8_t* sp_ptr = sp.const_data_ptr<uint8_t>();
  c10::Float8_e4m3fn* out_ptr = output.data_ptr<c10::Float8_e4m3fn>();

  const int64_t nthreads = clamped_threads(threads, M * N * K);
  const int64_t grain = std::max<int64_t>(1, n_tiles / nthreads);

  at::parallel_for(0, n_tiles, grain, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const int64_t n0 = t * kVnniNTile;
      const uint8_t* q_tile = qp_ptr + t * K_groups * kVnniGroupBytes;
      const uint8_t* s_tile = sp_ptr + t * K_groups * kVnniNTile;
#if GPTQMODEL_MXFP4_X86
      if (mxfp4_cpu_supports_vnni()) {
        int64_t m0 = 0;
        for (; m0 + 4 <= M; m0 += 4) {
          mxfp4_vnni_tile<4>(n0, N, m0, K, K_groups, a_i8, a_scale, a_corr,
                             q_tile, s_tile, wscale_half_table, out_ptr);
        }
        for (; m0 + 2 <= M; m0 += 2) {
          mxfp4_vnni_tile<2>(n0, N, m0, K, K_groups, a_i8, a_scale, a_corr,
                             q_tile, s_tile, wscale_half_table, out_ptr);
        }
        for (; m0 < M; ++m0) {
          mxfp4_vnni_tile<1>(n0, N, m0, K, K_groups, a_i8, a_scale, a_corr,
                             q_tile, s_tile, wscale_half_table, out_ptr);
        }
        continue;
      }
#endif
      mxfp4_vnni_tile_scalar(n0, N, M, K, K_groups, a_i8, a_scale, a_corr,
                             q_tile, s_tile, wscale_half_table, out_ptr);
    }
  });

  return output;
}

} // namespace gptqmodel_mxfp4

TORCH_LIBRARY(gptqmodel_mxfp4, m) {
  m.def(
      "mxfp4_linear_cpu(Tensor input, Tensor qweight, Tensor scales, int threads=0, int variant=0, int fp16_flush=0) "
      "-> Tensor");
  m.def("mxfp4_prepack_vnni(Tensor qweight, Tensor scales) -> Tensor[]");
  m.def("mxfp4_linear_cpu_vnni(Tensor input, Tensor qpack, Tensor spack, int N, int threads) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_mxfp4, CPU, m) {
  m.impl("mxfp4_linear_cpu", TORCH_FN(gptqmodel_mxfp4::mxfp4_linear_cpu));
  m.impl("mxfp4_prepack_vnni", TORCH_FN(gptqmodel_mxfp4::mxfp4_prepack_vnni));
  m.impl("mxfp4_linear_cpu_vnni", TORCH_FN(gptqmodel_mxfp4::mxfp4_linear_cpu_vnni));
}
