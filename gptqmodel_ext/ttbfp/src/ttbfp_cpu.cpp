// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
/*
 * ttbfp - CPU kernels for Tenstorrent BFP8_b / BFP4_b emulation.
 *
 * Reference semantics:
 *   tt_metal/tt-llk/tests/python_tests/helpers/pack.py
 *   tt_metal/tt-llk/tests/python_tests/helpers/unpack.py
 *   tt_metal/tt-llk/tests/python_tests/helpers/bfp_format_utils.py
 *
 * Hot loops exist in a portable scalar form and an x86 AVX2+FMA form chosen
 * at runtime.  OpenMP parallelizes independent output tiles / rows.
 *
 * Threading
 * ---------
 * Every parallel region is a loop over independent output tiles (or rows)
 * with no cross-thread reduction, so results are bitwise independent of the
 * thread count.  This library never calls omp_set_*; the caller owns the
 * pool.  Two settings matter when running or measuring:
 *
 *   OMP_NUM_THREADS=<cores>   do not use more threads than cores you own
 *   OMP_WAIT_POLICY=passive   stop idle workers from spinning
 *
 * GNU OpenMP spins by default after a parallel region ends, for a long time
 * (hundreds of ms with the default GOMP_SPINCOUNT).  Back-to-back kernels
 * then burn a full core per idle worker: an 8-thread run on a 32-core box
 * reports roughly 8x the CPU time, and the spinners steal issue slots and
 * cache from the next kernel.  With OMP_WAIT_POLICY=passive the workers
 * sleep and CPU time tracks actual work.  It affects performance only,
 * never numerics.
 *
 * Because of that spinning, wall-clock timing on a busy shared host is
 * misleading (samples quantize to ~1 ms); measure with process CPU time.
 */

#include "ttbfp.h"

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define TTBFP_X86 1
#else
#  define TTBFP_X86 0
#endif

#if TTBFP_X86 && (defined(__GNUC__) || defined(__clang__))
#  define TTBFP_AVX2 __attribute__((target("avx2,fma")))
#  define TTBFP_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")))
#else
#  define TTBFP_AVX2
#  define TTBFP_AVX512
#endif

namespace {

struct ScaleTables {
    float bf8[256];
    float bf4[256];
    float bf2[256];
    ScaleTables() {
        bf8[0] = 0.0f;
        bf4[0] = 0.0f;
        bf2[0] = 0.0f;
        for (int e = 1; e < 256; ++e) {
            // decode: mag * 2^(e - 126 - magnitude_bits)
            bf8[e] = std::ldexp(1.0f, e - 133);
            bf4[e] = std::ldexp(1.0f, e - 129);
            bf2[e] = std::ldexp(1.0f, e - 127);
        }
    }
};

inline const ScaleTables &scales() {
    static const ScaleTables t;
    return t;
}

// Per-format block scale: mag * 2^(e - 126 - magnitude_bits).
inline const float *scale_table(ttbfp_fmt fmt) {
    return fmt == TTBFP_TTBF8 ? scales().bf8
         : fmt == TTBFP_TTBF4 ? scales().bf4
                              : scales().bf2;
}

inline int have_avx2_fma() {
#if TTBFP_X86 && (defined(__GNUC__) || defined(__clang__))
    static int cached = -1;
    if (cached < 0) {
        const char *force = std::getenv("TTBFP_FORCE_SCALAR");
        if (force && *force && *force != '0') {
            cached = 0;
        } else {
            cached = (__builtin_cpu_supports("avx2") &&
                      __builtin_cpu_supports("fma")) ? 1 : 0;
        }
    }
    return cached;
#else
    return 0;
#endif
}

inline int have_avx512() {
#if TTBFP_X86 && (defined(__GNUC__) || defined(__clang__))
    static int cached = -1;
    if (cached < 0) {
        const char *force_scalar = std::getenv("TTBFP_FORCE_SCALAR");
        const char *force_avx2 = std::getenv("TTBFP_FORCE_AVX2");
        if ((force_scalar && *force_scalar && *force_scalar != '0') ||
            (force_avx2 && *force_avx2 && *force_avx2 != '0')) {
            cached = 0;
        } else {
            cached = (__builtin_cpu_supports("avx512f") &&
                      __builtin_cpu_supports("avx512bw") &&
                      __builtin_cpu_supports("avx512dq") &&
                      __builtin_cpu_supports("avx512vl")) ? 1 : 0;
        }
    }
    return cached;
#else
    return 0;
#endif
}

inline int datum_bytes(ttbfp_fmt fmt) {
    return fmt == TTBFP_TTBF8 ? 16 : (fmt == TTBFP_TTBF4 ? 8 : 4);
}

inline int magnitude_bits(ttbfp_fmt fmt) {
    return fmt == TTBFP_TTBF8 ? 7 : (fmt == TTBFP_TTBF4 ? 3 : 1);
}

/* ------------------------------------------------------------------ */
/* Quantizer (bit-exact with tt-llk pack.py)                          */
/* ------------------------------------------------------------------ */

inline void quantize_block(const float *x, int n, int mb,
                           uint8_t *exp_out, uint8_t *dat_out) {
    int32_t exps[TTBFP_BLOCK];
    int32_t mants[TTBFP_BLOCK];
    int32_t signs[TTBFP_BLOCK];

    int maxe = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t b;
        std::memcpy(&b, &x[i], sizeof(b));
        // The reference packer truncates fp32 -> bf16 (top 16 bits).
        uint32_t bf = b >> 16;
        // Quiet signaling NaNs (bit 22 of the fp32 pattern == bit 6 of bf16).
        if (((bf >> 7) & 0xFF) == 0xFF && (bf & 0x7F) != 0) bf |= 0x0040u;
        int e = (int)((bf >> 7) & 0xFF);
        exps[i] = e;
        signs[i] = (int)(bf >> 15);
        mants[i] = (int)(((bf & 0x7F) >> 1) | 0x40);
        if (e > maxe) maxe = e;
    }

    const int shift_cap = 8;  // BFP_MAX_ALIGNMENT_SHIFT
    *exp_out = (uint8_t)maxe;
    uint8_t dat[TTBFP_BLOCK];
    std::memset(dat, 0, sizeof(dat));
    for (int i = 0; i < n; ++i) {
        int d = maxe - exps[i];
        int shift = d > shift_cap ? shift_cap : d;
        int guard = 0;
        if (d > 0) {
            int gs = d - 1;
            if (gs > shift_cap) gs = shift_cap;
            guard = (mants[i] >> gs) & 1;
        }
        int q = ((mants[i] >> shift) + guard) & 0x7F;
        q >>= (7 - mb);
        // A sign-only datum would decode to -0 (and to -inf in hardware),
        // so the reference drops the sign when the magnitude is zero.
        dat[i] = q ? (uint8_t)((signs[i] << mb) | q) : (uint8_t)0;
    }
    if (mb == 7) {
        std::memcpy(dat_out, dat, (size_t)n);
    } else if (mb == 3) {
        const int nbytes = (n + 1) / 2;
        for (int i = 0; i < nbytes; ++i) {
            uint8_t lo = dat[2 * i];
            uint8_t hi = (2 * i + 1 < n) ? dat[2 * i + 1] : 0;
            dat_out[i] = (uint8_t)(lo | (hi << 4));
        }
    } else {  // mb == 1: four 2-bit datums per byte, lowest datum first
        const int nbytes = (n + 3) / 4;
        for (int i = 0; i < nbytes; ++i) {
            uint8_t v = 0;
            for (int k = 0; k < 4; ++k) {
                const int idx = 4 * i + k;
                if (idx < n) v = (uint8_t)(v | (dat[idx] << (2 * k)));
            }
            dat_out[i] = v;
        }
    }
}

void quantize_rows(ttbfp_fmt fmt, const float *src, int64_t rows, int64_t cols,
                   uint8_t *exps, uint8_t *data) {
    const int mb = magnitude_bits(fmt);
    const int db = datum_bytes(fmt);
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(cols);

#pragma omp parallel for schedule(static) if (rows > 1)
    for (int64_t r = 0; r < rows; ++r) {
        const float *srow = src + r * cols;
        uint8_t *erow = exps + r * nblk;
        uint8_t *drow = data + r * nblk * db;
        for (int64_t j = 0; j < nblk; ++j) {
            int n = (int)std::min<int64_t>(TTBFP_BLOCK, cols - j * TTBFP_BLOCK);
            std::memset(drow + j * db, 0, (size_t)db);
            if (n > 0) {
                quantize_block(srow + j * TTBFP_BLOCK, n, mb,
                               erow + j, drow + j * db);
            } else {
                erow[j] = 0;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scalar decode                                                       */
/* ------------------------------------------------------------------ */

inline void decode_block_scalar(ttbfp_fmt fmt, uint8_t e, const uint8_t *d,
                                float *out) {
    if (e == 0) {
        for (int i = 0; i < TTBFP_BLOCK; ++i) out[i] = 0.0f;
        return;
    }
    const float scale = scale_table(fmt)[e];
    const int mb = magnitude_bits(fmt);
    const int mask = (1 << mb) - 1;
    const int sbit = 1 << mb;
    if (fmt == TTBFP_TTBF8) {
        for (int i = 0; i < TTBFP_BLOCK; ++i) {
            uint8_t v = d[i];
            float f = (float)(v & mask) * scale;
            out[i] = (v & sbit) ? -f : f;
        }
    } else if (fmt == TTBFP_TTBF4) {
        for (int i = 0; i < TTBFP_BLOCK; ++i) {
            uint8_t b = d[i >> 1];
            uint8_t v = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
            float f = (float)(v & mask) * scale;
            out[i] = (v & sbit) ? -f : f;
        }
    } else {
        for (int i = 0; i < TTBFP_BLOCK; ++i) {
            uint8_t b = d[i >> 2];
            uint8_t v = (uint8_t)((b >> ((i & 3) * 2)) & 0x03);
            float f = (float)(v & mask) * scale;
            out[i] = (v & sbit) ? -f : f;
        }
    }
}

void dequantize_rows_scalar(ttbfp_fmt fmt, const uint8_t *exps,
                            const uint8_t *data, int64_t rows, int64_t cols,
                            float *dst, int ftz) {
    const int db = datum_bytes(fmt);
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(cols);
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t *erow = exps + r * nblk;
        const uint8_t *drow = data + r * nblk * db;
        float *orow = dst + r * cols;
        for (int64_t j = 0; j < nblk; ++j) {
            float tmp[TTBFP_BLOCK];
            decode_block_scalar(fmt, erow[j], drow + j * db, tmp);
            if (ftz && erow[j] < 7) {
                for (int i = 0; i < TTBFP_BLOCK; ++i) {
                    if (std::fabs(tmp[i]) < FLT_MIN) tmp[i] = 0.0f;
                }
            }
            int n = (int)std::min<int64_t>(TTBFP_BLOCK, cols - j * TTBFP_BLOCK);
            std::memcpy(orow + j * TTBFP_BLOCK, tmp, (size_t)n * sizeof(float));
        }
    }
}

/* ------------------------------------------------------------------ */
/* AVX2 decode                                                         */
/* ------------------------------------------------------------------ */

#if TTBFP_X86

TTBFP_AVX2 inline __m256 bf8_decode8(const uint8_t *p) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
    __m256i v = _mm256_cvtepu8_epi32(b);
    __m256i mag = _mm256_and_si256(v, _mm256_set1_epi32(0x7F));
    __m256i sgn = _mm256_slli_epi32(
        _mm256_and_si256(v, _mm256_set1_epi32(0x80)), 24);
    return _mm256_xor_ps(_mm256_cvtepi32_ps(mag), _mm256_castsi256_ps(sgn));
}

TTBFP_AVX2 inline __m256 bf4_sign_mag(__m256i v) {
    __m256i mag = _mm256_and_si256(v, _mm256_set1_epi32(7));
    __m256i sgn = _mm256_slli_epi32(
        _mm256_and_si256(v, _mm256_set1_epi32(8)), 28);
    return _mm256_xor_ps(_mm256_cvtepi32_ps(mag), _mm256_castsi256_ps(sgn));
}

TTBFP_AVX2 inline void bf4_decode16(const uint8_t *p, __m256 *lo, __m256 *hi) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
    __m128i m = _mm_set1_epi8(0x0F);
    __m128i l = _mm_and_si128(b, m);
    __m128i h = _mm_and_si128(_mm_srli_epi16(b, 4), m);
    __m128i inter = _mm_unpacklo_epi8(l, h);  // l0,h0,l1,h1,...
    __m256i v0 = _mm256_cvtepu8_epi32(inter);
    __m256i v1 = _mm256_cvtepu8_epi32(_mm_srli_si128(inter, 8));
    *lo = bf4_sign_mag(v0);
    *hi = bf4_sign_mag(v1);
}

// 16 two-bit datums in 4 bytes -> two 8-lane vectors.  Each byte holds four
// datums, lowest datum in bits[1:0]; the byte order matches tt-llk's
// pack_bfp2_b.
TTBFP_AVX2 inline void bf2_decode16(const uint8_t *p, __m256 *lo, __m256 *hi) {
    uint32_t w;
    std::memcpy(&w, p, sizeof(w));
    __m128i b = _mm_cvtsi32_si128((int)w);
    __m128i m2 = _mm_set1_epi8(0x03);
    __m128i s0 = _mm_and_si128(b, m2);
    __m128i s1 = _mm_and_si128(_mm_srli_epi16(b, 2), m2);
    __m128i s2 = _mm_and_si128(_mm_srli_epi16(b, 4), m2);
    __m128i s3 = _mm_and_si128(_mm_srli_epi16(b, 6), m2);
    __m128i a01 = _mm_unpacklo_epi8(s0, s1);
    __m128i a23 = _mm_unpacklo_epi8(s2, s3);
    __m128i inter = _mm_unpacklo_epi16(a01, a23);  // 16 datums, even first
    __m256i v0 = _mm256_cvtepu8_epi32(inter);
    __m256i v1 = _mm256_cvtepu8_epi32(_mm_srli_si128(inter, 8));
    __m256i one = _mm256_set1_epi32(1);
    __m256i two = _mm256_set1_epi32(2);
    __m256i mag0 = _mm256_and_si256(v0, one);
    __m256i sgn0 = _mm256_slli_epi32(_mm256_and_si256(v0, two), 30);
    __m256i mag1 = _mm256_and_si256(v1, one);
    __m256i sgn1 = _mm256_slli_epi32(_mm256_and_si256(v1, two), 30);
    *lo = _mm256_xor_ps(_mm256_cvtepi32_ps(mag0), _mm256_castsi256_ps(sgn0));
    *hi = _mm256_xor_ps(_mm256_cvtepi32_ps(mag1), _mm256_castsi256_ps(sgn1));
}

TTBFP_AVX2 inline __m256 flush_denorm(__m256 v) {
    __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 a = _mm256_andnot_ps(sign, v);
    __m256 keep = _mm256_cmp_ps(a, _mm256_set1_ps(FLT_MIN), _CMP_GE_OQ);
    return _mm256_and_ps(v, keep);
}

TTBFP_AVX2 void dequantize_row_avx2(ttbfp_fmt fmt, const uint8_t *exps,
                                    const uint8_t *data, int64_t cols,
                                    float *dst, int ftz) {
    const int db = datum_bytes(fmt);
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(cols);
    const float *tab = scale_table(fmt);
    for (int64_t j = 0; j < nblk; ++j) {
        uint8_t e = exps[j];
        if (e == 0) {
            int n = (int)std::min<int64_t>(TTBFP_BLOCK, cols - j * TTBFP_BLOCK);
            std::memset(dst + j * TTBFP_BLOCK, 0, (size_t)n * sizeof(float));
            continue;
        }
        __m256 s = _mm256_set1_ps(tab[e]);
        const uint8_t *p = data + j * db;
        __m256 v0, v1;
        if (fmt == TTBFP_TTBF8) {
            v0 = bf8_decode8(p);
            v1 = bf8_decode8(p + 8);
        } else if (fmt == TTBFP_TTBF4) {
            bf4_decode16(p, &v0, &v1);
        } else {
            bf2_decode16(p, &v0, &v1);
        }
        v0 = _mm256_mul_ps(v0, s);
        v1 = _mm256_mul_ps(v1, s);
        if (ftz && e < 7) {
            v0 = flush_denorm(v0);
            v1 = flush_denorm(v1);
        }
        float *o = dst + j * TTBFP_BLOCK;
        if (j * TTBFP_BLOCK + TTBFP_BLOCK <= cols) {
            _mm256_storeu_ps(o, v0);
            _mm256_storeu_ps(o + 8, v1);
        } else {
            alignas(32) float tmp[TTBFP_BLOCK];
            _mm256_store_ps(tmp, v0);
            _mm256_store_ps(tmp + 8, v1);
            int n = (int)std::min<int64_t>(TTBFP_BLOCK, cols - j * TTBFP_BLOCK);
            std::memcpy(o, tmp, (size_t)n * sizeof(float));
        }
    }
}

/* ------------------------------------------------------------------ */
/* AVX-512 decode (16 datums per vector)                               */
/* ------------------------------------------------------------------ */

TTBFP_AVX512 inline __m512 bf8_decode16_512(const uint8_t *p) {
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    __m512i v = _mm512_cvtepu8_epi32(b);
    __m512i mag = _mm512_and_si512(v, _mm512_set1_epi32(0x7F));
    __m512i sgn = _mm512_slli_epi32(
        _mm512_and_si512(v, _mm512_set1_epi32(0x80)), 24);
    return _mm512_xor_ps(_mm512_cvtepi32_ps(mag), _mm512_castsi512_ps(sgn));
}

TTBFP_AVX512 inline __m512 bf4_decode16_512(const uint8_t *p) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
    __m128i m = _mm_set1_epi8(0x0F);
    __m128i l = _mm_and_si128(b, m);
    __m128i h = _mm_and_si128(_mm_srli_epi16(b, 4), m);
    __m128i inter = _mm_unpacklo_epi8(l, h);  // 16 nibbles, even first
    __m512i v = _mm512_cvtepu8_epi32(inter);
    __m512i mag = _mm512_and_si512(v, _mm512_set1_epi32(7));
    __m512i sgn = _mm512_slli_epi32(
        _mm512_and_si512(v, _mm512_set1_epi32(8)), 28);
    return _mm512_xor_ps(_mm512_cvtepi32_ps(mag), _mm512_castsi512_ps(sgn));
}

TTBFP_AVX512 inline __m512 bf2_decode16_512(const uint8_t *p) {
    uint32_t w;
    std::memcpy(&w, p, sizeof(w));
    __m128i b = _mm_cvtsi32_si128((int)w);
    __m128i m2 = _mm_set1_epi8(0x03);
    __m128i s0 = _mm_and_si128(b, m2);
    __m128i s1 = _mm_and_si128(_mm_srli_epi16(b, 2), m2);
    __m128i s2 = _mm_and_si128(_mm_srli_epi16(b, 4), m2);
    __m128i s3 = _mm_and_si128(_mm_srli_epi16(b, 6), m2);
    __m128i a01 = _mm_unpacklo_epi8(s0, s1);
    __m128i a23 = _mm_unpacklo_epi8(s2, s3);
    __m128i inter = _mm_unpacklo_epi16(a01, a23);  // 16 datums, even first
    __m512i v = _mm512_cvtepu8_epi32(inter);
    __m512i mag = _mm512_and_si512(v, _mm512_set1_epi32(1));
    __m512i sgn = _mm512_slli_epi32(
        _mm512_and_si512(v, _mm512_set1_epi32(2)), 30);
    return _mm512_xor_ps(_mm512_cvtepi32_ps(mag), _mm512_castsi512_ps(sgn));
}

TTBFP_AVX512 inline __m512 flush_denorm_512(__m512 v) {
    __m512 a = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), v);
    __mmask16 keep = _mm512_cmp_ps_mask(a, _mm512_set1_ps(FLT_MIN), _CMP_GE_OQ);
    return _mm512_maskz_mov_ps(keep, v);
}

TTBFP_AVX512 void dequantize_row_avx512(ttbfp_fmt fmt, const uint8_t *exps,
                                        const uint8_t *data, int64_t cols,
                                        float *dst, int ftz) {
    const int db = datum_bytes(fmt);
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(cols);
    const float *tab = scale_table(fmt);
    for (int64_t j = 0; j < nblk; ++j) {
        uint8_t e = exps[j];
        int n = (int)std::min<int64_t>(TTBFP_BLOCK, cols - j * TTBFP_BLOCK);
        float *o = dst + j * TTBFP_BLOCK;
        if (e == 0) {
            std::memset(o, 0, (size_t)n * sizeof(float));
            continue;
        }
        const uint8_t *p = data + j * db;
        __m512 v = fmt == TTBFP_TTBF8 ? bf8_decode16_512(p)
                 : fmt == TTBFP_TTBF4 ? bf4_decode16_512(p)
                                      : bf2_decode16_512(p);
        v = _mm512_mul_ps(v, _mm512_set1_ps(tab[e]));
        if (ftz && e < 7) v = flush_denorm_512(v);
        if (n == TTBFP_BLOCK) {
            _mm512_storeu_ps(o, v);
        } else {
            alignas(64) float tmp[TTBFP_BLOCK];
            _mm512_store_ps(tmp, v);
            std::memcpy(o, tmp, (size_t)n * sizeof(float));
        }
    }
}

#endif /* TTBFP_X86 */

/* ------------------------------------------------------------------ */
/* Matmul microkernels                                                 */
/*                                                                     */
/*   MR rows of A  x  J consecutive 16-wide packed-B blocks, all of K. */
/*                                                                     */
/*   A is staged through a small float panel, so its loads are         */
/*   sequential and each A element is broadcast once and reused for J  */
/*   B blocks.  J blocks are decoded per k, so a 64-byte cache line of */
/*   packed B is consumed completely.                                  */
/*                                                                     */
/*   The k order of accumulation is unchanged, so every variant        */
/*   (scalar / AVX2 / AVX-512, any J, any thread count) stays bitwise  */
/*   identical.  The same kernel serves both the row-major and the     */
/*   block-major (transposed) packed layouts; only the strides differ. */
/* ------------------------------------------------------------------ */

struct FloatLoader {
    static inline float load(const float *p, int64_t i) { return p[i]; }
};

struct BF16Loader {
    static inline float load(const uint16_t *p, int64_t i) {
        uint32_t bits = (uint32_t)p[i] << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

constexpr int kAPanel = 32;  // A panel depth along K

/* ---- datum decoders ------------------------------------------------ */
/*
 * The matmul kernels are parameterized on how a 16-datum block becomes a
 * vector of signed floats.  Two representations are supported:
 *
 *   SmDec - the TT on-device layout: 1 sign bit + magnitude bits, sign and
 *           magnitude in separate bit fields (8 uops per ttbf8 block).
 *   S8Dec - a kernel-native layout storing sign*magnitude as a signed int8
 *           (3 uops per block: vpmovsxbd, vcvtdq2ps, vmulps).
 *
 * Both decode to exactly the same values, so results are bitwise identical;
 * S8Dec just moves the sign into the datum.  This is the same trade ATen
 * makes for _weight_int4pack_mm / _weight_int8pack_mm: pack the weights into
 * whatever shape makes the inner loop cheapest.  See ttbfp_convert_s8().
 */
struct SmDec {
    static inline void raw_scalar(ttbfp_fmt fmt, const uint8_t *d, float *out) {
        const int mb = magnitude_bits(fmt);
        const int mask = (1 << mb) - 1;
        const int sbit = 1 << mb;
        if (fmt == TTBFP_TTBF8) {
            for (int i = 0; i < TTBFP_BLOCK; ++i) {
                int v = d[i];
                float m = (float)(v & mask);
                out[i] = (v & sbit) ? -m : m;
            }
        } else {
            for (int i = 0; i < TTBFP_BLOCK; ++i) {
                int v = (i & 1) ? (d[i >> 1] >> 4) : (d[i >> 1] & 0x0F);
                float m = (float)(v & mask);
                out[i] = (v & sbit) ? -m : m;
            }
        }
    }
#if TTBFP_X86
    TTBFP_AVX2 static inline void raw_avx2(ttbfp_fmt fmt, const uint8_t *p,
                                           __m256 *lo, __m256 *hi) {
        if (fmt == TTBFP_TTBF8) {
            *lo = bf8_decode8(p);
            *hi = bf8_decode8(p + 8);
        } else if (fmt == TTBFP_TTBF4) {
            bf4_decode16(p, lo, hi);
        } else {
            bf2_decode16(p, lo, hi);
        }
    }
    TTBFP_AVX512 static inline __m512 raw_avx512(ttbfp_fmt fmt, const uint8_t *p) {
        return fmt == TTBFP_TTBF8 ? bf8_decode16_512(p)
             : fmt == TTBFP_TTBF4 ? bf4_decode16_512(p)
                                  : bf2_decode16_512(p);
    }
#endif
};

struct S8Dec {
    static inline void raw_scalar(ttbfp_fmt, const uint8_t *d, float *out) {
        for (int i = 0; i < TTBFP_BLOCK; ++i) out[i] = (float)(int8_t)d[i];
    }
#if TTBFP_X86
    TTBFP_AVX2 static inline void raw_avx2(ttbfp_fmt, const uint8_t *p,
                                           __m256 *lo, __m256 *hi) {
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
        *lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        *hi = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(b, 8)));
    }
    TTBFP_AVX512 static inline __m512 raw_avx512(ttbfp_fmt, const uint8_t *p) {
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
        return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b));
    }
#endif
};

/* ---- scalar panel -------------------------------------------------- */

template <int MR, int J, class AT, class Loader, class Dec>
void mm_panel_scalar(ttbfp_fmt fmt, const AT *A, int64_t lda,
                     const uint8_t *e0, const uint8_t *d0,
                     int64_t e_k, int64_t d_k, int64_t e_j, int64_t d_j,
                     int64_t K, float *C, int64_t ldc, int ftz) {
    float acc[MR][J][TTBFP_BLOCK];
    std::memset(acc, 0, sizeof(acc));
    alignas(64) float ap[MR][kAPanel];
    const float *tab = scale_table(fmt);

    for (int64_t k0 = 0; k0 < K; k0 += kAPanel) {
        const int kb = (int)std::min<int64_t>(kAPanel, K - k0);
        for (int r = 0; r < MR; ++r) {
            for (int t = 0; t < kb; ++t) ap[r][t] = Loader::load(A + r * lda + k0, t);
        }
        for (int t = 0; t < kb; ++t) {
            const int64_t k = k0 + t;
            float b[J][TTBFP_BLOCK];
            for (int jj = 0; jj < J; ++jj) {
                uint8_t e = e0[k * e_k + jj * e_j];
                Dec::raw_scalar(fmt, d0 + k * d_k + jj * d_j, b[jj]);
                const float sc = tab[e];
                for (int x = 0; x < TTBFP_BLOCK; ++x) b[jj][x] *= sc;
                if (ftz && e < 7) {
                    for (int x = 0; x < TTBFP_BLOCK; x += 1) {
                        if (std::fabs(b[jj][x]) < FLT_MIN) b[jj][x] = 0.0f;
                    }
                }
            }
            for (int r = 0; r < MR; ++r) {
                const float a = ap[r][t];
                for (int jj = 0; jj < J; ++jj) {
                    for (int x = 0; x < TTBFP_BLOCK; x += 1) {
                        acc[r][jj][x] = std::fma(a, b[jj][x], acc[r][jj][x]);
                    }
                }
            }
        }
    }
    for (int r = 0; r < MR; ++r) {
        for (int jj = 0; jj < J; ++jj) {
            std::memcpy(C + r * ldc + jj * TTBFP_BLOCK, acc[r][jj],
                        TTBFP_BLOCK * sizeof(float));
        }
    }
}

#if TTBFP_X86

/* ---- AVX2 panel ---------------------------------------------------- */

template <int MR, int J, class AT, class Loader, class Dec>
TTBFP_AVX2 void mm_panel_avx2(ttbfp_fmt fmt_, const AT *A, int64_t lda,
                              const uint8_t *e0, const uint8_t *d0,
                              int64_t e_k, int64_t d_k, int64_t e_j, int64_t d_j,
                              int64_t K, float *C, int64_t ldc, int ftz) {
    static_assert(J <= 2, "AVX2 panel supports at most 2 B blocks");
    const ttbfp_fmt fmt = fmt_;
    __m256 acc[MR][J][2];
    for (int r = 0; r < MR; ++r) {
        for (int jj = 0; jj < J; ++jj) {
            acc[r][jj][0] = _mm256_setzero_ps();
            acc[r][jj][1] = _mm256_setzero_ps();
        }
    }
    alignas(64) float ap[MR][kAPanel];
    const float *tab = scale_table(fmt);

    for (int64_t k0 = 0; k0 < K; k0 += kAPanel) {
        const int kb = (int)std::min<int64_t>(kAPanel, K - k0);
        for (int r = 0; r < MR; ++r) {
            for (int t = 0; t < kb; ++t) ap[r][t] = Loader::load(A + r * lda + k0, t);
        }
        for (int t = 0; t < kb; ++t) {
            const int64_t k = k0 + t;
            __m256 b[J][2];
            for (int jj = 0; jj < J; ++jj) {
                uint8_t e = e0[k * e_k + jj * e_j];
                __m256 v0, v1;
                Dec::raw_avx2(fmt, d0 + k * d_k + jj * d_j, &v0, &v1);
                __m256 s = _mm256_set1_ps(tab[e]);
                v0 = _mm256_mul_ps(v0, s);
                v1 = _mm256_mul_ps(v1, s);
                if (ftz && e < 7) {
                    v0 = flush_denorm(v0);
                    v1 = flush_denorm(v1);
                }
                b[jj][0] = v0;
                b[jj][1] = v1;
            }
            for (int r = 0; r < MR; ++r) {
                __m256 a = _mm256_set1_ps(ap[r][t]);
                for (int jj = 0; jj < J; ++jj) {
                    acc[r][jj][0] = _mm256_fmadd_ps(a, b[jj][0], acc[r][jj][0]);
                    acc[r][jj][1] = _mm256_fmadd_ps(a, b[jj][1], acc[r][jj][1]);
                }
            }
        }
    }
    for (int r = 0; r < MR; ++r) {
        for (int jj = 0; jj < J; ++jj) {
            _mm256_storeu_ps(C + r * ldc + jj * TTBFP_BLOCK, acc[r][jj][0]);
            _mm256_storeu_ps(C + r * ldc + jj * TTBFP_BLOCK + 8, acc[r][jj][1]);
        }
    }
}

/* ---- AVX-512 panel ------------------------------------------------- */

// Decode the J column blocks at row k of the (block-major) packed B into
// J float vectors.  Pulled out of the panel so the software pipeline below
// can call it for k+1 while the FMAs for k are still in flight; it must
// carry the target attribute itself because a lambda would not inherit it.
template <int J, class Dec>
TTBFP_AVX512 inline void decode_blocks_avx512(ttbfp_fmt fmt, const float *tab,
                                              const uint8_t *e0,
                                              const uint8_t *d0,
                                              int64_t e_k, int64_t d_k,
                                              int64_t e_j, int64_t d_j,
                                              int64_t k, int ftz, __m512 *b) {
    for (int jj = 0; jj < J; ++jj) {
        const uint8_t e = e0[k * e_k + jj * e_j];
        __m512 v = Dec::raw_avx512(fmt, d0 + k * d_k + jj * d_j);
        v = _mm512_mul_ps(v, _mm512_set1_ps(tab[e]));
        if (ftz && e < 7) v = flush_denorm_512(v);
        b[jj] = v;
    }
}

// The FMA half of the panel: one A broadcast per row, reused by all J
// column blocks.  Factored out so the pipelined and plain k loops below
// share it.
template <int MR, int J>
TTBFP_AVX512 inline void fma_panel(__m512 (&acc)[MR][J],
                                   const float (&ap)[MR][kAPanel], int t,
                                   const __m512 *b) {
    for (int r = 0; r < MR; ++r) {
        const __m512 a = _mm512_set1_ps(ap[r][t]);
        for (int jj = 0; jj < J; ++jj)
            acc[r][jj] = _mm512_fmadd_ps(a, b[jj], acc[r][jj]);
    }
}

template <int MR, int J, class AT, class Loader, class Dec>
TTBFP_AVX512 void mm_panel_avx512(ttbfp_fmt fmt_, const AT *A, int64_t lda,
                                  const uint8_t *e0, const uint8_t *d0,
                                  int64_t e_k, int64_t d_k, int64_t e_j, int64_t d_j,
                                  int64_t K, float *C, int64_t ldc, int ftz) {
    const ttbfp_fmt fmt = fmt_;
    __m512 acc[MR][J];
    for (int r = 0; r < MR; ++r) {
        for (int jj = 0; jj < J; ++jj) acc[r][jj] = _mm512_setzero_ps();
    }
    alignas(64) float ap[MR][kAPanel];
    const float *tab = scale_table(fmt);

    for (int64_t k0 = 0; k0 < K; k0 += kAPanel) {
        const int kb = (int)std::min<int64_t>(kAPanel, K - k0);
        for (int r = 0; r < MR; ++r) {
            for (int t = 0; t < kb; ++t) ap[r][t] = Loader::load(A + r * lda + k0, t);
        }
        // Software pipeline: decode B for k+1 while the FMAs for k run, so
        // the decode latency hides under the FMA stream instead of stalling
        // it.  Only independent work moves; the k order of the accumulation
        // is untouched, so the output stays bitwise identical to the scalar
        // and AVX2 kernels.
        //
        // Only worth it for J<=2: at J=4 the extra live B vectors cost more
        // registers than the overlap buys (measured ~15% slower at M=4).
        if constexpr (J <= 2) {
            __m512 bn[J];
            decode_blocks_avx512<J, Dec>(fmt, tab, e0, d0, e_k, d_k, e_j, d_j,
                                         k0, ftz, bn);
            for (int t = 0; t < kb; ++t) {
                __m512 b[J];
                for (int jj = 0; jj < J; ++jj) b[jj] = bn[jj];
                if (t + 1 < kb)
                    decode_blocks_avx512<J, Dec>(fmt, tab, e0, d0, e_k, d_k,
                                                 e_j, d_j, k0 + t + 1, ftz, bn);
                fma_panel<MR, J>(acc, ap, t, b);
            }
        } else {
            for (int t = 0; t < kb; ++t) {
                __m512 b[J];
                decode_blocks_avx512<J, Dec>(fmt, tab, e0, d0, e_k, d_k, e_j,
                                             d_j, k0 + t, ftz, b);
                fma_panel<MR, J>(acc, ap, t, b);
            }
        }
    }
    for (int r = 0; r < MR; ++r) {
        for (int jj = 0; jj < J; ++jj) {
            _mm512_storeu_ps(C + r * ldc + jj * TTBFP_BLOCK, acc[r][jj]);
        }
    }
}

#endif /* TTBFP_X86 */

/* ---- ISA dispatch for one panel ----------------------------------- */

template <int MR, int J, class AT, class Loader, class Dec>
inline void mm_panel(ttbfp_fmt fmt_, const AT *A, int64_t lda,
                     const uint8_t *e0, const uint8_t *d0,
                     int64_t e_k, int64_t d_k, int64_t e_j, int64_t d_j,
                     int64_t K, int64_t j0, float *C, int64_t ldc, int ftz,
                     int level) {
    const uint8_t *ep = e0 + j0 * e_j;
    const uint8_t *dp = d0 + j0 * d_j;
#if TTBFP_X86
    if (level >= 2) {
        mm_panel_avx512<MR, J, AT, Loader, Dec>(fmt_, A, lda, ep, dp, e_k, d_k,
                                                e_j, d_j, K, C, ldc, ftz);
        return;
    }
    if constexpr (J <= 2) {
        if (level == 1) {
            mm_panel_avx2<MR, J, AT, Loader, Dec>(fmt_, A, lda, ep, dp, e_k, d_k,
                                                  e_j, d_j, K, C, ldc, ftz);
            return;
        }
    }
#endif
    (void)level;
    mm_panel_scalar<MR, J, AT, Loader, Dec>(fmt_, A, lda, ep, dp, e_k, d_k, e_j,
                                            d_j, K, C, ldc, ftz);
}

template <int MR, class AT, class Loader, class Dec>
inline void mm_run(ttbfp_fmt fmt_, const AT *A, int64_t lda,
                   const uint8_t *e0, const uint8_t *d0,
                   int64_t e_k, int64_t d_k, int64_t e_j, int64_t d_j,
                   int64_t K, int64_t j0, int jc,
                   float *C, int64_t ldc, int ftz, int level) {
    // Larger row tiles use fewer B blocks so the accumulators stay in
    // registers (16 rows x 1 block and 8 rows x 2 blocks both use 16 zmm).
    constexpr int JMAX = (MR >= 16) ? 1 : ((MR >= 8) ? 2 : 4);
    int jj = 0;
    while (jj < jc) {
        if constexpr (JMAX >= 4) {
            if (level >= 2 && jc - jj >= 4) {
                mm_panel<MR, 4, AT, Loader, Dec>(fmt_, A, lda, e0, d0, e_k, d_k,
                                                 e_j, d_j, K, j0 + jj,
                                                 C + jj * TTBFP_BLOCK, ldc, ftz,
                                                 level);
                jj += 4;
                continue;
            }
        }
        if (jc - jj >= 2) {
            mm_panel<MR, 2, AT, Loader, Dec>(fmt_, A, lda, e0, d0, e_k, d_k, e_j,
                                             d_j, K, j0 + jj,
                                             C + jj * TTBFP_BLOCK, ldc, ftz,
                                             level);
            jj += 2;
        } else {
            mm_panel<MR, 1, AT, Loader, Dec>(fmt_, A, lda, e0, d0, e_k, d_k, e_j,
                                             d_j, K, j0 + jj,
                                             C + jj * TTBFP_BLOCK, ldc, ftz,
                                             level);
            jj += 1;
        }
    }
}

/* ---- driver ------------------------------------------------------- */

template <class AT, class Loader, class Dec>
void matmul_driver(ttbfp_fmt fmt_, const AT *A, int64_t lda,
                   const uint8_t *exps, const uint8_t *data,
                   int64_t M, int64_t K, int64_t N,
                   float *C, int64_t ldc, int ftz,
                   int64_t e_k, int64_t d_k, int64_t e_j, int64_t d_j,
                   int mrt_hint, int jm_hint) {
    const ttbfp_fmt fmt = fmt_;
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(N);
    const int64_t nfull = N / TTBFP_BLOCK;  // complete 16-wide blocks
    const int level = have_avx512() ? 2 : (have_avx2_fma() ? 1 : 0);

    int mrt = mrt_hint;
    int jm = jm_hint;
    // Optional tile-shape override for tuning on a particular CPU.
    const char *env_mrt_s = std::getenv("TTBFP_MRT");
    const char *env_jm_s = std::getenv("TTBFP_JM");
    const int env_mrt = (env_mrt_s && *env_mrt_s) ? std::atoi(env_mrt_s) : 0;
    const int env_jm = (env_jm_s && *env_jm_s) ? std::atoi(env_jm_s) : 0;
    if (env_mrt > 0) mrt = std::min(env_mrt, (int)M);
    if (env_jm > 0) jm = env_jm;
    if (mrt > 16) mrt = 16;
    if (mrt * jm > 16) jm = std::max(1, 16 / mrt);
    const int64_t mtiles = (M + mrt - 1) / mrt;
    const int64_t jpanels = (nfull + jm - 1) / jm;

    if (jpanels > 0) {
        // One iteration per (row tile, column panel) output block; nothing is
        // reduced across threads, so any thread count gives the same bits.
        // Run with OMP_WAIT_POLICY=passive when benchmarking back-to-back.
#pragma omp parallel for collapse(2) schedule(static) if (mtiles * jpanels > 1)
        for (int64_t mt = 0; mt < mtiles; ++mt) {
            for (int64_t jp = 0; jp < jpanels; ++jp) {
                const int mr = (int)std::min<int64_t>(mrt, M - mt * mrt);
                const int64_t j0 = jp * jm;
                const int jc = (int)std::min<int64_t>(jm, nfull - j0);
                const AT *Ap = A + mt * mrt * lda;
                float *Cp = C + mt * mrt * ldc + j0 * TTBFP_BLOCK;
                switch (mr) {
                    case 16: mm_run<16, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 8: mm_run<8, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 7: mm_run<7, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 6: mm_run<6, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 5: mm_run<5, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 4: mm_run<4, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 3: mm_run<3, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 2: mm_run<2, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    case 1: mm_run<1, AT, Loader, Dec>(fmt, Ap, lda, exps, data, e_k, d_k, e_j, d_j, K, j0, jc, Cp, ldc, ftz, level); break;
                    default: break;
                }
            }
        }
    }

    if (nblk > nfull) {
        // Final, partially filled 16-wide block.
        const int64_t j = nfull;
        const int64_t n = N - j * TTBFP_BLOCK;
        const uint8_t *ep = exps + j * e_j;
        const uint8_t *dp = data + j * d_j;
#pragma omp parallel for schedule(static) if (mtiles > 1)
        for (int64_t mt = 0; mt < mtiles; ++mt) {
            const int mr = (int)std::min<int64_t>(mrt, M - mt * mrt);
            float tmp[16 * TTBFP_BLOCK];
            const AT *Ap = A + mt * mrt * lda;
            switch (mr) {
                case 16: mm_run<16, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 8: mm_run<8, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 7: mm_run<7, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 6: mm_run<6, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 5: mm_run<5, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 4: mm_run<4, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 3: mm_run<3, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 2: mm_run<2, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                case 1: mm_run<1, AT, Loader, Dec>(fmt, Ap, lda, ep, dp, e_k, d_k, 0, 0, K, 0, 1, tmp, TTBFP_BLOCK, ftz, level); break;
                default: break;
            }
            for (int r = 0; r < mr; ++r) {
                std::memcpy(C + (mt * mrt + r) * ldc + j * TTBFP_BLOCK,
                            tmp + r * TTBFP_BLOCK, (size_t)n * sizeof(float));
            }
        }
    }
}

}  // namespace

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

size_t ttbfp_blocks_per_row(int64_t cols) {
    if (cols <= 0) return 0;
    return (size_t)((cols + TTBFP_BLOCK - 1) / TTBFP_BLOCK);
}

size_t ttbfp_exps_bytes(ttbfp_fmt, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return 0;
    return (size_t)rows * ttbfp_blocks_per_row(cols);
}

size_t ttbfp_data_bytes(ttbfp_fmt fmtm, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return 0;
    return (size_t)rows * ttbfp_blocks_per_row(cols) * (size_t)datum_bytes(fmtm);
}

size_t ttbfp_packed_bytes(ttbfp_fmt fmtm, int64_t rows, int64_t cols) {
    return ttbfp_exps_bytes(fmtm, rows, cols) +
           ttbfp_data_bytes(fmtm, rows, cols);
}

size_t ttbfp_tile_bytes(ttbfp_fmt fmtm) {
    // 64 exponents + 64 blocks of 16 datums at (magnitude_bits + 1) bits.
    return 64 + 128u * (size_t)(magnitude_bits(fmtm) + 1);
}

void ttbfp_quantize(ttbfp_fmt fmt,
                    const float *src, int64_t rows, int64_t cols,
                    uint8_t *exps, uint8_t *data) {
    if (!src || !exps || !data || rows <= 0 || cols <= 0) return;
    quantize_rows(fmt, src, rows, cols, exps, data);
}

void ttbfp_dequantize(ttbfp_fmt fmt,
                      const uint8_t *exps, const uint8_t *data,
                      int64_t rows, int64_t cols,
                      float *dst, int flush_denormals) {
    if (!exps || !data || !dst || rows <= 0 || cols <= 0) return;
#if TTBFP_X86
    if (have_avx512()) {
        const int64_t nblk = (int64_t)ttbfp_blocks_per_row(cols);
        const int db = datum_bytes(fmt);
#pragma omp parallel for schedule(static) if (rows > 1)
        for (int64_t r = 0; r < rows; ++r) {
            dequantize_row_avx512(fmt, exps + r * nblk, data + r * nblk * db,
                                  cols, dst + r * cols, flush_denormals);
        }
        return;
    }
    if (have_avx2_fma()) {
        const int64_t nblk = (int64_t)ttbfp_blocks_per_row(cols);
        const int db = datum_bytes(fmt);
#pragma omp parallel for schedule(static) if (rows > 1)
        for (int64_t r = 0; r < rows; ++r) {
            dequantize_row_avx2(fmt, exps + r * nblk, data + r * nblk * db,
                                cols, dst + r * cols, flush_denormals);
        }
        return;
    }
#endif
    dequantize_rows_scalar(fmt, exps, data, rows, cols, dst, flush_denormals);
}

void ttbfp_matmul(ttbfp_fmt fmt,
                  const float *A, int64_t lda,
                  const uint8_t *exps, const uint8_t *data,
                  int64_t M, int64_t K, int64_t N,
                  float *C, int64_t ldc, int flush_denormals) {
    if (!A || !exps || !data || !C || M <= 0 || K <= 0 || N <= 0) return;
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(N);
    const int db = datum_bytes(fmt);
    // Row-major packed B: k advances by one whole row, j by one block.
    matmul_driver<float, FloatLoader, SmDec>(fmt, A, lda, exps, data, M, K, N, C, ldc,
                                      flush_denormals,
                                      nblk, (int64_t)nblk * db, 1, db,
                                      4, 4);
}

void ttbfp_matmul_bf16(ttbfp_fmt fmt,
                       const uint16_t *A, int64_t lda,
                       const uint8_t *exps,
                       const uint8_t *data,
                       int64_t M, int64_t K, int64_t N,
                       float *C, int64_t ldc, int flush_denormals) {
    if (!A || !exps || !data || !C || M <= 0 || K <= 0 || N <= 0) return;
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(N);
    const int db = datum_bytes(fmt);
    matmul_driver<uint16_t, BF16Loader, SmDec>(fmt, A, lda, exps, data, M, K, N, C,
                                        ldc, flush_denormals,
                                        nblk, (int64_t)nblk * db, 1, db, 4, 4);
}

/* Tile shape for the block-major layout.
 *
 * An MR x J tile spends two resources that trade off against each other,
 * with MR*J capped at 16 by the accumulator registers:
 *
 *   decode work  ~ 1/MR   one B block is decoded per MR FMAs
 *   A traffic    ~ 1/J    A is re-staged once per column panel
 *
 * Which one binds depends on the decoder, so the shape is chosen per
 * decoder (measured at K=N=4096 with bench/ab_compare.py; the numbers are
 * single-thread, so a multi-thread run just scales them down):
 *
 *   sign-magnitude  16 rows x 1 block from M=16 up (decode dominates),
 *                   8 rows x 1 block at M=8, 4 x 4 below that.
 *   int8             4 rows x 4 blocks at every M: the 3-uop decode is
 *                   cheap enough that cutting A traffic 4x wins, by
 *                   16% at M>=16 and 34% at M=1.
 *
 * The shape only decides which B blocks share an A broadcast.  The k order
 * of every accumulation is untouched, so all shapes are bitwise equal. */
static void prepacked_tile(int64_t M, bool s8, int *mrt, int *jm) {
    const int lvl = have_avx512() ? 2 : (have_avx2_fma() ? 1 : 0);
    if (lvl < 2) {  // AVX2 / scalar: keep the long-standing shape
        *mrt = 4;
        *jm = 4;
        return;
    }
    if (s8) {
        // int8 decode is 3 uops, so the binding cost is A traffic, which
        // falls as 1/J.  MR4 x J4 won at every M (16-34% over MR16/J1).
        *mrt = M >= 4 ? 4 : (int)M;
        *jm = 4;
        return;
    }
    // Sign-magnitude decode is ~7 uops, so keep the row tile as tall as
    // possible (decode work ~ 1/MR) and let J=1 carry the A traffic.
    if (M >= 16) { *mrt = 16; *jm = 1; return; }
    if (M >= 8)  { *mrt = 8;  *jm = 1; return; }
    *mrt = 4;
    *jm = 4;
}

void ttbfp_transpose_packed(ttbfp_fmt fmt,
                            const uint8_t *exps, const uint8_t *data,
                            int64_t K, int64_t N,
                            uint8_t *exps_t, uint8_t *data_t) {
    if (!exps || !data || !exps_t || !data_t || K <= 0 || N <= 0) return;
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(N);
    const int db = datum_bytes(fmt);
#pragma omp parallel for schedule(static) if (nblk > 1)
    for (int64_t j = 0; j < nblk; ++j) {
        uint8_t *et = exps_t + j * K;
        uint8_t *dt = data_t + j * K * db;
        for (int64_t k = 0; k < K; ++k) {
            et[k] = exps[k * nblk + j];
            std::memcpy(dt + k * db, data + (k * nblk + j) * db, (size_t)db);
        }
    }
}

void ttbfp_matmul_prepacked(ttbfp_fmt fmt,
                            const float *A, int64_t lda,
                            const uint8_t *exps_t, const uint8_t *data_t,
                            int64_t M, int64_t K, int64_t N,
                            float *C, int64_t ldc, int flush_denormals) {
    if (!A || !exps_t || !data_t || !C || M <= 0 || K <= 0 || N <= 0) return;
    const int db = datum_bytes(fmt);
    int mrt = 4, jm = 4;
    prepacked_tile(M, false, &mrt, &jm);
    // Block-major: k advances by one block, j by a whole K-long stream.
    matmul_driver<float, FloatLoader, SmDec>(fmt, A, lda, exps_t, data_t, M, K, N, C,
                                      ldc, flush_denormals,
                                      1, db, K, (int64_t)K * db, mrt, jm);
}

void ttbfp_matmul_prepacked_bf16(ttbfp_fmt fmt,
                                 const uint16_t *A, int64_t lda,
                                 const uint8_t *exps_t, const uint8_t *data_t,
                                 int64_t M, int64_t K, int64_t N,
                                 float *C, int64_t ldc, int flush_denormals) {
    if (!A || !exps_t || !data_t || !C || M <= 0 || K <= 0 || N <= 0) return;
    const int db = datum_bytes(fmt);
    int mrt = 4, jm = 4;
    prepacked_tile(M, false, &mrt, &jm);
    matmul_driver<uint16_t, BF16Loader, SmDec>(fmt, A, lda, exps_t, data_t, M, K, N, C,
                                        ldc, flush_denormals,
                                        1, db, K, (int64_t)K * db, mrt, jm);
}

void ttbfp_convert_s8(ttbfp_fmt fmt,
                      const uint8_t *exps, const uint8_t *data,
                      int64_t K, int64_t N,
                      uint8_t *exps_t, uint8_t *s8_t) {
    if (!exps || !data || !exps_t || !s8_t || K <= 0 || N <= 0) return;
    const int64_t nblk = (int64_t)ttbfp_blocks_per_row(N);
    const int db = datum_bytes(fmt);
    const int mb = magnitude_bits(fmt);
    const int mask = (1 << mb) - 1;
    const int sbit = 1 << mb;
#pragma omp parallel for schedule(static) if (nblk > 1)
    for (int64_t j = 0; j < nblk; ++j) {
        uint8_t *et = exps_t + j * K;
        uint8_t *st = s8_t + j * K * TTBFP_BLOCK;
        for (int64_t k = 0; k < K; ++k) {
            et[k] = exps[k * nblk + j];
            const uint8_t *src = data + (k * nblk + j) * db;
            uint8_t *dst = st + k * TTBFP_BLOCK;
            for (int i = 0; i < TTBFP_BLOCK; ++i) {
                const int v = fmt == TTBFP_TTBF8
                    ? src[i]
                    : fmt == TTBFP_TTBF4
                        ? ((i & 1) ? (src[i >> 1] >> 4)
                                   : (src[i >> 1] & 0x0F))
                        : ((src[i >> 2] >> ((i & 3) * 2)) & 0x03);
                const int mag = v & mask;
                dst[i] = (uint8_t)(int8_t)((v & sbit) ? -mag : mag);
            }
        }
    }
}

void ttbfp_matmul_prepacked_s8(ttbfp_fmt fmt,
                               const float *A, int64_t lda,
                               const uint8_t *exps_t, const uint8_t *s8_t,
                               int64_t M, int64_t K, int64_t N,
                               float *C, int64_t ldc, int flush_denormals) {
    if (!A || !exps_t || !s8_t || !C || M <= 0 || K <= 0 || N <= 0) return;
    int mrt = 4, jm = 4;
    prepacked_tile(M, true, &mrt, &jm);
    matmul_driver<float, FloatLoader, S8Dec>(
        fmt, A, lda, exps_t, s8_t, M, K, N, C, ldc, flush_denormals,
        1, TTBFP_BLOCK, K, (int64_t)K * TTBFP_BLOCK, mrt, jm);
}

void ttbfp_matmul_prepacked_s8_bf16(ttbfp_fmt fmt,
                                    const uint16_t *A, int64_t lda,
                                    const uint8_t *exps_t,
                                    const uint8_t *s8_t,
                                    int64_t M, int64_t K, int64_t N,
                                    float *C, int64_t ldc, int flush_denormals) {
    if (!A || !exps_t || !s8_t || !C || M <= 0 || K <= 0 || N <= 0) return;
    int mrt = 4, jm = 4;
    prepacked_tile(M, true, &mrt, &jm);
    matmul_driver<uint16_t, BF16Loader, S8Dec>(
        fmt, A, lda, exps_t, s8_t, M, K, N, C, ldc, flush_denormals,
        1, TTBFP_BLOCK, K, (int64_t)K * TTBFP_BLOCK, mrt, jm);
}

void ttbfp_tile_pack(ttbfp_fmt fmtm, const float *tile, uint8_t *out) {
    if (!tile || !out) return;
    float tmp[1024];
    for (int f = 0; f < 4; ++f) {
        const int fr = f >> 1;
        const int fc = f & 1;
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                tmp[f * 256 + r * 16 + c] =
                    tile[(fr * 16 + r) * 32 + (fc * 16 + c)];
            }
        }
    }
    ttbfp_quantize(fmtm, tmp, 64, 16, out, out + 64);
}

void ttbfp_tile_unpack(ttbfp_fmt fmtm,
                       const uint8_t *in, float *tile,
                       int flush_denormals) {
    if (!in || !tile) return;
    float tmp[1024];
    ttbfp_dequantize(fmtm, in, in + 64, 64, 16, tmp, flush_denormals);
    for (int f = 0; f < 4; ++f) {
        const int fr = f >> 1;
        const int fc = f & 1;
        for (int r = 0 ; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                tile[(fr * 16 + r) * 32 + (fc * 16 + c)] =
                    tmp[f * 256 + r * 16 + c];
            }
        }
    }
}

const char *ttbfp_version(void) {
    return "0.1.0";
}

const char *ttbfp_cpu_features(void) {
    if (have_avx512()) return "avx512";
    if (have_avx2_fma()) return "avx2+fma";
    return "scalar";
}
