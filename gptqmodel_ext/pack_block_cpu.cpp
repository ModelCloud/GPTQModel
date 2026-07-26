// SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
// SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0
// Contact: qubitium@modelcloud.ai, x.com/qubitium

#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define PACK_BLOCK_CPU_X86 1
#else
#define PACK_BLOCK_CPU_X86 0
#endif

namespace gptqmodel {

namespace {

inline bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    switch (value[0]) {
        case '1':
        case 'Y':
        case 'y':
        case 'T':
        case 't':
            return true;
        default:
            return false;
    }
}

inline int64_t clamped_threads(int64_t requested) {
    const int64_t hard_limit = 32;
    const int64_t available = at::get_num_threads();
    if (requested > 0) {
        return std::max<int64_t>(1, std::min<int64_t>(requested, std::min<int64_t>(available, hard_limit)));
    }
    return std::max<int64_t>(1, std::min<int64_t>(available, hard_limit));
}

#if PACK_BLOCK_CPU_X86
// __builtin_cpu_supports is only safe to use after the process has initialized
// the CPU feature bits. In JIT-loaded shared libraries this is not guaranteed,
// so force initialization at library load time.
__attribute__((constructor))
static void pack_block_cpu_init_cpu_features() {
    __builtin_cpu_init();
}

inline bool cpu_supports_avx2() {
    if (env_flag_enabled("GPTQMODEL_PACK_CPU_DISABLE_AVX2")) {
        return false;
    }
    return __builtin_cpu_supports("avx2");
}

inline bool cpu_supports_avx512() {
    if (env_flag_enabled("GPTQMODEL_PACK_CPU_DISABLE_AVX512")) {
        return false;
    }
    return __builtin_cpu_supports("avx512f");
}
#endif

// Normalize a 32-element g_idx block once, handling negative offsets and range checks.
void normalize_gidx_block(
    const int32_t* gidx_ptr,
    int64_t base_input,
    int64_t groups,
    int32_t* gidx_block) {
    for (int lane = 0; lane < 32; ++lane) {
        const int64_t input_idx = base_input + lane;
        const int32_t raw_group = gidx_ptr[input_idx];
        int32_t group = raw_group;
        if (group < 0) {
            group += static_cast<int32_t>(groups);
        }
        TORCH_CHECK(
            group >= 0 && group < groups,
            "pack_block_cpu: g_idx[",
            input_idx,
            "]=",
            raw_group,
            " is out of range for groups=",
            groups);
        gidx_block[lane] = group;
    }
}

void pack_qweight_scalar(
    const float* weight_ptr,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const int32_t* gidx_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t block_begin,
    int64_t block_end,
    int64_t groups,
    int bits,
    int max_q) {
    const int rows_per_group = (bits == 3) ? 3 : bits;
    const int pack_factor = (bits == 3) ? 0 : (32 / bits);
    alignas(64) int32_t qvals[32];
    alignas(64) int32_t gidx_block[32];

    for (int64_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
        const int64_t base_input = block_idx * 32;
        const int row_base = static_cast<int>(block_idx * rows_per_group);
        normalize_gidx_block(gidx_ptr, base_input, groups, gidx_block);

        for (int64_t out = 0; out < out_features; ++out) {
            for (int lane = 0; lane < 32; ++lane) {
                const int32_t group = gidx_block[lane];
                const int64_t input_idx = base_input + lane;
                float scale = scales_ptr[static_cast<int64_t>(group) * scales_stride + out];
                float offset = scale_zeros_ptr[static_cast<int64_t>(group) * scales_stride + out];
                float w = weight_ptr[static_cast<int64_t>(out) * out_stride + input_idx];
                if (scale == 0.0f) {
                    scale = 1e-6f;
                }
                float qf = std::nearbyint((w + offset) / scale);
                qf = std::max<float>(0.0f, std::min<float>(qf, static_cast<float>(max_q)));
                qvals[lane] = static_cast<int32_t>(qf);
            }

            if (bits == 3) {
                int64_t A = 0;
                for (int j = 0; j < 10; ++j) {
                    A |= static_cast<int64_t>(qvals[j]) << (3 * j);
                }
                A |= static_cast<int64_t>(qvals[10]) << 30;

                int64_t B = static_cast<int64_t>((qvals[10] >> 2) & 0x1);
                for (int j = 0; j < 10; ++j) {
                    B |= static_cast<int64_t>(qvals[11 + j]) << (3 * j + 1);
                }
                B |= static_cast<int64_t>(qvals[21]) << 31;

                int64_t C = static_cast<int64_t>((qvals[21] >> 1) & 0x3);
                for (int j = 0; j < 10; ++j) {
                    C |= static_cast<int64_t>(qvals[22 + j]) << (3 * j + 2);
                }

                qweight_ptr[(row_base + 0) * out_features + out] = static_cast<int32_t>(A & 0xFFFFFFFF);
                qweight_ptr[(row_base + 1) * out_features + out] = static_cast<int32_t>(B & 0xFFFFFFFF);
                qweight_ptr[(row_base + 2) * out_features + out] = static_cast<int32_t>(C & 0xFFFFFFFF);
            } else {
                for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
                    int64_t packed = 0;
                    for (int pf = 0; pf < pack_factor; ++pf) {
                        int idx = bit_plane * pack_factor + pf;
                        packed |= static_cast<int64_t>(qvals[idx]) << (bits * pf);
                    }
                    qweight_ptr[(row_base + bit_plane) * out_features + out] = static_cast<int32_t>(packed & 0xFFFFFFFF);
                }
            }
        }
    }
}

#if PACK_BLOCK_CPU_X86

// 16x16 float transpose kernel adapted from FBGEMM (BSD-licensed).
// Transposes a 16x16 matrix with source/destination strides ld_src/ld_dst.
__attribute__((target("avx512f")))
inline void transpose_16x16_avx512(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
    __m512 a = _mm512_loadu_ps(&src[0 * ld_src]);
    __m512 b = _mm512_loadu_ps(&src[1 * ld_src]);
    __m512 c = _mm512_loadu_ps(&src[2 * ld_src]);
    __m512 d = _mm512_loadu_ps(&src[3 * ld_src]);
    __m512 e = _mm512_loadu_ps(&src[4 * ld_src]);
    __m512 f = _mm512_loadu_ps(&src[5 * ld_src]);
    __m512 g = _mm512_loadu_ps(&src[6 * ld_src]);
    __m512 h = _mm512_loadu_ps(&src[7 * ld_src]);
    __m512 i = _mm512_loadu_ps(&src[8 * ld_src]);
    __m512 j = _mm512_loadu_ps(&src[9 * ld_src]);
    __m512 k = _mm512_loadu_ps(&src[10 * ld_src]);
    __m512 l = _mm512_loadu_ps(&src[11 * ld_src]);
    __m512 m = _mm512_loadu_ps(&src[12 * ld_src]);
    __m512 n = _mm512_loadu_ps(&src[13 * ld_src]);
    __m512 o = _mm512_loadu_ps(&src[14 * ld_src]);
    __m512 p = _mm512_loadu_ps(&src[15 * ld_src]);

    __m512 ta, tb, tc, td, te, tf, tg, th, ti, tj, tk, tl, tm, tn, to, tq;
    ta = _mm512_unpacklo_ps(a, b);
    tb = _mm512_unpackhi_ps(a, b);
    tc = _mm512_unpacklo_ps(c, d);
    td = _mm512_unpackhi_ps(c, d);
    te = _mm512_unpacklo_ps(e, f);
    tf = _mm512_unpackhi_ps(e, f);
    tg = _mm512_unpacklo_ps(g, h);
    th = _mm512_unpackhi_ps(g, h);
    ti = _mm512_unpacklo_ps(i, j);
    tj = _mm512_unpackhi_ps(i, j);
    tk = _mm512_unpacklo_ps(k, l);
    tl = _mm512_unpackhi_ps(k, l);
    tm = _mm512_unpacklo_ps(m, n);
    tn = _mm512_unpackhi_ps(m, n);
    to = _mm512_unpacklo_ps(o, p);
    tq = _mm512_unpackhi_ps(o, p);

    a = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(ta), _mm512_castps_pd(tc)));
    b = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(ta), _mm512_castps_pd(tc)));
    c = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(tb), _mm512_castps_pd(td)));
    d = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(tb), _mm512_castps_pd(td)));
    e = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(te), _mm512_castps_pd(tg)));
    f = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(te), _mm512_castps_pd(tg)));
    g = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(tf), _mm512_castps_pd(th)));
    h = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(tf), _mm512_castps_pd(th)));
    i = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(ti), _mm512_castps_pd(tk)));
    j = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(ti), _mm512_castps_pd(tk)));
    k = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(tj), _mm512_castps_pd(tl)));
    l = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(tj), _mm512_castps_pd(tl)));
    m = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(tm), _mm512_castps_pd(to)));
    n = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(tm), _mm512_castps_pd(to)));
    o = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(tn), _mm512_castps_pd(tq)));
    p = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(tn), _mm512_castps_pd(tq)));

    ta = _mm512_shuffle_f32x4(a, e, 0x88);
    tb = _mm512_shuffle_f32x4(b, f, 0x88);
    tc = _mm512_shuffle_f32x4(c, g, 0x88);
    td = _mm512_shuffle_f32x4(d, h, 0x88);
    te = _mm512_shuffle_f32x4(a, e, 0xdd);
    tf = _mm512_shuffle_f32x4(b, f, 0xdd);
    tg = _mm512_shuffle_f32x4(c, g, 0xdd);
    th = _mm512_shuffle_f32x4(d, h, 0xdd);
    ti = _mm512_shuffle_f32x4(i, m, 0x88);
    tj = _mm512_shuffle_f32x4(j, n, 0x88);
    tk = _mm512_shuffle_f32x4(k, o, 0x88);
    tl = _mm512_shuffle_f32x4(l, p, 0x88);
    tm = _mm512_shuffle_f32x4(i, m, 0xdd);
    tn = _mm512_shuffle_f32x4(j, n, 0xdd);
    to = _mm512_shuffle_f32x4(k, o, 0xdd);
    tq = _mm512_shuffle_f32x4(l, p, 0xdd);

    a = _mm512_shuffle_f32x4(ta, ti, 0x88);
    b = _mm512_shuffle_f32x4(tb, tj, 0x88);
    c = _mm512_shuffle_f32x4(tc, tk, 0x88);
    d = _mm512_shuffle_f32x4(td, tl, 0x88);
    e = _mm512_shuffle_f32x4(te, tm, 0x88);
    f = _mm512_shuffle_f32x4(tf, tn, 0x88);
    g = _mm512_shuffle_f32x4(tg, to, 0x88);
    h = _mm512_shuffle_f32x4(th, tq, 0x88);
    i = _mm512_shuffle_f32x4(ta, ti, 0xdd);
    j = _mm512_shuffle_f32x4(tb, tj, 0xdd);
    k = _mm512_shuffle_f32x4(tc, tk, 0xdd);
    l = _mm512_shuffle_f32x4(td, tl, 0xdd);
    m = _mm512_shuffle_f32x4(te, tm, 0xdd);
    n = _mm512_shuffle_f32x4(tf, tn, 0xdd);
    o = _mm512_shuffle_f32x4(tg, to, 0xdd);
    p = _mm512_shuffle_f32x4(th, tq, 0xdd);

    _mm512_storeu_ps(&dst[0 * ld_dst], a);
    _mm512_storeu_ps(&dst[1 * ld_dst], b);
    _mm512_storeu_ps(&dst[2 * ld_dst], c);
    _mm512_storeu_ps(&dst[3 * ld_dst], d);
    _mm512_storeu_ps(&dst[4 * ld_dst], e);
    _mm512_storeu_ps(&dst[5 * ld_dst], f);
    _mm512_storeu_ps(&dst[6 * ld_dst], g);
    _mm512_storeu_ps(&dst[7 * ld_dst], h);
    _mm512_storeu_ps(&dst[8 * ld_dst], i);
    _mm512_storeu_ps(&dst[9 * ld_dst], j);
    _mm512_storeu_ps(&dst[10 * ld_dst], k);
    _mm512_storeu_ps(&dst[11 * ld_dst], l);
    _mm512_storeu_ps(&dst[12 * ld_dst], m);
    _mm512_storeu_ps(&dst[13 * ld_dst], n);
    _mm512_storeu_ps(&dst[14 * ld_dst], o);
    _mm512_storeu_ps(&dst[15 * ld_dst], p);
}

// AVX-512 path for bits 2, 4, 8.  Processes 16 output channels at a time.
template <int bits, int pack_factor>
__attribute__((target("avx512f")))
void pack_qweight_avx512(
    const float* weight_ptr,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const int32_t* gidx_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t block_begin,
    int64_t block_end,
    int64_t groups,
    int max_q) {
    const __m512 zero_ps = _mm512_setzero_ps();
    const __m512 eps_ps = _mm512_set1_ps(1e-6f);
    const __m512 maxq_ps = _mm512_set1_ps(static_cast<float>(max_q));
    const __m512i zero_epi = _mm512_setzero_si512();

    alignas(64) int32_t out_offsets_data[16];
    for (int i = 0; i < 16; ++i) {
        out_offsets_data[i] = i * static_cast<int32_t>(out_stride);
    }
    const __m512i out_offsets = _mm512_load_si512(out_offsets_data);

    alignas(64) int32_t lane_qvals[32][16];
    alignas(64) float t0[16][16];
    alignas(64) float t1[16][16];

    for (int64_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
        const int64_t base_input = block_idx * 32;
        const int row_base = static_cast<int>(block_idx * bits);
        alignas(64) int32_t gidx_block[32];
        normalize_gidx_block(gidx_ptr, base_input, groups, gidx_block);

        bool uniform_group = true;
        for (int i = 1; i < 32; ++i) {
            if (gidx_block[i] != gidx_block[0]) {
                uniform_group = false;
                break;
            }
        }

        if (uniform_group) {
            // Uniform group: transpose contiguous 16x16 weight slices so we can
            // load each output row with efficient vmovups instead of strided gathers.
            const int32_t group = gidx_block[0];
            const int64_t scale_offset_base = static_cast<int64_t>(group) * scales_stride;
            for (int64_t out = 0; out < out_features; out += 16) {
                const float* src0 = weight_ptr + out * out_stride + base_input;
                transpose_16x16_avx512(src0, out_stride, &t0[0][0], 16);
                transpose_16x16_avx512(src0 + 16, out_stride, &t1[0][0], 16);

                __m512 scale = _mm512_loadu_ps(scales_ptr + scale_offset_base + out);
                const __m512 offset = _mm512_loadu_ps(scale_zeros_ptr + scale_offset_base + out);
                const __mmask16 zero_mask = _mm512_cmp_ps_mask(scale, zero_ps, _CMP_EQ_OQ);
                scale = _mm512_mask_blend_ps(zero_mask, scale, eps_ps);

                for (int lane = 0; lane < 16; ++lane) {
                    const __m512 w = _mm512_loadu_ps(&t0[lane][0]);
                    __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                    qf = _mm512_max_ps(qf, zero_ps);
                    qf = _mm512_min_ps(qf, maxq_ps);
                    const __m512i q = _mm512_cvt_roundps_epi32(qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(lane_qvals[lane]), q);
                }
                for (int lane = 16; lane < 32; ++lane) {
                    const __m512 w = _mm512_loadu_ps(&t1[lane - 16][0]);
                    __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                    qf = _mm512_max_ps(qf, zero_ps);
                    qf = _mm512_min_ps(qf, maxq_ps);
                    const __m512i q = _mm512_cvt_roundps_epi32(qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(lane_qvals[lane]), q);
                }

                for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
                    __m512i acc = zero_epi;
                    for (int pf = 0; pf < pack_factor; ++pf) {
                        const int idx = bit_plane * pack_factor + pf;
                        const int shift = bits * pf;
                        const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[idx]));
                        acc = _mm512_or_si512(acc, _mm512_sllv_epi32(qv, _mm512_set1_epi32(shift)));
                    }
                    _mm512_storeu_si512(
                        reinterpret_cast<__m512i*>(qweight_ptr + (row_base + bit_plane) * out_features + out),
                        acc);
                }
            }
        } else {
            for (int64_t out = 0; out < out_features; out += 16) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int32_t group = gidx_block[lane];
                    const float* wbase = weight_ptr + static_cast<int64_t>(out) * out_stride + base_input + lane;
                    const __m512 w = _mm512_i32gather_ps(out_offsets, wbase, 4);

                    const float* sbase = scales_ptr + static_cast<int64_t>(group) * scales_stride + out;
                    __m512 scale = _mm512_loadu_ps(sbase);
                    const __m512 offset = _mm512_loadu_ps(scale_zeros_ptr + static_cast<int64_t>(group) * scales_stride + out);

                    const __mmask16 zero_mask = _mm512_cmp_ps_mask(scale, zero_ps, _CMP_EQ_OQ);
                    scale = _mm512_mask_blend_ps(zero_mask, scale, eps_ps);

                    __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                    qf = _mm512_max_ps(qf, zero_ps);
                    qf = _mm512_min_ps(qf, maxq_ps);

                    const __m512i q = _mm512_cvt_roundps_epi32(qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(lane_qvals[lane]), q);
                }

                for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
                    __m512i acc = zero_epi;
                    for (int pf = 0; pf < pack_factor; ++pf) {
                        const int idx = bit_plane * pack_factor + pf;
                        const int shift = bits * pf;
                        const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[idx]));
                        acc = _mm512_or_si512(acc, _mm512_sllv_epi32(qv, _mm512_set1_epi32(shift)));
                    }
                    _mm512_storeu_si512(
                        reinterpret_cast<__m512i*>(qweight_ptr + (row_base + bit_plane) * out_features + out),
                        acc);
                }
            }
        }
    }
}

// AVX-512 3-bit packing helper.  Each 32-element lane block produces 3 packed 32-bit words.
__attribute__((target("avx512f")))
void pack_qweight_3bit_avx512(
    const float* weight_ptr,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const int32_t* gidx_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t block_begin,
    int64_t block_end,
    int64_t groups,
    int max_q) {
    const __m512 zero_ps = _mm512_setzero_ps();
    const __m512 eps_ps = _mm512_set1_ps(1e-6f);
    const __m512 maxq_ps = _mm512_set1_ps(static_cast<float>(max_q));

    alignas(64) int32_t out_offsets_data[16];
    for (int i = 0; i < 16; ++i) {
        out_offsets_data[i] = i * static_cast<int32_t>(out_stride);
    }
    const __m512i out_offsets = _mm512_load_si512(out_offsets_data);

    alignas(64) int32_t lane_qvals[32][16];
    alignas(64) float t0[16][16];
    alignas(64) float t1[16][16];

    for (int64_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
        const int64_t base_input = block_idx * 32;
        const int row_base = static_cast<int>(block_idx * 3);
        alignas(64) int32_t gidx_block[32];
        normalize_gidx_block(gidx_ptr, base_input, groups, gidx_block);

        bool uniform_group = true;
        for (int i = 1; i < 32; ++i) {
            if (gidx_block[i] != gidx_block[0]) {
                uniform_group = false;
                break;
            }
        }

        if (uniform_group) {
            const int32_t group = gidx_block[0];
            const int64_t scale_offset_base = static_cast<int64_t>(group) * scales_stride;
            for (int64_t out = 0; out < out_features; out += 16) {
                const float* src0 = weight_ptr + out * out_stride + base_input;
                transpose_16x16_avx512(src0, out_stride, &t0[0][0], 16);
                transpose_16x16_avx512(src0 + 16, out_stride, &t1[0][0], 16);

                __m512 scale = _mm512_loadu_ps(scales_ptr + scale_offset_base + out);
                const __m512 offset = _mm512_loadu_ps(scale_zeros_ptr + scale_offset_base + out);
                const __mmask16 zero_mask = _mm512_cmp_ps_mask(scale, zero_ps, _CMP_EQ_OQ);
                scale = _mm512_mask_blend_ps(zero_mask, scale, eps_ps);

                for (int lane = 0; lane < 16; ++lane) {
                    const __m512 w = _mm512_loadu_ps(&t0[lane][0]);
                    __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                    qf = _mm512_max_ps(qf, zero_ps);
                    qf = _mm512_min_ps(qf, maxq_ps);
                    const __m512i q = _mm512_cvt_roundps_epi32(qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(lane_qvals[lane]), q);
                }
                for (int lane = 16; lane < 32; ++lane) {
                    const __m512 w = _mm512_loadu_ps(&t1[lane - 16][0]);
                    __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                    qf = _mm512_max_ps(qf, zero_ps);
                    qf = _mm512_min_ps(qf, maxq_ps);
                    const __m512i q = _mm512_cvt_roundps_epi32(qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(lane_qvals[lane]), q);
                }

                __m512i A = _mm512_setzero_si512();
                for (int j = 0; j < 10; ++j) {
                    const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[j]));
                    A = _mm512_or_si512(A, _mm512_sllv_epi32(qv, _mm512_set1_epi32(3 * j)));
                }
                A = _mm512_or_si512(
                    A,
                    _mm512_sllv_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[10])),
                        _mm512_set1_epi32(30)));

                __m512i B = _mm512_and_si512(
                    _mm512_srli_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[10])),
                        2),
                    _mm512_set1_epi32(0x1));
                for (int j = 0; j < 10; ++j) {
                    const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[11 + j]));
                    B = _mm512_or_si512(B, _mm512_sllv_epi32(qv, _mm512_set1_epi32(3 * j + 1)));
                }
                B = _mm512_or_si512(
                    B,
                    _mm512_sllv_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[21])),
                        _mm512_set1_epi32(31)));

                __m512i C = _mm512_and_si512(
                    _mm512_srli_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[21])),
                        1),
                    _mm512_set1_epi32(0x3));
                for (int j = 0; j < 10; ++j) {
                    const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[22 + j]));
                    C = _mm512_or_si512(C, _mm512_sllv_epi32(qv, _mm512_set1_epi32(3 * j + 2)));
                }

                _mm512_storeu_si512(reinterpret_cast<__m512i*>(qweight_ptr + (row_base + 0) * out_features + out), A);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(qweight_ptr + (row_base + 1) * out_features + out), B);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(qweight_ptr + (row_base + 2) * out_features + out), C);
            }
        } else {
            for (int64_t out = 0; out < out_features; out += 16) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int32_t group = gidx_block[lane];
                    const float* wbase = weight_ptr + static_cast<int64_t>(out) * out_stride + base_input + lane;
                    const __m512 w = _mm512_i32gather_ps(out_offsets, wbase, 4);

                    const float* sbase = scales_ptr + static_cast<int64_t>(group) * scales_stride + out;
                    __m512 scale = _mm512_loadu_ps(sbase);
                    const __m512 offset = _mm512_loadu_ps(scale_zeros_ptr + static_cast<int64_t>(group) * scales_stride + out);

                    const __mmask16 zero_mask = _mm512_cmp_ps_mask(scale, zero_ps, _CMP_EQ_OQ);
                    scale = _mm512_mask_blend_ps(zero_mask, scale, eps_ps);

                    __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                    qf = _mm512_max_ps(qf, zero_ps);
                    qf = _mm512_min_ps(qf, maxq_ps);

                    const __m512i q = _mm512_cvt_roundps_epi32(qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(lane_qvals[lane]), q);
                }

                __m512i A = _mm512_setzero_si512();
                for (int j = 0; j < 10; ++j) {
                    const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[j]));
                    A = _mm512_or_si512(A, _mm512_sllv_epi32(qv, _mm512_set1_epi32(3 * j)));
                }
                A = _mm512_or_si512(
                    A,
                    _mm512_sllv_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[10])),
                        _mm512_set1_epi32(30)));

                __m512i B = _mm512_and_si512(
                    _mm512_srli_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[10])),
                        2),
                    _mm512_set1_epi32(0x1));
                for (int j = 0; j < 10; ++j) {
                    const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[11 + j]));
                    B = _mm512_or_si512(B, _mm512_sllv_epi32(qv, _mm512_set1_epi32(3 * j + 1)));
                }
                B = _mm512_or_si512(
                    B,
                    _mm512_sllv_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[21])),
                        _mm512_set1_epi32(31)));

                __m512i C = _mm512_and_si512(
                    _mm512_srli_epi32(
                        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[21])),
                        1),
                    _mm512_set1_epi32(0x3));
                for (int j = 0; j < 10; ++j) {
                    const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lane_qvals[22 + j]));
                    C = _mm512_or_si512(C, _mm512_sllv_epi32(qv, _mm512_set1_epi32(3 * j + 2)));
                }

                _mm512_storeu_si512(reinterpret_cast<__m512i*>(qweight_ptr + (row_base + 0) * out_features + out), A);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(qweight_ptr + (row_base + 1) * out_features + out), B);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(qweight_ptr + (row_base + 2) * out_features + out), C);
            }
        }
    }
}

// AVX2 path for bits 2, 4, 8.  Processes 8 output channels at a time.
template <int bits, int pack_factor>
__attribute__((target("avx2")))
void pack_qweight_avx2(
    const float* weight_ptr,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const int32_t* gidx_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t block_begin,
    int64_t block_end,
    int64_t groups,
    int max_q) {
    const __m256 zero_ps = _mm256_setzero_ps();
    const __m256 eps_ps = _mm256_set1_ps(1e-6f);
    const __m256 maxq_ps = _mm256_set1_ps(static_cast<float>(max_q));
    const __m256i zero_epi = _mm256_setzero_si256();

    alignas(32) int32_t out_offsets_data[8];
    for (int i = 0; i < 8; ++i) {
        out_offsets_data[i] = i * static_cast<int32_t>(out_stride);
    }
    const __m256i out_offsets = _mm256_load_si256(reinterpret_cast<const __m256i*>(out_offsets_data));

    alignas(32) int32_t lane_qvals[32][8];

    for (int64_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
        const int64_t base_input = block_idx * 32;
        const int row_base = static_cast<int>(block_idx * bits);
        alignas(64) int32_t gidx_block[32];
        normalize_gidx_block(gidx_ptr, base_input, groups, gidx_block);

        for (int64_t out = 0; out < out_features; out += 8) {
            for (int lane = 0; lane < 32; ++lane) {
                const int32_t group = gidx_block[lane];
                const float* wbase = weight_ptr + static_cast<int64_t>(out) * out_stride + base_input + lane;
                const __m256 w = _mm256_i32gather_ps(wbase, out_offsets, 4);

                const float* sbase = scales_ptr + static_cast<int64_t>(group) * scales_stride + out;
                __m256 scale = _mm256_loadu_ps(sbase);
                const __m256 offset = _mm256_loadu_ps(scale_zeros_ptr + static_cast<int64_t>(group) * scales_stride + out);

                const __m256 zero_mask = _mm256_cmp_ps(scale, zero_ps, _CMP_EQ_OQ);
                scale = _mm256_blendv_ps(scale, eps_ps, zero_mask);

                __m256 qf = _mm256_div_ps(_mm256_add_ps(w, offset), scale);
                qf = _mm256_max_ps(qf, zero_ps);
                qf = _mm256_min_ps(qf, maxq_ps);

                const __m256i q = _mm256_cvtps_epi32(qf);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(lane_qvals[lane]), q);
            }

            for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
                __m256i acc = zero_epi;
                for (int pf = 0; pf < pack_factor; ++pf) {
                    const int idx = bit_plane * pack_factor + pf;
                    const int shift = bits * pf;
                    const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[idx]));
                    acc = _mm256_or_si256(acc, _mm256_sllv_epi32(qv, _mm256_set1_epi32(shift)));
                }
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(qweight_ptr + (row_base + bit_plane) * out_features + out),
                    acc);
            }
        }
    }
}

// AVX2 3-bit packing helper.
__attribute__((target("avx2")))
void pack_qweight_3bit_avx2(
    const float* weight_ptr,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const int32_t* gidx_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t block_begin,
    int64_t block_end,
    int64_t groups,
    int max_q) {
    const __m256 zero_ps = _mm256_setzero_ps();
    const __m256 eps_ps = _mm256_set1_ps(1e-6f);
    const __m256 maxq_ps = _mm256_set1_ps(static_cast<float>(max_q));

    alignas(32) int32_t out_offsets_data[8];
    for (int i = 0; i < 8; ++i) {
        out_offsets_data[i] = i * static_cast<int32_t>(out_stride);
    }
    const __m256i out_offsets = _mm256_load_si256(reinterpret_cast<const __m256i*>(out_offsets_data));

    alignas(32) int32_t lane_qvals[32][8];

    for (int64_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
        const int64_t base_input = block_idx * 32;
        const int row_base = static_cast<int>(block_idx * 3);
        alignas(64) int32_t gidx_block[32];
        normalize_gidx_block(gidx_ptr, base_input, groups, gidx_block);

        for (int64_t out = 0; out < out_features; out += 8) {
            for (int lane = 0; lane < 32; ++lane) {
                const int32_t group = gidx_block[lane];
                const float* wbase = weight_ptr + static_cast<int64_t>(out) * out_stride + base_input + lane;
                const __m256 w = _mm256_i32gather_ps(wbase, out_offsets, 4);

                const float* sbase = scales_ptr + static_cast<int64_t>(group) * scales_stride + out;
                __m256 scale = _mm256_loadu_ps(sbase);
                const __m256 offset = _mm256_loadu_ps(scale_zeros_ptr + static_cast<int64_t>(group) * scales_stride + out);

                const __m256 zero_mask = _mm256_cmp_ps(scale, zero_ps, _CMP_EQ_OQ);
                scale = _mm256_blendv_ps(scale, eps_ps, zero_mask);

                __m256 qf = _mm256_div_ps(_mm256_add_ps(w, offset), scale);
                qf = _mm256_max_ps(qf, zero_ps);
                qf = _mm256_min_ps(qf, maxq_ps);

                const __m256i q = _mm256_cvtps_epi32(qf);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(lane_qvals[lane]), q);
            }

            __m256i A = _mm256_setzero_si256();
            for (int j = 0; j < 10; ++j) {
                const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[j]));
                A = _mm256_or_si256(A, _mm256_sllv_epi32(qv, _mm256_set1_epi32(3 * j)));
            }
            A = _mm256_or_si256(
                A,
                _mm256_sllv_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[10])),
                    _mm256_set1_epi32(30)));

            __m256i B = _mm256_and_si256(
                _mm256_srli_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[10])),
                    2),
                _mm256_set1_epi32(0x1));
            for (int j = 0; j < 10; ++j) {
                const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[11 + j]));
                B = _mm256_or_si256(B, _mm256_sllv_epi32(qv, _mm256_set1_epi32(3 * j + 1)));
            }
            B = _mm256_or_si256(
                B,
                _mm256_sllv_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[21])),
                    _mm256_set1_epi32(31)));

            __m256i C = _mm256_and_si256(
                _mm256_srli_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[21])),
                    1),
                _mm256_set1_epi32(0x3));
            for (int j = 0; j < 10; ++j) {
                const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_qvals[22 + j]));
                C = _mm256_or_si256(C, _mm256_sllv_epi32(qv, _mm256_set1_epi32(3 * j + 2)));
            }

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(qweight_ptr + (row_base + 0) * out_features + out), A);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(qweight_ptr + (row_base + 1) * out_features + out), B);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(qweight_ptr + (row_base + 2) * out_features + out), C);
        }
    }
}

#endif // PACK_BLOCK_CPU_X86

template <int bits, int pack_factor>
void dispatch_pack_qweight(
    const float* weight_ptr,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const int32_t* gidx_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t in_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t num_blocks,
    int64_t groups,
    int max_q,
    int64_t threads,
    int64_t block_in) {
    const int64_t threads_eff = clamped_threads(threads);
    const int64_t blocks_per_chunk = std::max<int64_t>(1, block_in / 32);
    int64_t grain = num_blocks / threads_eff;
    if (grain <= 0) {
        grain = 1;
    }
    if (grain > blocks_per_chunk) {
        grain = blocks_per_chunk;
    }

#if PACK_BLOCK_CPU_X86
    // Gather indices are 32-bit; ensure the largest per-vector offset stays in range.
    const bool gather_offsets_safe = out_stride <= (std::numeric_limits<int32_t>::max() / 16);
    if (gather_offsets_safe && out_features % 16 == 0 && cpu_supports_avx512()) {
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_avx512<bits, pack_factor>(
                weight_ptr,
                scales_ptr,
                scale_zeros_ptr,
                gidx_ptr,
                qweight_ptr,
                out_features,
                out_stride,
                scales_stride,
                begin,
                end,
                groups,
                max_q);
        });
        return;
    }
    if (gather_offsets_safe && out_features % 8 == 0 && cpu_supports_avx2()) {
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_avx2<bits, pack_factor>(
                weight_ptr,
                scales_ptr,
                scale_zeros_ptr,
                gidx_ptr,
                qweight_ptr,
                out_features,
                out_stride,
                scales_stride,
                begin,
                end,
                groups,
                max_q);
        });
        return;
    }
#endif

    at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
        pack_qweight_scalar(
            weight_ptr,
            scales_ptr,
            scale_zeros_ptr,
            gidx_ptr,
            qweight_ptr,
            out_features,
            out_stride,
            scales_stride,
            begin,
            end,
            groups,
            bits,
            max_q);
    });
}

void dispatch_pack_qweight_3bit(
    const float* weight_ptr,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const int32_t* gidx_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t in_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t num_blocks,
    int64_t groups,
    int max_q,
    int64_t threads,
    int64_t block_in) {
    const int64_t threads_eff = clamped_threads(threads);
    const int64_t blocks_per_chunk = std::max<int64_t>(1, block_in / 32);
    int64_t grain = num_blocks / threads_eff;
    if (grain <= 0) {
        grain = 1;
    }
    if (grain > blocks_per_chunk) {
        grain = blocks_per_chunk;
    }

#if PACK_BLOCK_CPU_X86
    const bool gather_offsets_safe = out_stride <= (std::numeric_limits<int32_t>::max() / 16);
    if (gather_offsets_safe && out_features % 16 == 0 && cpu_supports_avx512()) {
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_3bit_avx512(
                weight_ptr,
                scales_ptr,
                scale_zeros_ptr,
                gidx_ptr,
                qweight_ptr,
                out_features,
                out_stride,
                scales_stride,
                begin,
                end,
                groups,
                max_q);
        });
        return;
    }
    if (gather_offsets_safe && out_features % 8 == 0 && cpu_supports_avx2()) {
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_3bit_avx2(
                weight_ptr,
                scales_ptr,
                scale_zeros_ptr,
                gidx_ptr,
                qweight_ptr,
                out_features,
                out_stride,
                scales_stride,
                begin,
                end,
                groups,
                max_q);
        });
        return;
    }
#endif

    at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
        pack_qweight_scalar(
            weight_ptr,
            scales_ptr,
            scale_zeros_ptr,
            gidx_ptr,
            qweight_ptr,
            out_features,
            out_stride,
            scales_stride,
            begin,
            end,
            groups,
            3,
            max_q);
    });
}

void pack_qzeros_scalar(
    const int32_t* zeros_ptr,
    int64_t zeros_stride,
    int32_t* qzeros_ptr,
    int64_t qzeros_cols,
    int64_t g_begin,
    int64_t g_end,
    int bits) {
    if (bits == 3) {
        for (int64_t g = g_begin; g < g_end; ++g) {
            const int32_t* zeros_row = zeros_ptr + g * zeros_stride;
            int32_t* dst_row = qzeros_ptr + g * qzeros_cols;
            int64_t idx = 0;
            for (int64_t col = 0; col < qzeros_cols;) {
                int64_t A = 0;
                for (int j = 0; j < 10; ++j) {
                    A |= static_cast<int64_t>(zeros_row[idx + j]) << (3 * j);
                }
                A |= static_cast<int64_t>(zeros_row[idx + 10]) << 30;
                dst_row[col++] = static_cast<int32_t>(A & 0xFFFFFFFF);

                int64_t B = static_cast<int64_t>((zeros_row[idx + 10] >> 2) & 0x1);
                for (int j = 0; j < 10; ++j) {
                    B |= static_cast<int64_t>(zeros_row[idx + 11 + j]) << (3 * j + 1);
                }
                B |= static_cast<int64_t>(zeros_row[idx + 21]) << 31;
                dst_row[col++] = static_cast<int32_t>(B & 0xFFFFFFFF);

                int64_t C = static_cast<int64_t>((zeros_row[idx + 21] >> 1) & 0x3);
                for (int j = 0; j < 10; ++j) {
                    C |= static_cast<int64_t>(zeros_row[idx + 22 + j]) << (3 * j + 2);
                }
                dst_row[col++] = static_cast<int32_t>(C & 0xFFFFFFFF);

                idx += 32;
            }
        }
        return;
    }

    const int pack_factor = 32 / bits;
    for (int64_t g = g_begin; g < g_end; ++g) {
        const int32_t* zeros_row = zeros_ptr + g * zeros_stride;
        int32_t* dst_row = qzeros_ptr + g * qzeros_cols;
        for (int64_t col = 0; col < qzeros_cols; ++col) {
            int64_t base = col * pack_factor;
            int64_t packed = 0;
            for (int pf = 0; pf < pack_factor; ++pf) {
                packed |= static_cast<int64_t>(zeros_row[base + pf]) << (bits * pf);
            }
            dst_row[col] = static_cast<int32_t>(packed & 0xFFFFFFFF);
        }
    }
}

void pack_qzeros(
    const int32_t* zeros_ptr,
    int64_t zeros_stride,
    int32_t* qzeros_ptr,
    int64_t qzeros_cols,
    int64_t groups,
    int bits,
    int64_t threads) {
    const int64_t threads_eff = clamped_threads(threads);
    const int64_t grain = std::max<int64_t>(1, groups / threads_eff);
    at::parallel_for(0, groups, grain, [&](int64_t begin, int64_t end) {
        pack_qzeros_scalar(zeros_ptr, zeros_stride, qzeros_ptr, qzeros_cols, begin, end, bits);
    });
}

} // namespace

std::tuple<at::Tensor, at::Tensor> pack_block_cpu(
    const at::Tensor& weight,
    const at::Tensor& scales,
    const at::Tensor& zeros,
    const at::Tensor& g_idx,
    int64_t bits,
    int64_t word_bits,
    int64_t block_in,
    int64_t threads) {
    TORCH_CHECK(weight.device().is_cpu(), "weight must reside on CPU");
    TORCH_CHECK(scales.device().is_cpu(), "scales must reside on CPU");
    TORCH_CHECK(zeros.device().is_cpu(), "zeros must reside on CPU");
    TORCH_CHECK(g_idx.device().is_cpu(), "g_idx must reside on CPU");

    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out, in]");
    TORCH_CHECK(scales.dim() == 2, "scales must be 2D [groups, out]");
    TORCH_CHECK(zeros.dim() == 2, "zeros must be 2D [groups, out]");
    TORCH_CHECK(g_idx.dim() == 1, "g_idx must be 1D [in]");

    TORCH_CHECK(word_bits == 32, "Only 32-bit packing supported");

    at::Tensor weight_f = weight.contiguous().to(at::kFloat);
    at::Tensor scales_f = scales.contiguous().to(at::kFloat);
    at::Tensor zeros_i32 = zeros.contiguous().to(at::kInt);
    at::Tensor g_idx_i32 = g_idx.contiguous().to(at::kInt);

    at::Tensor scale_zeros = zeros_i32.to(at::kFloat) * scales_f;

    const int64_t out_features = weight_f.size(0);
    const int64_t in_features = weight_f.size(1);
    TORCH_CHECK(g_idx_i32.size(0) == in_features, "g_idx length mismatch");
    TORCH_CHECK(in_features % word_bits == 0, "in_features must be divisible by word_bits");

    const int64_t groups = scales_f.size(0);
    TORCH_CHECK(scales_f.size(1) == out_features, "scales shape mismatch");
    TORCH_CHECK(zeros_i32.size(0) == groups && zeros_i32.size(1) == out_features, "zeros shape mismatch");
    TORCH_CHECK(out_features % word_bits == 0, "out_features must be divisible by word_bits");

    if (block_in <= 0) {
        block_in = word_bits;
    }
    block_in = std::max<int64_t>(word_bits, (block_in / word_bits) * word_bits);
    if (block_in == 0) {
        block_in = word_bits;
    }

    const int rows_per_group = (bits == 3) ? 3 : static_cast<int>(bits);
    const int64_t num_blocks = in_features / word_bits;

    auto q_options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    at::Tensor qweight = at::empty({num_blocks * rows_per_group, out_features}, q_options);

    const int max_q = (1 << bits) - 1;
    const float* weight_ptr = weight_f.const_data_ptr<float>();
    const float* scales_ptr = scales_f.const_data_ptr<float>();
    const float* scale_zeros_ptr = scale_zeros.const_data_ptr<float>();
    const int32_t* gidx_ptr = g_idx_i32.const_data_ptr<int32_t>();
    int32_t* qweight_ptr = qweight.data_ptr<int32_t>();

    const int64_t out_stride = in_features;
    const int64_t scales_stride = out_features;

    if (bits == 2 || bits == 4 || bits == 8) {
        switch (bits) {
            case 2:
                dispatch_pack_qweight<2, 16>(
                    weight_ptr,
                    scales_ptr,
                    scale_zeros_ptr,
                    gidx_ptr,
                    qweight_ptr,
                    out_features,
                    in_features,
                    out_stride,
                    scales_stride,
                    num_blocks,
                    groups,
                    max_q,
                    threads,
                    block_in);
                break;
            case 4:
                dispatch_pack_qweight<4, 8>(
                    weight_ptr,
                    scales_ptr,
                    scale_zeros_ptr,
                    gidx_ptr,
                    qweight_ptr,
                    out_features,
                    in_features,
                    out_stride,
                    scales_stride,
                    num_blocks,
                    groups,
                    max_q,
                    threads,
                    block_in);
                break;
            case 8:
                dispatch_pack_qweight<8, 4>(
                    weight_ptr,
                    scales_ptr,
                    scale_zeros_ptr,
                    gidx_ptr,
                    qweight_ptr,
                    out_features,
                    in_features,
                    out_stride,
                    scales_stride,
                    num_blocks,
                    groups,
                    max_q,
                    threads,
                    block_in);
                break;
        }
    } else if (bits == 3) {
        dispatch_pack_qweight_3bit(
            weight_ptr,
            scales_ptr,
            scale_zeros_ptr,
            gidx_ptr,
            qweight_ptr,
            out_features,
            in_features,
            out_stride,
            scales_stride,
            num_blocks,
            groups,
            max_q,
            threads,
            block_in);
    } else {
        TORCH_CHECK(false, "Unsupported bits value", bits);
    }

    at::Tensor zeros_i32_contig = zeros_i32.contiguous();
    const int32_t* zeros_ptr = zeros_i32_contig.const_data_ptr<int32_t>();
    const int64_t zeros_stride = zeros_i32_contig.size(1);

    int64_t qzeros_cols = (out_features / word_bits) * bits;
    at::Tensor qzeros = at::zeros({groups, qzeros_cols}, q_options);
    int32_t* qzeros_ptr = qzeros.data_ptr<int32_t>();

    pack_qzeros(zeros_ptr, zeros_stride, qzeros_ptr, qzeros_cols, groups, static_cast<int>(bits), threads);

    return {qweight, qzeros};
}

// AWQ packing helper. Takes pre-quantized integer tensors and packs them into the
// interleaved AWQ layout (e.g. 4-bit order 0,2,4,6,1,3,5,7 per 32-bit word).
std::tuple<at::Tensor, at::Tensor> pack_awq_cpu(
    const at::Tensor& intweight,
    const at::Tensor& zeros,
    int64_t bits) {
    TORCH_CHECK(intweight.device().is_cpu(), "pack_awq_cpu: intweight must reside on CPU");
    TORCH_CHECK(zeros.device().is_cpu(), "pack_awq_cpu: zeros must reside on CPU");

    TORCH_CHECK(intweight.dim() == 2, "pack_awq_cpu: intweight must be 2D [in, out]");
    TORCH_CHECK(zeros.dim() == 2, "pack_awq_cpu: zeros must be 2D [groups, out]");
    TORCH_CHECK(bits == 2 || bits == 4 || bits == 8, "pack_awq_cpu: bits must be 2, 4, or 8");

    const int pack_factor = 32 / static_cast<int>(bits);
    const int max_q = (1 << bits) - 1;

    const int64_t in_features = intweight.size(0);
    const int64_t out_features = intweight.size(1);
    const int64_t groups = zeros.size(0);
    TORCH_CHECK(zeros.size(1) == out_features, "pack_awq_cpu: zeros second dim must match out_features");

    const int64_t out_packs = out_features / pack_factor;
    TORCH_CHECK(out_features % pack_factor == 0, "pack_awq_cpu: out_features must be divisible by pack_factor");

    at::Tensor intweight_i32 = intweight.contiguous().to(at::kInt);
    at::Tensor zeros_i32 = zeros.contiguous().to(at::kInt);

    auto q_options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    at::Tensor qweight = at::empty({in_features, out_packs}, q_options);
    at::Tensor qzeros = at::empty({groups, out_packs}, q_options);

    const int32_t* iw_ptr = intweight_i32.const_data_ptr<int32_t>();
    const int32_t* z_ptr = zeros_i32.const_data_ptr<int32_t>();
    int32_t* qw_ptr = qweight.data_ptr<int32_t>();
    int32_t* qz_ptr = qzeros.data_ptr<int32_t>();

    const int64_t iw_stride_in = intweight_i32.stride(0);
    const int64_t iw_stride_out = intweight_i32.stride(1);
    const int64_t z_stride_group = zeros_i32.stride(0);
    const int64_t z_stride_out = zeros_i32.stride(1);

    // AWQ interleaves the 4-bit columns. For 2/8-bit use the natural order.
    int order[16];
    for (int i = 0; i < pack_factor; ++i) {
        order[i] = i;
    }
    if (bits == 4) {
        const int awq_order_4[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        for (int i = 0; i < 8; ++i) {
            order[i] = awq_order_4[i];
        }
    }

    at::parallel_for(0, in_features * out_packs, 0, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            const int64_t i = idx / out_packs;
            const int64_t p = idx % out_packs;
            int32_t packed = 0;
            for (int k = 0; k < pack_factor; ++k) {
                const int64_t o = p * pack_factor + order[k];
                int32_t v = iw_ptr[i * iw_stride_in + o * iw_stride_out];
                v = std::max<int32_t>(0, std::min<int32_t>(v, max_q));
                packed |= (v & max_q) << (bits * k);
            }
            qw_ptr[i * out_packs + p] = packed;
        }
    });

    at::parallel_for(0, groups * out_packs, 0, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; ++idx) {
            const int64_t g = idx / out_packs;
            const int64_t p = idx % out_packs;
            int32_t packed = 0;
            for (int k = 0; k < pack_factor; ++k) {
                const int64_t o = p * pack_factor + order[k];
                int32_t v = z_ptr[g * z_stride_group + o * z_stride_out];
                v = std::max<int32_t>(0, std::min<int32_t>(v, max_q));
                packed |= (v & max_q) << (bits * k);
            }
            qz_ptr[g * out_packs + p] = packed;
        }
    });

    return {qweight, qzeros};
}

} // namespace gptqmodel

TORCH_LIBRARY(gptqmodel, m) {
    m.def(
        "pack_block_cpu(Tensor weight, Tensor scales, Tensor zeros, Tensor g_idx, int bits, int word_bits, int block_in, int threads) -> (Tensor, Tensor)"
    );
    m.impl(
        "pack_block_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::pack_block_cpu)
    );
    m.def(
        "pack_awq_cpu(Tensor intweight, Tensor zeros, int bits) -> (Tensor, Tensor)"
    );
    m.impl(
        "pack_awq_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::pack_awq_cpu)
    );
}
