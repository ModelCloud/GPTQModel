// SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
// SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0
// Contact: qubitium@modelcloud.ai, x.com/qubitium

#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <torch/library.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

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

inline int64_t clamped_threads(int64_t requested, int64_t rows, int64_t cols) {
    const int64_t hard_limit = 32;
    const int64_t available = at::get_num_threads();
    // Auto-select thread count based on workload size.  Target one thread per
    // ~512K output elements to keep OpenMP/allocator overhead low for small
    // layers while still scaling up to all available threads for large layers.
    const int64_t n = rows * cols;
    const int64_t per_thread = 524288;
    int64_t auto_threads = (n + per_thread - 1) / per_thread;
    if (auto_threads < 1) {
        auto_threads = 1;
    }
    auto_threads = std::min<int64_t>(auto_threads, std::min<int64_t>(available, hard_limit));
    if (requested > 0) {
        return std::max<int64_t>(1, std::min<int64_t>(requested, auto_threads));
    }
    return auto_threads;
}

// RAII helper to temporarily limit the number of OpenMP/ATen threads for the
// packing loops.  Conversions (aten::to) are done before the guard is created so
// they remain multi-threaded; only the cache-sensitive bit-packing loops run
// with the requested thread count.
//
// We set the OpenMP thread limit directly with omp_set_num_threads because
// at::set_num_threads from a JIT extension may resolve to a per-module inline
// copy and not affect libtorch's at::parallel_for scheduling.
class NumThreadsGuard {
    int saved_;
    bool active_;

public:
    explicit NumThreadsGuard(int64_t requested)
#if defined(_OPENMP)
        : saved_(omp_get_max_threads()), active_(false) {
        if (requested > 0 && requested != saved_) {
            omp_set_num_threads(static_cast<int>(requested));
            active_ = true;
        }
#else
        : saved_(at::get_num_threads()), active_(false) {
        if (requested > 0 && requested != saved_) {
            at::set_num_threads(static_cast<int>(requested));
            active_ = true;
        }
#endif
    }
    ~NumThreadsGuard() {
        if (active_) {
#if defined(_OPENMP)
            omp_set_num_threads(saved_);
#else
            at::set_num_threads(saved_);
#endif
        }
    }
};

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

struct PackBlockRun {
    int64_t start;
    int64_t count;
    int32_t group;
};

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

// Convert 16 packed bfloat16 values to 16 floats by shifting the 16-bit pattern
// into the upper half of a 32-bit float word (lower mantissa bits are zero).
__attribute__((target("avx512f")))
inline __m512 bf16_vec_to_float(__m256i v16) {
    const __m512i v32 = _mm512_cvtepu16_epi32(v16);
    return _mm512_castsi512_ps(_mm512_slli_epi32(v32, 16));
}

// Load 16 output rows of 32 input weights and transpose into two 16x16 float
// blocks: t0 covers input lanes 0..15 and t1 covers lanes 16..31.
__attribute__((target("avx512f")))
inline void pack_qweight_load_transpose_32x16(
    const void* weight_ptr,
    bool weight_is_bf16,
    int64_t out,
    int64_t out_stride,
    int64_t base_input,
    float* t0,
    float* t1) {
    if (weight_is_bf16) {
        const at::BFloat16* w = static_cast<const at::BFloat16*>(weight_ptr);
        alignas(64) float row_f0[16][16];
        alignas(64) float row_f1[16][16];
        for (int i = 0; i < 16; ++i) {
            const at::BFloat16* p = w + (out + i) * out_stride + base_input;
            const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16));
            _mm512_storeu_ps(&row_f0[i][0], bf16_vec_to_float(lo));
            _mm512_storeu_ps(&row_f1[i][0], bf16_vec_to_float(hi));
        }
        transpose_16x16_avx512(&row_f0[0][0], 16, t0, 16);
        transpose_16x16_avx512(&row_f1[0][0], 16, t1, 16);
    } else {
        const float* w = static_cast<const float*>(weight_ptr);
        const float* src0 = w + out * out_stride + base_input;
        transpose_16x16_avx512(src0, out_stride, t0, 16);
        transpose_16x16_avx512(src0 + 16, out_stride, t1, 16);
    }
}

// Uniform-block AVX-512 helper: all 32 lanes share one scale/offset.
// Use one true division per weight to stay bit-exact with the Python reference.
template <int bits, int pack_factor>
__attribute__((target("avx512f"))) __attribute__((always_inline))
inline void pack_qweight_avx512_uniform_block(
    const void* weight_ptr,
    bool weight_is_bf16,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t base_input,
    int row_base,
    int32_t group,
    int max_q) {
    const __m512 zero_ps = _mm512_setzero_ps();
    const __m512 eps_ps = _mm512_set1_ps(1e-6f);
    const __m512 maxq_ps = _mm512_set1_ps(static_cast<float>(max_q));
    const __m512i zero_epi = _mm512_setzero_si512();

    alignas(64) float t0_a[16][16];
    alignas(64) float t1_a[16][16];
    alignas(64) float t0_b[16][16];
    alignas(64) float t1_b[16][16];

    const int64_t scale_offset_base = static_cast<int64_t>(group) * scales_stride;

    for (int64_t out = 0; out < out_features; out += 32) {
        const int64_t out_b = out + 16;
        const bool has_b = (out_b < out_features);

        pack_qweight_load_transpose_32x16(
            weight_ptr, weight_is_bf16, out, out_stride, base_input, &t0_a[0][0], &t1_a[0][0]);
        if (has_b) {
            pack_qweight_load_transpose_32x16(
                weight_ptr, weight_is_bf16, out_b, out_stride, base_input, &t0_b[0][0], &t1_b[0][0]);
        }

        __m512 scale_a = _mm512_loadu_ps(scales_ptr + scale_offset_base + out);
        const __m512 offset_a = _mm512_loadu_ps(scale_zeros_ptr + scale_offset_base + out);
        __mmask16 zero_mask = _mm512_cmp_ps_mask(scale_a, zero_ps, _CMP_EQ_OQ);
        scale_a = _mm512_mask_blend_ps(zero_mask, scale_a, eps_ps);

        __m512 scale_b = zero_ps;
        __m512 offset_b = zero_ps;
        if (has_b) {
            scale_b = _mm512_loadu_ps(scales_ptr + scale_offset_base + out_b);
            offset_b = _mm512_loadu_ps(scale_zeros_ptr + scale_offset_base + out_b);
            zero_mask = _mm512_cmp_ps_mask(scale_b, zero_ps, _CMP_EQ_OQ);
            scale_b = _mm512_mask_blend_ps(zero_mask, scale_b, eps_ps);
        }

        __m512i acc_a[bits];
        __m512i acc_b[bits];
        for (int bp = 0; bp < bits; ++bp) {
            acc_a[bp] = zero_epi;
            acc_b[bp] = zero_epi;
        }

        for (int lane = 0; lane < 32; ++lane) {
            const float* ptr_a = (lane < 16) ? &t0_a[lane][0] : &t1_a[lane - 16][0];
            const __m512 w_a = _mm512_loadu_ps(ptr_a);
            __m512 qf_a = _mm512_div_ps(_mm512_add_ps(w_a, offset_a), scale_a);
            qf_a = _mm512_max_ps(qf_a, zero_ps);
            qf_a = _mm512_min_ps(qf_a, maxq_ps);
            const __m512i q_a = _mm512_cvt_roundps_epi32(
                qf_a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const int idx_a = lane / pack_factor;
            const int shift_a = bits * (lane - idx_a * pack_factor);
            acc_a[idx_a] = _mm512_or_si512(
                acc_a[idx_a], _mm512_sllv_epi32(q_a, _mm512_set1_epi32(shift_a)));

            if (has_b) {
                const float* ptr_b = (lane < 16) ? &t0_b[lane][0] : &t1_b[lane - 16][0];
                const __m512 w_b = _mm512_loadu_ps(ptr_b);
                __m512 qf_b = _mm512_div_ps(_mm512_add_ps(w_b, offset_b), scale_b);
                qf_b = _mm512_max_ps(qf_b, zero_ps);
                qf_b = _mm512_min_ps(qf_b, maxq_ps);
                const __m512i q_b = _mm512_cvt_roundps_epi32(
                    qf_b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const int idx_b = lane / pack_factor;
                const int shift_b = bits * (lane - idx_b * pack_factor);
                acc_b[idx_b] = _mm512_or_si512(
                    acc_b[idx_b], _mm512_sllv_epi32(q_b, _mm512_set1_epi32(shift_b)));
            }
        }

        for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
            _mm512_storeu_si512(
                reinterpret_cast<__m512i*>(qweight_ptr + (row_base + bit_plane) * out_features + out),
                acc_a[bit_plane]);
            if (has_b) {
                _mm512_storeu_si512(
                    reinterpret_cast<__m512i*>(qweight_ptr + (row_base + bit_plane) * out_features + out_b),
                    acc_b[bit_plane]);
            }
        }
    }
}

// Fused uniform-block AVX-512 helper: process multiple consecutive 32-input
// blocks that all share the same group, amortizing scale/offset loads.
template <int bits, int pack_factor>
__attribute__((target("avx512f"))) __attribute__((always_inline))
inline void pack_qweight_avx512_uniform_fused(
    const void* weight_ptr,
    bool weight_is_bf16,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_begin,
    int64_t out_end,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t block_idx_start,
    int64_t block_count,
    int32_t group,
    int max_q) {
    const __m512 zero_ps = _mm512_setzero_ps();
    const __m512 eps_ps = _mm512_set1_ps(1e-6f);
    const __m512 maxq_ps = _mm512_set1_ps(static_cast<float>(max_q));
    const __m512i zero_epi = _mm512_setzero_si512();

    alignas(64) float t0[16][16];
    alignas(64) float t1[16][16];

    const int64_t scale_offset_base = static_cast<int64_t>(group) * scales_stride;

    for (int64_t out = out_begin; out < out_end; out += 16) {
        __m512 scale = _mm512_loadu_ps(scales_ptr + scale_offset_base + out);
        const __m512 offset = _mm512_loadu_ps(scale_zeros_ptr + scale_offset_base + out);
        const __mmask16 zero_mask = _mm512_cmp_ps_mask(scale, zero_ps, _CMP_EQ_OQ);
        scale = _mm512_mask_blend_ps(zero_mask, scale, eps_ps);

        for (int64_t b = 0; b < block_count; ++b) {
            const int64_t base_input = (block_idx_start + b) * 32;
            const int row_base = static_cast<int>((block_idx_start + b) * bits);
            pack_qweight_load_transpose_32x16(
                weight_ptr, weight_is_bf16, out, out_stride, base_input, &t0[0][0], &t1[0][0]);

            __m512i acc[bits];
            for (int bp = 0; bp < bits; ++bp) {
                acc[bp] = zero_epi;
            }

            for (int lane = 0; lane < 32; ++lane) {
                const float* ptr = (lane < 16) ? &t0[lane][0] : &t1[lane - 16][0];
                const __m512 w = _mm512_loadu_ps(ptr);
                __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                qf = _mm512_max_ps(qf, zero_ps);
                qf = _mm512_min_ps(qf, maxq_ps);
                const __m512i q = _mm512_cvt_roundps_epi32(
                    qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const int idx = lane / pack_factor;
                const int shift = bits * (lane - idx * pack_factor);
                acc[idx] = _mm512_or_si512(
                    acc[idx], _mm512_sllv_epi32(q, _mm512_set1_epi32(shift)));
            }

            for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
                _mm512_storeu_si512(
                    reinterpret_cast<__m512i*>(qweight_ptr + (row_base + bit_plane) * out_features + out),
                    acc[bit_plane]);
            }
        }
    }
}

// Non-uniform-block AVX-512 helper: each lane may use a different scale/offset.
template <int bits, int pack_factor>
__attribute__((target("avx512f"))) __attribute__((always_inline))
inline void pack_qweight_avx512_nonuniform_block(
    const void* weight_ptr,
    bool weight_is_bf16,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_stride,
    int64_t scales_stride,
    int64_t base_input,
    int row_base,
    const int32_t* gidx_block,
    int max_q) {
    const __m512 zero_ps = _mm512_setzero_ps();
    const __m512 eps_ps = _mm512_set1_ps(1e-6f);
    const __m512 maxq_ps = _mm512_set1_ps(static_cast<float>(max_q));
    const __m512i zero_epi = _mm512_setzero_si512();

    alignas(64) int32_t lane_qvals[32][16];
    alignas(64) float t0[16][16];
    alignas(64) float t1[16][16];

    for (int64_t out = 0; out < out_features; out += 16) {
        pack_qweight_load_transpose_32x16(
            weight_ptr, weight_is_bf16, out, out_stride, base_input, &t0[0][0], &t1[0][0]);

        for (int lane = 0; lane < 32; ++lane) {
            const int32_t group = gidx_block[lane];
            const float* w_ptr = (lane < 16) ? &t0[lane][0] : &t1[lane - 16][0];
            const __m512 w = _mm512_loadu_ps(w_ptr);

            const float* sbase = scales_ptr + static_cast<int64_t>(group) * scales_stride + out;
            __m512 scale = _mm512_loadu_ps(sbase);
            const __m512 offset = _mm512_loadu_ps(
                scale_zeros_ptr + static_cast<int64_t>(group) * scales_stride + out);

            const __mmask16 zero_mask = _mm512_cmp_ps_mask(scale, zero_ps, _CMP_EQ_OQ);
            scale = _mm512_mask_blend_ps(zero_mask, scale, eps_ps);

            __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
            qf = _mm512_max_ps(qf, zero_ps);
            qf = _mm512_min_ps(qf, maxq_ps);

            const __m512i q = _mm512_cvt_roundps_epi32(
                qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(lane_qvals[lane]), q);
        }

        for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
            __m512i acc = zero_epi;
            for (int pf = 0; pf < pack_factor; ++pf) {
                const int idx = bit_plane * pack_factor + pf;
                const int shift = bits * pf;
                const __m512i qv = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(lane_qvals[idx]));
                acc = _mm512_or_si512(acc, _mm512_sllv_epi32(qv, _mm512_set1_epi32(shift)));
            }
            _mm512_storeu_si512(
                reinterpret_cast<__m512i*>(qweight_ptr + (row_base + bit_plane) * out_features + out),
                acc);
        }
    }
}

// 2D output-channel tiling for uniform g_idx: parallelize over output tiles and
// process each input-group run over the tile, so weight rows and qweight rows are
// both accessed in contiguous streams.
template <int bits, int pack_factor>
__attribute__((target("avx512f")))
void pack_qweight_avx512_output_tiled(
    const void* weight_ptr,
    bool weight_is_bf16,
    const float* scales_ptr,
    const float* scale_zeros_ptr,
    const PackBlockRun* runs,
    int64_t num_runs,
    int32_t* qweight_ptr,
    int64_t out_features,
    int64_t out_begin,
    int64_t out_end,
    int64_t out_stride,
    int64_t scales_stride,
    int max_q) {
    for (int64_t r = 0; r < num_runs; ++r) {
        pack_qweight_avx512_uniform_fused<bits, pack_factor>(
            weight_ptr,
            weight_is_bf16,
            scales_ptr,
            scale_zeros_ptr,
            qweight_ptr,
            out_features,
            out_begin,
            out_end,
            out_stride,
            scales_stride,
            runs[r].start,
            runs[r].count,
            runs[r].group,
            max_q);
    }
}

// AVX-512 dispatcher for bits 2, 4, 8.  Delegates uniform vs non-uniform blocks to
// dedicated helpers; for uniform desc_act=False groups it fuses consecutive
// same-group 32-input blocks to amortize scale/offset loads.
template <int bits, int pack_factor>
__attribute__((target("avx512f")))
void pack_qweight_avx512(
    const void* weight_ptr,
    bool weight_is_bf16,
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
    int64_t block_idx = block_begin;
    while (block_idx < block_end) {
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
            const int32_t group = gidx_block[0];
            // Find how many consecutive blocks share this exact uniform group.
            int64_t run_end = block_idx + 1;
            while (run_end < block_end) {
                const int64_t next_base = run_end * 32;
                alignas(64) int32_t next_gidx[32];
                normalize_gidx_block(gidx_ptr, next_base, groups, next_gidx);
                bool next_uniform = true;
                for (int i = 0; i < 32; ++i) {
                    if (next_gidx[i] != group) {
                        next_uniform = false;
                        break;
                    }
                }
                if (!next_uniform) {
                    break;
                }
                ++run_end;
            }
            const int64_t block_count = run_end - block_idx;
            if (block_count == 1) {
                pack_qweight_avx512_uniform_block<bits, pack_factor>(
                    weight_ptr, weight_is_bf16, scales_ptr, scale_zeros_ptr, qweight_ptr,
                    out_features, out_stride, scales_stride, base_input, row_base,
                    group, max_q);
            } else {
                pack_qweight_avx512_uniform_fused<bits, pack_factor>(
                    weight_ptr, weight_is_bf16, scales_ptr, scale_zeros_ptr, qweight_ptr,
                    out_features, 0, out_features, out_stride, scales_stride, block_idx, block_count,
                    group, max_q);
            }
            block_idx = run_end;
        } else {
            pack_qweight_avx512_nonuniform_block<bits, pack_factor>(
                weight_ptr, weight_is_bf16, scales_ptr, scale_zeros_ptr, qweight_ptr,
                out_features, out_stride, scales_stride, base_input, row_base,
                gidx_block, max_q);
            ++block_idx;
        }
    }
}

// AVX-512 3-bit packing helper.  Each 32-element lane block produces 3 packed 32-bit words.
__attribute__((target("avx512f")))
void pack_qweight_3bit_avx512(
    const void* weight_ptr,
    bool weight_is_bf16,
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
                pack_qweight_load_transpose_32x16(
                    weight_ptr, weight_is_bf16, out, out_stride, base_input, &t0[0][0], &t1[0][0]);

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
                pack_qweight_load_transpose_32x16(
                    weight_ptr, weight_is_bf16, out, out_stride, base_input, &t0[0][0], &t1[0][0]);

                for (int lane = 0; lane < 32; ++lane) {
                    const int32_t group = gidx_block[lane];
                    const float* w_ptr = (lane < 16) ? &t0[lane][0] : &t1[lane - 16][0];
                    const __m512 w = _mm512_loadu_ps(w_ptr);

                    const float* sbase = scales_ptr + static_cast<int64_t>(group) * scales_stride + out;
                    __m512 scale = _mm512_loadu_ps(sbase);
                    const __m512 offset = _mm512_loadu_ps(
                        scale_zeros_ptr + static_cast<int64_t>(group) * scales_stride + out);

                    const __mmask16 zero_mask = _mm512_cmp_ps_mask(scale, zero_ps, _CMP_EQ_OQ);
                    scale = _mm512_mask_blend_ps(zero_mask, scale, eps_ps);

                    __m512 qf = _mm512_div_ps(_mm512_add_ps(w, offset), scale);
                    qf = _mm512_max_ps(qf, zero_ps);
                    qf = _mm512_min_ps(qf, maxq_ps);

                    const __m512i q = _mm512_cvt_roundps_epi32(
                        qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
    const void* weight_ptr,
    bool weight_is_bf16,
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
    const int64_t threads_eff = clamped_threads(threads, in_features, out_features);
    const int64_t blocks_per_chunk = std::max<int64_t>(1, block_in / 32);
    int64_t grain = num_blocks / threads_eff;
    if (grain <= 0) {
        grain = 1;
    }
    if (grain > blocks_per_chunk) {
        grain = blocks_per_chunk;
    }

#if PACK_BLOCK_CPU_X86
    // AVX-512 path loads contiguous 16x32 weight blocks and transposes them,
    // so it no longer relies on 32-bit gather offsets.
    if (out_features % 16 == 0 && cpu_supports_avx512()) {
        // desc_act=False gives uniform 32-input blocks. Fuse consecutive blocks
        // that share the same group, then parallelize over output-channel tiles
        // so each thread streams a contiguous output slice through all groups.
        bool all_uniform = true;
        std::vector<int32_t> block_groups(num_blocks);
        for (int64_t b = 0; b < num_blocks; ++b) {
            alignas(64) int32_t gidx_block[32];
            normalize_gidx_block(gidx_ptr, b * 32, groups, gidx_block);
            bool uniform = true;
            for (int i = 1; i < 32; ++i) {
                if (gidx_block[i] != gidx_block[0]) {
                    uniform = false;
                    break;
                }
            }
            if (!uniform) {
                all_uniform = false;
                break;
            }
            block_groups[b] = gidx_block[0];
        }
        if (all_uniform) {
            std::vector<PackBlockRun> runs;
            runs.reserve(num_blocks);
            int64_t b = 0;
            while (b < num_blocks) {
                int64_t e = b + 1;
                while (e < num_blocks && block_groups[e] == block_groups[b]) {
                    ++e;
                }
                runs.push_back({b, e - b, block_groups[b]});
                b = e;
            }
            // Parallelize over 16-column groups so every task boundary is
            // 16-aligned; out_features is guaranteed to be a multiple of 16 here.
            const int64_t total_out_groups = out_features / 16;
            int64_t out_tile_groups = ((out_features + threads_eff - 1) / threads_eff + 15) / 16;
            if (out_tile_groups < 8) {
                out_tile_groups = 8;
            }
            if (out_tile_groups > total_out_groups) {
                out_tile_groups = total_out_groups;
            }
            at::parallel_for(0, total_out_groups, out_tile_groups, [&](int64_t gb, int64_t ge) {
                const int64_t out_begin = gb * 16;
                const int64_t out_end = ge * 16;
                pack_qweight_avx512_output_tiled<bits, pack_factor>(
                    weight_ptr,
                    weight_is_bf16,
                    scales_ptr,
                    scale_zeros_ptr,
                    runs.data(),
                    static_cast<int64_t>(runs.size()),
                    qweight_ptr,
                    out_features,
                    out_begin,
                    out_end,
                    out_stride,
                    scales_stride,
                    max_q);
            });
            return;
        }
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_avx512<bits, pack_factor>(
                weight_ptr,
                weight_is_bf16,
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
    // Gather indices are 32-bit; ensure the largest per-vector offset stays in range.
    const bool gather_offsets_safe = out_stride <= (std::numeric_limits<int32_t>::max() / 16);
    if (!weight_is_bf16 && gather_offsets_safe && out_features % 8 == 0 && cpu_supports_avx2()) {
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_avx2<bits, pack_factor>(
                static_cast<const float*>(weight_ptr),
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

    TORCH_CHECK(!weight_is_bf16, "bfloat16 weights require AVX-512 packing path");
    at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
        pack_qweight_scalar(
            static_cast<const float*>(weight_ptr),
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
    const void* weight_ptr,
    bool weight_is_bf16,
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
    const int64_t threads_eff = clamped_threads(threads, in_features, out_features);
    const int64_t blocks_per_chunk = std::max<int64_t>(1, block_in / 32);
    int64_t grain = num_blocks / threads_eff;
    if (grain <= 0) {
        grain = 1;
    }
    if (grain > blocks_per_chunk) {
        grain = blocks_per_chunk;
    }

#if PACK_BLOCK_CPU_X86
    if (out_features % 16 == 0 && cpu_supports_avx512()) {
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_3bit_avx512(
                weight_ptr,
                weight_is_bf16,
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
    const bool gather_offsets_safe = out_stride <= (std::numeric_limits<int32_t>::max() / 16);
    if (!weight_is_bf16 && gather_offsets_safe && out_features % 8 == 0 && cpu_supports_avx2()) {
        at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
            pack_qweight_3bit_avx2(
                static_cast<const float*>(weight_ptr),
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

    TORCH_CHECK(!weight_is_bf16, "bfloat16 weights require AVX-512 packing path");
    at::parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
        pack_qweight_scalar(
            static_cast<const float*>(weight_ptr),
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
    int64_t out_features,
    int bits,
    int64_t threads) {
    const int64_t threads_eff = clamped_threads(threads, groups, out_features);
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

    at::Tensor weight_contig;
    at::Tensor weight_f;
    const void* weight_ptr = nullptr;
    bool weight_is_bf16 = false;
    const bool input_is_bf16 = weight.scalar_type() == at::kBFloat16;

    at::Tensor scales_f = scales.contiguous().to(at::kFloat);
    at::Tensor zeros_i32 = zeros.contiguous().to(at::kInt);
    at::Tensor g_idx_i32 = g_idx.contiguous().to(at::kInt);

    const int64_t out_features = weight.size(0);
    const int64_t in_features = weight.size(1);
    const int64_t threads_eff = clamped_threads(threads, in_features, out_features);
    TORCH_CHECK(g_idx_i32.size(0) == in_features, "g_idx length mismatch");
    TORCH_CHECK(in_features % word_bits == 0, "in_features must be divisible by word_bits");

    const int64_t groups = scales_f.size(0);
    TORCH_CHECK(scales_f.size(1) == out_features, "scales shape mismatch");
    TORCH_CHECK(zeros_i32.size(0) == groups && zeros_i32.size(1) == out_features, "zeros shape mismatch");
    TORCH_CHECK(out_features % word_bits == 0, "out_features must be divisible by word_bits");

    bool use_avx512_for_bf16 = false;
#if PACK_BLOCK_CPU_X86
    use_avx512_for_bf16 = input_is_bf16 && cpu_supports_avx512();
#endif
    if (use_avx512_for_bf16) {
        weight_contig = weight.contiguous();
        weight_ptr = weight_contig.const_data_ptr<at::BFloat16>();
        weight_is_bf16 = true;
    } else {
        weight_f = weight.contiguous().to(at::kFloat);
        weight_ptr = weight_f.const_data_ptr<float>();
    }

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
    const float* scales_ptr = scales_f.const_data_ptr<float>();
    const int32_t* zeros_ptr = zeros_i32.const_data_ptr<int32_t>();
    const int32_t* gidx_ptr = g_idx_i32.const_data_ptr<int32_t>();
    int32_t* qweight_ptr = qweight.data_ptr<int32_t>();

    at::Tensor scale_zeros = zeros_i32.to(at::kFloat) * scales_f;
    const float* scale_zeros_ptr = scale_zeros.const_data_ptr<float>();

    const int64_t out_stride = in_features;
    const int64_t scales_stride = out_features;

    NumThreadsGuard guard(threads_eff);

    if (bits == 2 || bits == 4 || bits == 8) {
        switch (bits) {
            case 2:
                dispatch_pack_qweight<2, 16>(
                    weight_ptr,
                    weight_is_bf16,
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
                    threads_eff,
                    block_in);
                break;
            case 4:
                dispatch_pack_qweight<4, 8>(
                    weight_ptr,
                    weight_is_bf16,
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
                    threads_eff,
                    block_in);
                break;
            case 8:
                dispatch_pack_qweight<8, 4>(
                    weight_ptr,
                    weight_is_bf16,
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
                    threads_eff,
                    block_in);
                break;
        }
    } else if (bits == 3) {
        dispatch_pack_qweight_3bit(
            weight_ptr,
            weight_is_bf16,
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
            threads_eff,
            block_in);
    } else {
        TORCH_CHECK(false, "Unsupported bits value", bits);
    }

    const int64_t zeros_stride = zeros_i32.size(1);

    int64_t qzeros_cols = (out_features / word_bits) * bits;
    at::Tensor qzeros = at::zeros({groups, qzeros_cols}, q_options);
    int32_t* qzeros_ptr = qzeros.data_ptr<int32_t>();

    pack_qzeros(zeros_ptr, zeros_stride, qzeros_ptr, qzeros_cols, groups, out_features, static_cast<int>(bits), threads_eff);

    return {qweight, qzeros};
}

namespace {

// Scalar fallback for packing one row/group of AWQ integer weights.
inline void pack_awq_word_scalar(
    const int32_t* src,
    int64_t src_stride,
    int32_t* dst,
    int64_t out_packs,
    int bits,
    int pack_factor,
    int max_q,
    const int* order) {
    for (int64_t p = 0; p < out_packs; ++p) {
        int32_t packed = 0;
        for (int k = 0; k < pack_factor; ++k) {
            const int64_t o = p * pack_factor + order[k];
            int32_t v = src[o * src_stride];
            v = std::max<int32_t>(0, std::min<int32_t>(v, max_q));
            packed |= (v & max_q) << (bits * k);
        }
        dst[p] = packed;
    }
}

#if PACK_BLOCK_CPU_X86
// AVX-512 path for AWQ row packing.  Processes 16 output packs per iteration
// using a strided 32-bit gather so each 512-bit lane holds the same nibble
// position across 16 consecutive 32-bit words.  The gathered values are
// masked, shifted by the AWQ interleave offsets, and summed (which is safe
// because the shifted nibble masks never overlap).
__attribute__((target("avx512f")))
inline void pack_awq_row_avx512(
    const int32_t* src,
    int32_t* dst,
    int64_t out_packs,
    int bits,
    int pack_factor,
    int max_q,
    const int* order_inv) {
    const __m512i v_max_q = _mm512_set1_epi32(max_q);

    // base_idx[j] = j * pack_factor  (columns to read for lane j)
    __m512i base_idx;
    switch (bits) {
        case 2:
            base_idx = _mm512_setr_epi32(
                0, 16, 32, 48, 64, 80, 96, 112,
                128, 144, 160, 176, 192, 208, 224, 240);
            break;
        case 4:
            base_idx = _mm512_setr_epi32(
                0, 8, 16, 24, 32, 40, 48, 56,
                64, 72, 80, 88, 96, 104, 112, 120);
            break;
        case 8:
            base_idx = _mm512_setr_epi32(
                0, 4, 8, 12, 16, 20, 24, 28,
                32, 36, 40, 44, 48, 52, 56, 60);
            break;
        default:
            return;
    }

    const int64_t full = (out_packs / 16) * 16;
    for (int64_t p = 0; p < full; p += 16) {
        __m512i acc = _mm512_setzero_si512();
        const int32_t block_start = static_cast<int32_t>(p * pack_factor);
        for (int k = 0; k < pack_factor; ++k) {
            __m512i idx = _mm512_add_epi32(
                _mm512_set1_epi32(block_start + k), base_idx);
            __m512i v = _mm512_i32gather_epi32(idx, src, 4);
            v = _mm512_and_si512(v, v_max_q);
            const int shift = bits * order_inv[k];
            acc = _mm512_add_epi32(acc, _mm512_slli_epi32(v, shift));
        }
        _mm512_storeu_si512(dst + p, acc);
    }
}
#endif

} // namespace

// AWQ packing helper. Takes pre-quantized integer tensors and packs them into the
// interleaved AWQ layout (e.g. 4-bit order 0,2,4,6,1,3,5,7 per 32-bit word).
std::tuple<at::Tensor, at::Tensor> pack_awq_cpu(
    const at::Tensor& intweight,
    const at::Tensor& zeros,
    int64_t bits,
    int64_t threads) {
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
    int order_inv[16];
    for (int i = 0; i < pack_factor; ++i) {
        order[i] = i;
        order_inv[i] = i;
    }
    if (bits == 4) {
        const int awq_order_4[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        for (int i = 0; i < 8; ++i) {
            order[i] = awq_order_4[i];
            order_inv[awq_order_4[i]] = i;
        }
    }

    const bool use_avx512 =
#if PACK_BLOCK_CPU_X86
        cpu_supports_avx512() &&
        (out_features <= std::numeric_limits<int32_t>::max() - 16 * pack_factor) &&
        out_packs >= 16;
#else
        false;
#endif

    const int64_t awq_threads = clamped_threads(threads, in_features, out_features);
    NumThreadsGuard awq_guard(awq_threads);

    const int64_t grain_in = std::max<int64_t>(1, in_features / awq_threads);
    at::parallel_for(0, in_features, grain_in, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            const int32_t* src = iw_ptr + i * iw_stride_in;
            int32_t* dst = qw_ptr + i * out_packs;
#if PACK_BLOCK_CPU_X86
            if (use_avx512) {
                pack_awq_row_avx512(src, dst, out_packs, bits, pack_factor, max_q, order_inv);
                const int64_t full = (out_packs / 16) * 16;
                if (full < out_packs) {
                    pack_awq_word_scalar(
                        src + full * pack_factor,
                        iw_stride_out,
                        dst + full,
                        out_packs - full,
                        bits,
                        pack_factor,
                        max_q,
                        order);
                }
            } else
#endif
            {
                pack_awq_word_scalar(src, iw_stride_out, dst, out_packs, bits, pack_factor, max_q, order);
            }
        }
    });

    const int64_t grain_groups = std::max<int64_t>(1, groups / awq_threads);
    at::parallel_for(0, groups, grain_groups, [&](int64_t start, int64_t end) {
        for (int64_t g = start; g < end; ++g) {
            const int32_t* src = z_ptr + g * z_stride_group;
            int32_t* dst = qz_ptr + g * out_packs;
#if PACK_BLOCK_CPU_X86
            if (use_avx512) {
                pack_awq_row_avx512(src, dst, out_packs, bits, pack_factor, max_q, order_inv);
                const int64_t full = (out_packs / 16) * 16;
                if (full < out_packs) {
                    pack_awq_word_scalar(
                        src + full * pack_factor,
                        z_stride_out,
                        dst + full,
                        out_packs - full,
                        bits,
                        pack_factor,
                        max_q,
                        order);
                }
            } else
#endif
            {
                pack_awq_word_scalar(src, z_stride_out, dst, out_packs, bits, pack_factor, max_q, order);
            }
        }
    });

    return {qweight, qzeros};
}

// QQQ-style nibble packing.  Takes an int32 tensor where each value holds a
// 4-bit code in the low nibble and packs 8 consecutive columns into one 32-bit
// word (natural order).  This is used for the final Marlin bit-pack step.
at::Tensor pack_qqq_cpu(const at::Tensor& int4_matrix, int64_t bits, int64_t threads) {
    TORCH_CHECK(int4_matrix.device().is_cpu(), "pack_qqq_cpu: int4_matrix must reside on CPU");
    TORCH_CHECK(bits == 4, "pack_qqq_cpu: only 4-bit packing is currently supported");

    const int pack_factor = 32 / static_cast<int>(bits);  // 8 for 4-bit
    const int max_q = (1 << bits) - 1;

    const int64_t rows = int4_matrix.size(0);
    const int64_t cols = int4_matrix.size(1);
    TORCH_CHECK(cols % pack_factor == 0, "pack_qqq_cpu: cols must be divisible by pack_factor");

    at::Tensor int4_i32 = int4_matrix.contiguous().to(at::kInt);

    const int64_t out_cols = cols / pack_factor;
    auto q_options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    at::Tensor q = at::empty({rows, out_cols}, q_options);

    const int32_t* src_ptr = int4_i32.const_data_ptr<int32_t>();
    int32_t* dst_ptr = q.data_ptr<int32_t>();

    const int order[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int order_inv[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    const bool use_avx512 =
#if PACK_BLOCK_CPU_X86
        cpu_supports_avx512() &&
        (cols <= std::numeric_limits<int32_t>::max() - 16 * pack_factor) &&
        out_cols >= 16;
#else
        false;
#endif

    const int64_t qqq_threads = clamped_threads(threads, rows, cols);
    NumThreadsGuard qqq_guard(qqq_threads);

    const int64_t grain_rows = std::max<int64_t>(1, rows / qqq_threads);
    at::parallel_for(0, rows, grain_rows, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            const int32_t* src = src_ptr + i * cols;
            int32_t* dst = dst_ptr + i * out_cols;
#if PACK_BLOCK_CPU_X86
            if (use_avx512) {
                pack_awq_row_avx512(src, dst, out_cols, bits, pack_factor, max_q, order_inv);
                const int64_t full = (out_cols / 16) * 16;
                if (full < out_cols) {
                    pack_awq_word_scalar(
                        src + full * pack_factor,
                        1,
                        dst + full,
                        out_cols - full,
                        bits,
                        pack_factor,
                        max_q,
                        order);
                }
            } else
#endif
            {
                pack_awq_word_scalar(src, 1, dst, out_cols, bits, pack_factor, max_q, order);
            }
        }
    });

    return q;
}

// Round-to-nearest-even for float32.  This matches torch.round and the
// Triton _round_half_to_even helper used by the GPU scale-search kernels.
inline float round_half_to_even(float x) {
    if (!std::isfinite(x)) {
        return x;
    }
    float s = (x >= 0.0f) ? 1.0f : -1.0f;
    float ax = x * s;
    float y = std::floor(ax);
    float frac = ax - y;
    bool is_odd = (static_cast<int32_t>(y) & 1) != 0;
    bool inc = (frac > 0.5f) || (frac == 0.5f && is_odd);
    return s * (y + (inc ? 1.0f : 0.0f));
}

// Dequantize a single weight with the same formula used by the eager GPTQ
// block loop (and the native CUDA block kernel). This keeps the CPU block kernel
// bit-identical to the existing PyTorch reference.
inline float quantize_gptq(
    float w,
    float scale,
    float zero,
    float maxq,
    bool groupwise) {
    float ratio = w / scale;
    // Clamp before rounding so the integer floor/parity logic stays well within
    // the range of a 32-bit int.  The exact clamp bounds are one wider than
    // the final quantization range, which preserves tie-breaking at the edges.
    float ratio_min;
    float ratio_max;
    if (groupwise) {
        ratio_min = -maxq - 1.0f;
        ratio_max = maxq + 1.0f;
    } else {
        ratio_min = -zero - 1.0f;
        ratio_max = maxq - zero + 1.0f;
    }
    if (ratio < ratio_min) {
        ratio = ratio_min;
    }
    if (ratio > ratio_max) {
        ratio = ratio_max;
    }
    float rounded = round_half_to_even(ratio);
    if (groupwise) {
        if (rounded < -maxq) {
            rounded = -maxq;
        }
        if (rounded > maxq) {
            rounded = maxq;
        }
        return rounded * scale;
    }
    float q_int = rounded + zero;
    if (q_int < 0.0f) {
        q_int = 0.0f;
    }
    if (q_int > maxq) {
        q_int = maxq;
    }
    return scale * (q_int - zero);
}

// Batched activation / MSE scale search for CPU.  This mirrors the Triton
// fast path used on CUDA: it scans all shrink candidates in C++, maintains a
// small top-k of the best candidates per (row, group), and returns their
// indices/scale/zero so the Python caller can recompute the exact loss with
// the same arithmetic used by the eager fallback.
std::tuple<at::Tensor, at::Tensor, at::Tensor> find_params_batched_cpu_topk(
    const at::Tensor& x,
    const at::Tensor& xmin,
    const at::Tensor& xmax,
    const c10::optional<at::Tensor>& importance_opt,
    int64_t grid,
    double maxshrink,
    int64_t maxq,
    bool sym,
    bool groupwise,
    double mse) {
    TORCH_CHECK(x.device().is_cpu(), "find_params_batched_cpu: x must reside on CPU");
    TORCH_CHECK(x.dim() == 3, "find_params_batched_cpu: x must be 3D");
    TORCH_CHECK(xmin.dim() == 2, "find_params_batched_cpu: xmin must be 2D");
    TORCH_CHECK(xmax.dim() == 2, "find_params_batched_cpu: xmax must be 2D");
    TORCH_CHECK(
        xmin.sizes() == xmax.sizes(),
        "find_params_batched_cpu: xmin and xmax must have the same shape");

    at::Tensor x_f = x.contiguous().to(at::kFloat);
    at::Tensor xmin_f = xmin.contiguous().to(at::kFloat);
    at::Tensor xmax_f = xmax.contiguous().to(at::kFloat);

    const int64_t rows = x_f.size(0);
    const int64_t num_groups = x_f.size(1);
    const int64_t group_size = x_f.size(2);

    TORCH_CHECK(xmin_f.size(0) == rows, "find_params_batched_cpu: xmin rows mismatch");
    TORCH_CHECK(
        xmin_f.size(1) == num_groups, "find_params_batched_cpu: xmin groups mismatch");

    bool has_importance = false;
    at::Tensor importance_f;
    if (importance_opt.has_value()) {
        importance_f = importance_opt->contiguous().to(at::kFloat);
        TORCH_CHECK(
            importance_f.dim() == 2,
            "find_params_batched_cpu: importance must be 2D");
        TORCH_CHECK(
            importance_f.size(0) == num_groups,
            "find_params_batched_cpu: importance groups mismatch");
        TORCH_CHECK(
            importance_f.size(1) == group_size,
            "find_params_batched_cpu: importance group_size mismatch");
        has_importance = true;
    }

    int64_t candidate_count = static_cast<int64_t>(maxshrink * static_cast<double>(grid));
    if (candidate_count < 1) {
        candidate_count = 1;
    }
    constexpr int64_t TOPK_CAPACITY = 8;
    const int64_t topk = std::min<int64_t>(candidate_count, TOPK_CAPACITY);

    at::Tensor topk_idx = at::zeros(
        {rows, num_groups, topk},
        at::TensorOptions().dtype(at::kLong).device(x_f.device()));
    at::Tensor topk_scale = at::empty(
        {rows, num_groups, topk},
        at::TensorOptions().dtype(at::kFloat).device(x_f.device()));
    at::Tensor topk_zero = at::empty(
        {rows, num_groups, topk},
        at::TensorOptions().dtype(at::kFloat).device(x_f.device()));

    const float* x_ptr = x_f.const_data_ptr<float>();
    const float* xmin_ptr = xmin_f.const_data_ptr<float>();
    const float* xmax_ptr = xmax_f.const_data_ptr<float>();
    const float* imp_ptr = has_importance ? importance_f.const_data_ptr<float>() : nullptr;
    int64_t* idx_ptr = topk_idx.data_ptr<int64_t>();
    float* scale_ptr = topk_scale.data_ptr<float>();
    float* zero_ptr = topk_zero.data_ptr<float>();

    const float maxq_f = static_cast<float>(maxq);
    const float grid_f = static_cast<float>(grid);
    const float mse_f = static_cast<float>(mse);
    const bool mse_is_two = std::abs(mse_f - 2.0f) < 1e-7f;
    const float const_zero_sym = (maxq_f + 1.0f) / 2.0f;
    const int64_t x_stride_r = num_groups * group_size;
    const int64_t x_stride_g = group_size;

    at::parallel_for(0, rows * num_groups, 64, [&](int64_t start, int64_t end) {
        std::array<int64_t, TOPK_CAPACITY> best_idx{};
        std::array<float, TOPK_CAPACITY> best_loss{};

        for (int64_t rg = start; rg < end; ++rg) {
            const int64_t row = rg / num_groups;
            const int64_t g = rg % num_groups;
            const float* x_rowg = x_ptr + row * x_stride_r + g * x_stride_g;
            const float xmin_val = xmin_ptr[rg];
            const float xmax_val = xmax_ptr[rg];
            const float* imp_g = has_importance ? (imp_ptr + g * group_size) : nullptr;

            for (int64_t ti = 0; ti < topk; ++ti) {
                best_loss[ti] = std::numeric_limits<float>::infinity();
                best_idx[ti] = 0;
            }

            for (int64_t c = 0; c < candidate_count; ++c) {
                const float p = 1.0f - static_cast<float>(c) / grid_f;
                const float xmin_p = p * xmin_val;
                const float xmax_p = p * xmax_val;
                float scale_c;
                float zero_c;
                if (groupwise) {
                    scale_c = xmax_p / maxq_f;
                    zero_c = 0.0f;
                } else {
                    scale_c = (xmax_p - xmin_p) / maxq_f;
                    zero_c = sym ? const_zero_sym : round_half_to_even(-xmin_p / scale_c);
                }

                float loss = 0.0f;
                for (int64_t j = 0; j < group_size; ++j) {
                    const float q_val = quantize_gptq(x_rowg[j], scale_c, zero_c, maxq_f, groupwise);
                    const float error = q_val - x_rowg[j];
                    float w;
                    if (has_importance) {
                        // Activation scale search always uses squared error weighted
                        // by the per-element importance (the mse field is ignored).
                        w = error * error * imp_g[j];
                    } else if (mse_is_two) {
                        w = error * error;
                    } else {
                        w = std::pow(std::abs(error), mse_f);
                    }
                    loss += w;
                }

                if (loss < best_loss[topk - 1]) {
                    int64_t pos = topk - 1;
                    best_loss[pos] = loss;
                    best_idx[pos] = c;
                    while (pos > 0 && best_loss[pos] < best_loss[pos - 1]) {
                        std::swap(best_loss[pos], best_loss[pos - 1]);
                        std::swap(best_idx[pos], best_idx[pos - 1]);
                        --pos;
                    }
                }
            }

            int64_t* idx_out = idx_ptr + rg * topk;
            float* scale_out = scale_ptr + rg * topk;
            float* zero_out = zero_ptr + rg * topk;
            for (int64_t ti = 0; ti < topk; ++ti) {
                const int64_t c = best_idx[ti];
                idx_out[ti] = c;
                const float p = 1.0f - static_cast<float>(c) / grid_f;
                const float xmin_p = p * xmin_val;
                const float xmax_p = p * xmax_val;
                if (groupwise) {
                    scale_out[ti] = xmax_p / maxq_f;
                    zero_out[ti] = 0.0f;
                } else {
                    scale_out[ti] = (xmax_p - xmin_p) / maxq_f;
                    zero_out[ti] = sym ? const_zero_sym : round_half_to_even(-xmin_p / scale_out[ti]);
                }
            }
        }
    });

    return {topk_idx, topk_scale, topk_zero};
}

// Fused GPTQ block step for CPU.  Processes one contiguous column block for all
// output rows in parallel, fusing the per-column quantize/diff/error/update
// loop that the eager path issues as many small torch.addr/addmm calls.
std::tuple<at::Tensor, at::Tensor> gptq_block_cpu(
    const at::Tensor& W1,
    const at::Tensor& Hinv1,
    const at::Tensor& scale,
    const at::Tensor& zero,
    int64_t maxq,
    int64_t group_size,
    bool groupwise) {
    TORCH_CHECK(W1.device().is_cpu(), "gptq_block_cpu: W1 must reside on CPU");
    TORCH_CHECK(Hinv1.device().is_cpu(), "gptq_block_cpu: Hinv1 must reside on CPU");
    TORCH_CHECK(scale.device().is_cpu(), "gptq_block_cpu: scale must reside on CPU");
    TORCH_CHECK(zero.device().is_cpu(), "gptq_block_cpu: zero must reside on CPU");
    TORCH_CHECK(W1.dim() == 2, "gptq_block_cpu: W1 must be 2D");
    TORCH_CHECK(Hinv1.dim() == 2, "gptq_block_cpu: Hinv1 must be 2D");
    TORCH_CHECK(W1.size(1) == Hinv1.size(0), "gptq_block_cpu: W1/Hinv1 size mismatch");
    TORCH_CHECK(Hinv1.size(0) == Hinv1.size(1), "gptq_block_cpu: Hinv1 must be square");

    at::Tensor W = W1.to(at::kFloat).clone();
    at::Tensor H = Hinv1.to(at::kFloat);
    at::Tensor s = scale.to(at::kFloat);
    at::Tensor z = zero.to(at::kFloat);

    const int64_t rows = W.size(0);
    const int64_t count = W.size(1);
    TORCH_CHECK(group_size > 0, "gptq_block_cpu: group_size must be positive");
    const int64_t num_full_groups = count / group_size;
    const int64_t tail = count - num_full_groups * group_size;
    const int64_t groups = num_full_groups + (tail > 0 ? 1 : 0);
    TORCH_CHECK(
        s.sizes() == at::IntArrayRef({rows, groups}),
        "gptq_block_cpu: scale shape must be (",
        rows,
        ", ",
        groups,
        "), got ",
        s.sizes());
    TORCH_CHECK(z.sizes() == s.sizes(), "gptq_block_cpu: zero shape must match scale");

    at::Tensor Q = at::empty_like(W);
    at::Tensor Err = at::empty_like(W);
    const float maxq_f = static_cast<float>(maxq);

    for (int64_t i = 0; i < count; ++i) {
        int64_t g = i / group_size;
        if (g >= num_full_groups) {
            g = num_full_groups;
        }
        at::Tensor sc = s.select(1, g).unsqueeze(1);
        at::Tensor zv = z.select(1, g).unsqueeze(1);
        at::Tensor w = W.select(1, i).unsqueeze(1);

        at::Tensor q;
        if (groupwise) {
            q = at::clamp(at::round(w / sc), -maxq_f, maxq_f).mul(sc);
        } else {
            q = at::clamp(at::round(w / sc) + zv, 0.0f, maxq_f).sub(zv).mul(sc);
        }

        Q.select(1, i).copy_(q.squeeze(1));

        at::Tensor d = H.select(0, i).select(0, i);
        at::Tensor err = (w - q) / d;
        Err.select(1, i).copy_(err.squeeze(1));

        at::Tensor tail_view = W.narrow(1, i, count - i);
        at::Tensor hrow = H.select(0, i).narrow(0, i, count - i);
        // In-place torch.addr on the non-contiguous trailing slice replicates the
        // eager serial loop exactly, including the diagonal update that sets the
        // current column to q.
        tail_view.addr_(err.view(-1), hrow, c10::Scalar(1.0), c10::Scalar(-1.0));
    }

    return {Q, Err};
}

// Hessian X^T X accumulation.  This intentionally dispatches to ATen's
// addmm_out, which on CPU uses MKL/OpenBLAS with AVX-512/AVX2 vectorization,
// so the result is bit-exact with torch.addmm while removing Python wrapper
// overhead from the GPU-OOM CPU fallback path.
at::Tensor hessian_xtx_cpu(
    const at::Tensor& X,
    const c10::optional<at::Tensor>& out_opt,
    double beta,
    double alpha) {
    TORCH_CHECK(X.device().is_cpu(), "hessian_xtx_cpu: X must reside on CPU");
    TORCH_CHECK(X.dim() == 2, "hessian_xtx_cpu: X must be 2D");

    const int64_t rows = X.size(0);
    const int64_t cols = X.size(1);

    at::Tensor out;
    if (out_opt.has_value()) {
        out = *out_opt;
        TORCH_CHECK(
            out.sizes() == at::IntArrayRef({cols, cols}),
            "hessian_xtx_cpu: out shape must be (",
            cols,
            ", ",
            cols,
            "), got ",
            out.sizes());
        TORCH_CHECK(out.device().is_cpu(), "hessian_xtx_cpu: out must reside on CPU");
        TORCH_CHECK(out.scalar_type() == at::kFloat, "hessian_xtx_cpu: out must be float32");
    } else {
        out = at::zeros({cols, cols}, X.options().dtype(at::kFloat));
    }

    if (rows == 0 || alpha == 0.0) {
        if (beta == 0.0) {
            out.zero_();
        } else {
            out.mul_(beta);
        }
        return out;
    }

    at::Tensor X_f = X.contiguous().to(at::kFloat);

    if (out_opt.has_value()) {
        // Match the Python `out.addmm_(mat1.T, mat1, beta=..., alpha=...)`
        // path bit-exactly so chunked Hessian accumulation stays stable.
        at::addmm_out(out, out, X_f.t(), X_f, beta, alpha);
        return out;
    }

    // No output supplied: mirror `torch.matmul(mat32.T, mat32)` and let ATen
    // pick the best MKL/OpenBLAS path (including syrk when recognized).
    at::Tensor prod = at::matmul(X_f.t(), X_f);
    if (alpha != 1.0) {
        prod.mul_(alpha);
    }
    return prod;
}

// Hessian inverse via Cholesky.  Clones H, adds diag_delta to the diagonal,
// attempts Cholesky, and returns the upper Cholesky factor of H^{-1} on
// success.  Bit-exact with the Python _hessian_inverse_try_cholesky +
// _hessian_inverse_factor sequence because it uses the same ATen/LAPACK
// kernels (MKL/LAPACK with AVX-512/AVX2 SIMD on x86).
std::tuple<at::Tensor, at::Tensor> hessian_inverse_cholesky_cpu(
    const at::Tensor& H,
    const at::Tensor& diag_delta) {
    TORCH_CHECK(H.device().is_cpu(), "hessian_inverse_cholesky_cpu: H must reside on CPU");
    TORCH_CHECK(H.dim() == 2 && H.size(0) == H.size(1), "hessian_inverse_cholesky_cpu: H must be square");
    TORCH_CHECK(H.scalar_type() == at::kFloat, "hessian_inverse_cholesky_cpu: H must be float32");

    at::Tensor H_eff = H.clone();
    if (diag_delta.defined() && diag_delta.numel() > 0) {
        H_eff.diagonal().add_(diag_delta);
    }

    auto cholesky_result = at::linalg_cholesky_ex(H_eff, false);
    at::Tensor L = std::get<0>(cholesky_result);
    at::Tensor info = std::get<1>(cholesky_result);

    at::Tensor success = (info == 0).view({});
    at::Tensor Hinv = at::empty({0}, H.options());

    if (success.item<bool>()) {
        Hinv = at::linalg_cholesky(at::cholesky_inverse(L), true).contiguous();
    }

    return {Hinv, success};
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
        "pack_awq_cpu(Tensor intweight, Tensor zeros, int bits, int threads=-1) -> (Tensor, Tensor)"
    );
    m.impl(
        "pack_awq_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::pack_awq_cpu)
    );
    m.def(
        "pack_qqq_cpu(Tensor int4_matrix, int bits, int threads=-1) -> Tensor"
    );
    m.impl(
        "pack_qqq_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::pack_qqq_cpu)
    );
    m.def(
        "hessian_xtx_cpu(Tensor X, Tensor? out=None, float beta=0, float alpha=1) -> Tensor"
    );
    m.impl(
        "hessian_xtx_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::hessian_xtx_cpu)
    );
    m.def(
        "hessian_inverse_cholesky_cpu(Tensor H, Tensor diag_delta) -> (Tensor, Tensor)"
    );
    m.impl(
        "hessian_inverse_cholesky_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::hessian_inverse_cholesky_cpu)
    );
    m.def(
        "find_params_batched_cpu_topk(Tensor x, Tensor xmin, Tensor xmax, Tensor? importance, int grid, float maxshrink, int maxq, bool sym, bool groupwise, float mse) -> (Tensor, Tensor, Tensor)"
    );
    m.impl(
        "find_params_batched_cpu_topk",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::find_params_batched_cpu_topk)
    );
    m.def(
        "gptq_block_cpu(Tensor W1, Tensor Hinv1, Tensor scale, Tensor zero, int maxq, int group_size, bool groupwise) -> (Tensor, Tensor)"
    );
    m.impl(
        "gptq_block_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::gptq_block_cpu)
    );
}
