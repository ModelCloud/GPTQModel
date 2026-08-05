// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0
// Contact: qubitium@modelcloud.ai, x.com/qubitium

// Native decode-regime GEMV for the planar (gptq_p) GPTQ format at 3/5/6/7 bits
// on the host CPU.  The kernel is fully fused: each 32-code block loads its
// `bits` packed word rows once per column vector, decodes codes with bitwise
// shifts/masks, scales and subtracts the zero point, and accumulates the dot
// product with the input activations.
//
// CPU-specific optimizations vs the Python planar unpack + torch.matmul path:
//   - No dense fp16 weight matrix materialization; only packed qweight/qzeros
//     are read, which saves memory traffic and cache pressure.
//   - SIMD (AVX2/AVX-512) decode over 8 or 16 output columns at a time.
//   - FMA for the (code * scale - zero*scale) * activation accumulation.
//   - OpenMP parallel-for over output column chunks to use all cores.

#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define PLANAR_GEMV_CPU_X86 1
#else
#define PLANAR_GEMV_CPU_X86 0
#endif

namespace pangolin_cpu {

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


struct Plane {
  int w;
  int off;
  int start;
};

template <int Bits>
constexpr std::array<Plane, 3> plane_info() {
  return {Plane{0, 0, 0}, Plane{0, 0, 0}, Plane{0, 0, 0}};
}

template <>
constexpr std::array<Plane, 3> plane_info<3>() {
  return {Plane{2, 0, 0}, Plane{1, 2, 2}, Plane{0, 0, 0}};
}

template <>
constexpr std::array<Plane, 3> plane_info<5>() {
  return {Plane{4, 0, 0}, Plane{1, 4, 4}, Plane{0, 0, 0}};
}

template <>
constexpr std::array<Plane, 3> plane_info<6>() {
  return {Plane{4, 0, 0}, Plane{2, 4, 4}, Plane{0, 0, 0}};
}

template <>
constexpr std::array<Plane, 3> plane_info<7>() {
  return {Plane{4, 0, 0}, Plane{2, 4, 4}, Plane{1, 6, 6}};
}

// Pre-expanded uint8 layout: each 32-bit word holds 4 consecutive uint8 codes
// in its bytes.  The kernel sees this as bits==8 (8 words per 32-row K-block).
template <>
constexpr std::array<Plane, 3> plane_info<8>() {
  return {Plane{8, 0, 0}, Plane{0, 0, 0}, Plane{0, 0, 0}};
}

template <int Bits>
inline int decode_zero_code(
    const int32_t* qzeros,
    int group,
    int n,
    int num_groups,
    int64_t size_n,
    int64_t qzeros_stride) {
  if (group < 0) {
    group += num_groups;
  }
  const int cb = n / 32;
  const int pos = n % 32;
  const int32_t* base = qzeros + static_cast<int64_t>(group) * qzeros_stride + static_cast<int64_t>(cb) * Bits;
  constexpr auto planes = plane_info<Bits>();
  int code = 0;
  for (int p = 0; p < 3; ++p) {
    const int w = planes[p].w;
    if (w == 0) {
      break;
    }
    const int pack_factor = 32 / w;
    const int word = base[planes[p].start + pos / pack_factor];
    const int part = (word >> (w * (pos % pack_factor))) & ((1 << w) - 1);
    code |= part << planes[p].off;
  }
  return code;
}

#if PLANAR_GEMV_CPU_X86

__attribute__((constructor))
static void pangolin_gemv_cpu_init_features() {
  __builtin_cpu_init();
}

inline bool cpu_supports_avx2() {
  if (env_flag_enabled("GPTQMODEL_PANGOLIN_CPU_DISABLE_AVX2")) {
    return false;
  }
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

inline bool cpu_supports_avx512_core() {
  if (env_flag_enabled("GPTQMODEL_PANGOLIN_CPU_DISABLE_AVX512")) {
    return false;
  }
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512bf16");
}

// AVX2 256-bit vector helpers.
__attribute__((target("avx2,fma"))) inline __m256i load_i_256(const void* p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

__attribute__((target("avx2,fma"))) inline __m256 load_f_256(const float* p) {
  return _mm256_loadu_ps(p);
}

__attribute__((target("avx2,fma"))) inline __m256 load_scale_256(const at::BFloat16* p) {
  __m256i v16 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  __m256i shifted = _mm256_slli_epi32(v16, 16);
  return _mm256_castsi256_ps(shifted);
}

__attribute__((target("avx2,fma"))) inline __m256i set1_i_256(int v) {
  return _mm256_set1_epi32(v);
}

__attribute__((target("avx2,fma"))) inline __m256 set1_f_256(float v) {
  return _mm256_set1_ps(v);
}

__attribute__((target("avx2,fma"))) inline __m256i srlv_256(__m256i a, __m256i b) {
  return _mm256_srlv_epi32(a, b);
}

__attribute__((target("avx2,fma"))) inline __m256i slli_256(__m256i a, int imm) {
  return _mm256_slli_epi32(a, imm);
}

__attribute__((target("avx2,fma"))) inline __m256i and_256(__m256i a, __m256i b) {
  return _mm256_and_si256(a, b);
}

__attribute__((target("avx2,fma"))) inline __m256i or_256(__m256i a, __m256i b) {
  return _mm256_or_si256(a, b);
}

__attribute__((target("avx2,fma"))) inline __m256 cvt_i_f_256(__m256i a) {
  return _mm256_cvtepi32_ps(a);
}

__attribute__((target("avx2,fma"))) inline __m256 fmsub_256(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmsub_ps(a, b, c);
}

__attribute__((target("avx2,fma"))) inline __m256 fmadd_256(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}

__attribute__((target("avx2,fma"))) inline void store_f_avx2(float* p, __m256 a) {
  _mm256_storeu_ps(p, a);
}

// AVX-512 512-bit vector helpers.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512i load_i_512(const void* p) {
  return _mm512_loadu_si512(p);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512 load_f_512(const float* p) {
  return _mm512_loadu_ps(p);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) inline __m512 load_scale_512(const at::BFloat16* p) {
  __m512i v16 = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
  __m512i shifted = _mm512_slli_epi32(v16, 16);
  return _mm512_castsi512_ps(shifted);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512i set1_i_512(int v) {
  return _mm512_set1_epi32(v);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512 set1_f_512(float v) {
  return _mm512_set1_ps(v);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512i srlv_512(__m512i a, __m512i b) {
  return _mm512_srlv_epi32(a, b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512i slli_512(__m512i a, int imm) {
  return _mm512_slli_epi32(a, imm);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512i and_512(__m512i a, __m512i b) {
  return _mm512_and_si512(a, b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512i or_512(__m512i a, __m512i b) {
  return _mm512_or_si512(a, b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512 cvt_i_f_512(__m512i a) {
  return _mm512_cvtepi32_ps(a);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512 fmsub_512(__m512 a, __m512 b, __m512 c) {
  return _mm512_fmsub_ps(a, b, c);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512 fmadd_512(__m512 a, __m512 b, __m512 c) {
  return _mm512_fmadd_ps(a, b, c);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline void store_f_avx512(float* p, __m512 a) {
  _mm512_storeu_ps(p, a);
}

__attribute__((target("avx2,fma"))) inline __m256i srli_256(__m256i a, int imm) {
  return _mm256_srli_epi32(a, imm);
}

__attribute__((target("avx2,fma"))) inline __m256 add_f_256(__m256 a, __m256 b) {
  return _mm256_add_ps(a, b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512i srli_512(__m512i a, int imm) {
  return _mm512_srli_epi32(a, imm);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) inline __m512 add_f_512(__m512 a, __m512 b) {
  return _mm512_add_ps(a, b);
}

__attribute__((target("avx2,fma"))) inline void store_bf16_256(at::BFloat16* out, __m256 a) {
  alignas(64) float tmp[8];
  _mm256_storeu_ps(tmp, a);
  uint16_t* out_u16 = reinterpret_cast<uint16_t*>(out);
  for (int i = 0; i < 8; ++i) {
    out_u16[i] = static_cast<uint16_t>(at::BFloat16(tmp[i]).x);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16"))) inline void store_bf16_512(at::BFloat16* out, __m512 a) {
  __m256bh bh = _mm512_cvtneps_pbh(a);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), (__m256i)bh);
}

#if PLANAR_GEMV_CPU_X86

// Pre-computed shift vectors for extracting w-bit fields from a 32-bit packed
// word when decoding zero points.  Only the AVX-512 (16-lane) path is shown;
// the scalar/AVX2 paths keep the existing decode_zero_code helper.
alignas(64) static const int32_t k_zero_shift_w4_512[16] = {
    0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28};
alignas(64) static const int32_t k_zero_shift_w2_512[16] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) static const int32_t k_zero_shift_w1_0_512[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
alignas(64) static const int32_t k_zero_shift_w1_16_512[16] = {
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m512i decode_zero_code_vec_512(
    const int32_t* qzeros,
    int group,
    int64_t col0,
    int64_t N,
    int num_groups,
    int64_t qzeros_stride) {
  if (group < 0) {
    group += num_groups;
  }
  const int cb = static_cast<int>(col0 / 32);
  const int pos0 = static_cast<int>(col0 % 32);
  const int32_t* base = qzeros + static_cast<int64_t>(group) * qzeros_stride + static_cast<int64_t>(cb) * Bits;
  constexpr auto planes = plane_info<Bits>();
  __m512i code = _mm512_setzero_si512();
  for (int p = 0; p < 3; ++p) {
    const int w = planes[p].w;
    if (w == 0) {
      break;
    }
    const int pack_factor = 32 / w;
    const int word_offset = pos0 / pack_factor;
    const int32_t* word_ptr = base + planes[p].start + word_offset;
    __m512i word_vec;
    if (w == 4) {
      // 16 columns span two packed words; broadcast each word to its 8 lanes and blend.
      const __m512i word0 = _mm512_set1_epi32(word_ptr[0]);
      const __m512i word1 = _mm512_set1_epi32(word_ptr[1]);
      word_vec = _mm512_mask_mov_epi32(word0, static_cast<__mmask16>(0xFF00), word1);
    } else {
      word_vec = _mm512_set1_epi32(word_ptr[0]);
    }
    __m512i shift_vec;
    if (w == 4) {
      shift_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(k_zero_shift_w4_512));
    } else if (w == 2) {
      shift_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(k_zero_shift_w2_512));
    } else {
      shift_vec = (pos0 == 0)
          ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(k_zero_shift_w1_0_512))
          : _mm512_loadu_si512(reinterpret_cast<const __m512i*>(k_zero_shift_w1_16_512));
    }
    const __m512i mask = _mm512_set1_epi32((1 << w) - 1);
    const __m512i part = _mm512_and_si512(_mm512_srlv_epi32(word_vec, shift_vec), mask);
    const __m512i off_vec = _mm512_set1_epi32(planes[p].off);
    code = _mm512_or_si512(code, _mm512_sllv_epi32(part, off_vec));
  }
  return code;
}

template <int Bits>
__attribute__((target("avx512f,avx512bw,avx512vl"), noinline))
static void write_zero_scale_block_avx512(
    const int32_t* qzeros_ptr,
    const at::BFloat16* scale_b,
    float* zero_scale_f,
    int group,
    int64_t col0,
    int64_t N,
    int num_groups,
    int64_t qzeros_stride) {
  const __m512i zcode = decode_zero_code_vec_512<Bits>(
      qzeros_ptr, group, col0, N, num_groups, qzeros_stride);
  const __m512 zcode_f = _mm512_cvtepi32_ps(zcode);
  const __m512 scale_vec = _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(scale_b + static_cast<int64_t>(group) * N + col0))),
      16));
  const __m512 zscale = _mm512_mul_ps(zcode_f, scale_vec);
  _mm512_storeu_ps(zero_scale_f + static_cast<int64_t>(group) * N + col0, zscale);
}

template <int Bits>
static void compute_zero_scale_avx512(
    const int32_t* qzeros_ptr,
    const at::BFloat16* scale_b,
    float* zero_scale_f,
    int64_t N,
    int num_groups,
    int64_t qzeros_stride) {
  const int64_t num_blocks = N / 16;
  const int64_t grain = std::max<int64_t>(1, num_groups * num_blocks / at::get_num_threads());
  at::parallel_for(0, num_groups * num_blocks, grain, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int group = static_cast<int>(idx / num_blocks);
      const int block = static_cast<int>(idx % num_blocks);
      const int64_t col0 = static_cast<int64_t>(block) * 16;
      write_zero_scale_block_avx512<Bits>(
          qzeros_ptr, scale_b, zero_scale_f, group, col0, N, num_groups, qzeros_stride);
    }
  });
}

#endif  // PLANAR_GEMV_CPU_X86

// Choose K_UNROLL so the FP32 accumulator array stays inside the AVX-512 ZMM
// budget while still hiding FMA latency.  M==1 can keep 32 independent partial
// sums in flight (the compiler spills a few, but the K-block fully unrolls and
// FMA latency is well hidden); for M>1 register pressure rises with SizeM.
// Bits 7 needs the most qword/decode registers, so for M>1 it gets the most
// conservative unroll; bits 3/5/6 can keep more independent partial sums.
template <int Bits, int SizeM>
constexpr int k_unroll_for() {
  if constexpr (SizeM == 1) {
    // 32 independent partials fully hide FMA latency but can spill ZMMs; 16 is a
    // middle ground that still covers the ~4-6 cycle FMA latency window while
    // leaving registers for the decoded qwords and scale vectors.
    return 16;
  } else if constexpr (SizeM <= 2) {
    return (Bits <= 6) ? 8 : 4;
  } else if constexpr (SizeM <= 4) {
    return (Bits <= 6) ? 4 : 2;
  } else if constexpr (SizeM <= 8) {
    return 2;
  } else {
    return 1;
  }
}

#define DEFINE_GEMV_KERNEL(suffix, target_features, LOAD_I, LOAD_F, LOAD_SCALE, SET1_I, SET1_F, SRLI, SLLI, AND, OR, ADD_F, CVT, FMSUB, FMADD, STORE_BF16, INT_T, FLOAT_T, VLEN) \
  template <int Bits, int SizeM> \
  __attribute__((target(target_features))) \
  void gemv_col_block_##suffix( \
      const float* __restrict__ x_f, \
      const int32_t* __restrict__ qweight, \
      const at::BFloat16* __restrict__ scale_b, \
      const float* __restrict__ zero_scale_f, \
      const int32_t* __restrict__ g_idx, \
      void* __restrict__ out_ptr, \
      bool out_float, \
      int64_t col0, \
      int64_t K, \
      int64_t N, \
      int64_t num_groups, \
      int64_t kb_start, \
      int64_t kb_end) { \
    constexpr auto planes = plane_info<Bits>(); \
    /* K-unrolling breaks the FP32 FMA dependency chain. */ \
    constexpr int K_UNROLL = k_unroll_for<Bits, SizeM>(); \
    FLOAT_T acc[SizeM][K_UNROLL]; \
    for (int m = 0; m < SizeM; ++m) { \
      for (int u = 0; u < K_UNROLL; ++u) { \
        acc[m][u] = SET1_F(0.0f); \
      } \
    } \
    float* out_f = reinterpret_cast<float*>(out_ptr); \
    at::BFloat16* out_b = reinterpret_cast<at::BFloat16*>(out_ptr); \
    const int64_t num_k_blocks = K / 32; \
    const int64_t chunk = col0 / 16; \
    const int lane = static_cast<int>(col0 % 16); \
    int prev_group = -1; \
    FLOAT_T scale_vec = SET1_F(0.0f); \
    FLOAT_T zscale_vec = SET1_F(0.0f); \
    for (int64_t kb = kb_start; kb < kb_end; ++kb) { \
      const int64_t row0 = kb * 32; \
      int group = g_idx[row0]; \
      if (__builtin_expect(group < 0, 0)) { \
        group += static_cast<int>(num_groups); \
      } \
      if (__builtin_expect(group != prev_group, 0)) { \
        const at::BFloat16* scale_ptr = scale_b + static_cast<int64_t>(group) * N + col0; \
        const float* zscale_ptr = zero_scale_f + static_cast<int64_t>(group) * N + col0; \
        scale_vec = LOAD_SCALE(scale_ptr); \
        zscale_vec = LOAD_F(zscale_ptr); \
        prev_group = group; \
      } \
      /* qweight is pre-packed as [N/16, num_k_blocks, Bits, 16] so the Bits \
       * 32-bit words for a fixed col_block are contiguous across all K-blocks. \
       * Load all Bits qwords once per K-block and reuse them across the 32 K \
       * values to avoid re-reading the same cache lines 32 times. */ \
      const int64_t base_offset = ((chunk * num_k_blocks + kb) * Bits) * 16 + lane; \
      const uint32_t* qbase = reinterpret_cast<const uint32_t*>(qweight) + base_offset; \
      INT_T qwords[Bits]; \
      for (int pw = 0; pw < Bits; ++pw) { \
        qwords[pw] = LOAD_I(qbase + pw * 16); \
      } \
      if (__builtin_expect(kb + 1 < kb_end, 1)) { \
        const uint32_t* next_qbase = qbase + Bits * 16; \
        for (int pw = 0; pw < Bits; ++pw) { \
          _mm_prefetch(reinterpret_cast<const char*>(next_qbase + pw * 16), _MM_HINT_T0); \
          _mm_prefetch(reinterpret_cast<const char*>(next_qbase + pw * 16 + 8), _MM_HINT_T0); \
        } \
        int next_group = g_idx[(kb + 1) * 32]; \
        if (__builtin_expect(next_group < 0, 0)) { \
          next_group += static_cast<int>(num_groups); \
        } \
        if (__builtin_expect(next_group != group, 0)) { \
          _mm_prefetch(reinterpret_cast<const char*>(scale_b + static_cast<int64_t>(next_group) * N + col0), _MM_HINT_T0); \
          _mm_prefetch(reinterpret_cast<const char*>(zero_scale_f + static_cast<int64_t>(next_group) * N + col0), _MM_HINT_T0); \
          _mm_prefetch(reinterpret_cast<const char*>(zero_scale_f + static_cast<int64_t>(next_group) * N + col0 + 8), _MM_HINT_T0); \
        } \
        for (int m = 0; m < SizeM; ++m) { \
          _mm_prefetch(reinterpret_cast<const char*>(x_f + static_cast<int64_t>(m) * K + (kb + 1) * 32), _MM_HINT_T0); \
        } \
      } \
      _Pragma("GCC unroll 32") \
      for (int k = 0; k < 32; ++k) { \
        INT_T code = SET1_I(0); \
        if constexpr (planes[0].w != 0) { \
          const int w = planes[0].w; \
          const int start = planes[0].start; \
          const int off = planes[0].off; \
          const int pack_factor = 32 / w; \
          INT_T shifted = SRLI(qwords[start + k / pack_factor], w * (k % pack_factor)); \
          INT_T masked = AND(shifted, SET1_I((1 << w) - 1)); \
          code = OR(code, SLLI(masked, off)); \
        } \
        if constexpr (planes[1].w != 0) { \
          const int w = planes[1].w; \
          const int start = planes[1].start; \
          const int off = planes[1].off; \
          const int pack_factor = 32 / w; \
          INT_T shifted = SRLI(qwords[start + k / pack_factor], w * (k % pack_factor)); \
          INT_T masked = AND(shifted, SET1_I((1 << w) - 1)); \
          code = OR(code, SLLI(masked, off)); \
        } \
        if constexpr (planes[2].w != 0) { \
          const int w = planes[2].w; \
          const int start = planes[2].start; \
          const int off = planes[2].off; \
          const int pack_factor = 32 / w; \
          INT_T shifted = SRLI(qwords[start + k / pack_factor], w * (k % pack_factor)); \
          INT_T masked = AND(shifted, SET1_I((1 << w) - 1)); \
          code = OR(code, SLLI(masked, off)); \
        } \
        FLOAT_T code_f = CVT(code); \
        FLOAT_T weight = FMSUB(code_f, scale_vec, zscale_vec); \
        for (int m = 0; m < SizeM; ++m) { \
          FLOAT_T xval = SET1_F(x_f[m * K + row0 + k]); \
          acc[m][k % K_UNROLL] = FMADD(xval, weight, acc[m][k % K_UNROLL]); \
        } \
      } \
    } \
    for (int m = 0; m < SizeM; ++m) { \
      FLOAT_T sum = acc[m][0]; \
      for (int u = 1; u < K_UNROLL; ++u) { \
        sum = ADD_F(sum, acc[m][u]); \
      } \
      if (out_float) { \
        store_f_##suffix(out_f + static_cast<int64_t>(m) * N + col0, sum); \
      } else { \
        STORE_BF16(out_b + static_cast<int64_t>(m) * N + col0, sum); \
      } \
    } \
  } \
  template <int Bits, int SizeM> \
  __attribute__((target(target_features))) \
  void gemv_kernel_##suffix( \
      const float* __restrict__ x_f, \
      const int32_t* __restrict__ qweight, \
      const at::BFloat16* __restrict__ scale_b, \
      const float* __restrict__ zero_scale_f, \
      const int32_t* __restrict__ g_idx, \
      at::BFloat16* __restrict__ out, \
      int64_t /*M*/, \
      int64_t K, \
      int64_t N, \
      int64_t num_groups) { \
    const int64_t num_k_blocks = K / 32; \
    const int64_t col_chunks = N / VLEN; \
    const int num_threads = at::get_num_threads(); \
    /* If there are few output column chunks, parallelize over K-blocks and \
     * reduce per-task partials. This avoids under-utilization on small-N \
     * layers.  Otherwise parallelize over column chunks as usual. */ \
    const bool k_parallel = (col_chunks < num_threads * 2) && (num_k_blocks > 1); \
    if (k_parallel) { \
      const size_t partial_per_task = static_cast<size_t>(SizeM) * N; \
      const int64_t k_grain = std::max<int64_t>(1, num_k_blocks / num_threads); \
      const int64_t num_tasks = (num_k_blocks + k_grain - 1) / k_grain; \
      std::vector<float> partial(static_cast<size_t>(num_tasks) * partial_per_task, 0.0f); \
      std::atomic<int64_t> task_counter{0}; \
      at::parallel_for(0, num_k_blocks, k_grain, [&](int64_t kb_begin, int64_t kb_end) { \
        const int64_t task_id = task_counter.fetch_add(1); \
        float* p_out = partial.data() + task_id * partial_per_task; \
        for (int64_t col_block = 0; col_block < col_chunks; ++col_block) { \
          gemv_col_block_##suffix<Bits, SizeM>( \
              x_f, qweight, scale_b, zero_scale_f, g_idx, p_out, true, \
              col_block * VLEN, K, N, num_groups, kb_begin, kb_end); \
        } \
      }); \
      for (int64_t col_block = 0; col_block < col_chunks; ++col_block) { \
        const int64_t col0 = col_block * VLEN; \
        for (int m = 0; m < SizeM; ++m) { \
          FLOAT_T sum = SET1_F(0.0f); \
          for (int64_t task_id = 0; task_id < num_tasks; ++task_id) { \
            const float* p = partial.data() + task_id * partial_per_task + \
                             static_cast<int64_t>(m) * N + col0; \
            sum = ADD_F(sum, LOAD_F(p)); \
          } \
          STORE_BF16(out + static_cast<int64_t>(m) * N + col0, sum); \
        } \
      } \
    } else { \
      const int64_t grain = std::max<int64_t>(1, col_chunks / num_threads); \
      at::parallel_for(0, col_chunks, grain, [&](int64_t begin, int64_t end) { \
        for (int64_t col_block = begin; col_block < end; ++col_block) { \
          gemv_col_block_##suffix<Bits, SizeM>( \
              x_f, qweight, scale_b, zero_scale_f, g_idx, out, false, \
              col_block * VLEN, K, N, num_groups, 0, num_k_blocks); \
        } \
      }); \
    } \
  }

DEFINE_GEMV_KERNEL(
    avx2,
    "avx2,fma",
    load_i_256,
    load_f_256,
    load_scale_256,
    set1_i_256,
    set1_f_256,
    srli_256,
    slli_256,
    and_256,
    or_256,
    add_f_256,
    cvt_i_f_256,
    fmsub_256,
    fmadd_256,
    store_bf16_256,
    __m256i,
    __m256,
    8)

DEFINE_GEMV_KERNEL(
    avx512,
    "avx512f,avx512bw,avx512vl,avx512bf16,fma",
    load_i_512,
    load_f_512,
    load_scale_512,
    set1_i_512,
    set1_f_512,
    srli_512,
    slli_512,
    and_512,
    or_512,
    add_f_512,
    cvt_i_f_512,
    fmsub_512,
    fmadd_512,
    store_bf16_512,
    __m512i,
    __m512,
    16)

#endif  // PLANAR_GEMV_CPU_X86

template <int Bits, int SizeM>
void gemv_kernel_scalar(
    const float* __restrict__ x_f,
    const int32_t* __restrict__ qweight,
    const at::BFloat16* __restrict__ scale_b,
    const float* __restrict__ zero_scale_f,
    const int32_t* __restrict__ g_idx,
    at::BFloat16* __restrict__ out,
    int64_t /*M*/,
    int64_t K,
    int64_t N,
    int64_t num_groups) {
  constexpr auto planes = plane_info<Bits>();
  const int64_t num_k_blocks = K / 32;
  const int64_t grain = std::max<int64_t>(1, N / at::get_num_threads());
  at::parallel_for(0, N, grain, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; ++n) {
      float acc[SizeM];
      for (int m = 0; m < SizeM; ++m) {
        acc[m] = 0.0f;
      }
      for (int64_t kb = 0; kb < num_k_blocks; ++kb) {
        const int64_t row0 = kb * 32;
        int group = g_idx[row0];
        if (group < 0) {
          group += static_cast<int>(num_groups);
        }
        const float scale = static_cast<float>(scale_b[static_cast<int64_t>(group) * N + n]);
        const float zscale = zero_scale_f[static_cast<int64_t>(group) * N + n];
        for (int k = 0; k < 32; ++k) {
          int code = 0;
          for (int p = 0; p < 3; ++p) {
            const int w = planes[p].w;
            if (w == 0) {
              break;
            }
            const int pack_factor = 32 / w;
            const int word_idx = planes[p].start + k / pack_factor;
            const int64_t chunk = n / 16;
            const int64_t num_k_blocks = K / 32;
            const int lane = static_cast<int>(n % 16);
            const uint32_t word = static_cast<uint32_t>(
                qweight[((chunk * num_k_blocks + kb) * Bits + word_idx) * 16 + lane]);
            const int part = (word >> (w * (k % pack_factor))) & ((1 << w) - 1);
            code |= part << planes[p].off;
          }
          const float weight = static_cast<float>(code) * scale - zscale;
          for (int m = 0; m < SizeM; ++m) {
            acc[m] += x_f[m * K + row0 + k] * weight;
          }
        }
      }
      for (int m = 0; m < SizeM; ++m) {
        out[m * N + n] = at::BFloat16(acc[m]);
      }
    }
  });
}

template <int Bits, int SizeM>
void run_gemv(
    const float* __restrict__ x_f,
    const int32_t* __restrict__ qweight,
    const at::BFloat16* __restrict__ scale_b,
    const float* __restrict__ zero_scale_f,
    const int32_t* __restrict__ g_idx,
    at::BFloat16* __restrict__ out,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t num_groups) {
#if PLANAR_GEMV_CPU_X86
  // AVX-512 is used for all supported M whenever the feature set and N are
  // compatible. M=32 is split into two M=16 passes in dispatch_size_m, so
  // SizeM here never exceeds 16 and the AVX-512 path stays within 32 ZMM.
  if (cpu_supports_avx512_core() && N % 16 == 0 && SizeM <= 16) {
    gemv_kernel_avx512<Bits, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
    return;
  }
  if (cpu_supports_avx2() && N % 8 == 0) {
    gemv_kernel_avx2<Bits, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
    return;
  }
#endif
  gemv_kernel_scalar<Bits, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
}

template <int SizeM>
static void dispatch_bits(
    int64_t kernel_bits,
    const float* __restrict__ x_f,
    const int32_t* __restrict__ qweight,
    const at::BFloat16* __restrict__ scale_b,
    const float* __restrict__ zero_scale_f,
    const int32_t* __restrict__ g_idx,
    at::BFloat16* __restrict__ out,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t num_groups) {
  switch (kernel_bits) {
    case 3:
      run_gemv<3, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 5:
      run_gemv<5, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 6:
      run_gemv<6, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 7:
      run_gemv<7, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 8:
      // Pre-expanded uint8 layout: one 8-bit code per K value, packed 4 per word.
      run_gemv<8, SizeM>(x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    default:
      TORCH_CHECK(false, "pangolin_gemv_cpu qweight must have bits 3/5/6/7 or 8 (uint8), got ", kernel_bits);
  }
}

static void dispatch_size_m(
    int64_t M,
    int64_t bits,
    const float* __restrict__ x_f,
    const int32_t* __restrict__ qweight,
    const at::BFloat16* __restrict__ scale_b,
    const float* __restrict__ zero_scale_f,
    const int32_t* __restrict__ g_idx,
    at::BFloat16* __restrict__ out,
    int64_t K,
    int64_t N,
    int64_t num_groups) {
  switch (M) {
    case 1:
      dispatch_bits<1>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 2:
      dispatch_bits<2>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 3:
      dispatch_bits<3>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 4:
      dispatch_bits<4>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 5:
      dispatch_bits<5>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 6:
      dispatch_bits<6>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 7:
      dispatch_bits<7>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 8:
      dispatch_bits<8>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 16:
      dispatch_bits<16>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 32:
      // M=32 exceeds the AVX-512 ZMM register budget and spills. Split it into
      // two M=16 passes to keep the working register set within 16 ZMM/YMM.
      dispatch_bits<16>(bits, x_f, qweight, scale_b, zero_scale_f, g_idx, out, 16, K, N, num_groups);
      dispatch_bits<16>(
          bits,
          x_f + static_cast<int64_t>(16) * K,
          qweight,
          scale_b,
          zero_scale_f,
          g_idx,
          out + static_cast<int64_t>(16) * N,
          16,
          K,
          N,
          num_groups);
      break;
    default:
      TORCH_CHECK(false, "pangolin_gemv_cpu supports M in {1..8,16,32}, got ", M);
  }
}

}  // namespace

torch::Tensor pangolin_gemv_cpu(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    int64_t bits) {
  TORCH_CHECK(input.device().is_cpu(), "pangolin_gemv_cpu input must be CPU");
  TORCH_CHECK(
      qweight.device().is_cpu() && scales.device().is_cpu() && qzeros.device().is_cpu() && g_idx.device().is_cpu(),
      "pangolin_gemv_cpu tensors must be on CPU");
  TORCH_CHECK(
      input.device() == qweight.device() && input.device() == scales.device() &&
          input.device() == qzeros.device() && input.device() == g_idx.device(),
      "pangolin_gemv_cpu tensors must be on the same CPU device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "pangolin_gemv_cpu input must be BF16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "pangolin_gemv_cpu qweight must be int32");
  TORCH_CHECK(qzeros.scalar_type() == at::kInt, "pangolin_gemv_cpu qzeros must be int32");
  TORCH_CHECK(g_idx.scalar_type() == at::kInt, "pangolin_gemv_cpu g_idx must be int32");
  TORCH_CHECK(
      scales.scalar_type() == at::kHalf || scales.scalar_type() == at::kBFloat16,
      "pangolin_gemv_cpu scales must be FP16 or BF16");
  TORCH_CHECK(
      input.dim() == 2 && scales.dim() == 2 && qzeros.dim() == 2 && g_idx.dim() == 1,
      "pangolin_gemv_cpu expects input[M,K], scales/qzeros 2D, g_idx 1D");
  TORCH_CHECK(
      input.is_contiguous() && scales.is_contiguous() && qzeros.is_contiguous() && g_idx.is_contiguous(),
      "pangolin_gemv_cpu tensors must be contiguous");
  TORCH_CHECK(bits == 3 || bits == 5 || bits == 6 || bits == 7, "pangolin_gemv_cpu supports bits 3/5/6/7");

  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  const int64_t N = scales.size(1);
  const int64_t num_groups = scales.size(0);

  const std::array<int64_t, 10> supported_m = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32};
  TORCH_CHECK(
      std::find(supported_m.begin(), supported_m.end(), M) != supported_m.end(),
      "pangolin_gemv_cpu supports M in {1,2,3,4,5,6,7,8,16,32}, got ",
      M);
  TORCH_CHECK(K % 32 == 0 && N % 32 == 0, "pangolin_gemv_cpu K and N must be divisible by 32");
  TORCH_CHECK(
      qweight.dim() == 4 && qweight.size(0) == N / 16 && qweight.size(1) == K / 32 &&
          qweight.size(3) == 16 && qweight.scalar_type() == at::kInt && qweight.is_contiguous(),
      "pangolin_gemv_cpu qweight must be pre-packed as [N/16, K/32, bits, 16] int32 "
      "or the uint8 pre-expanded layout [N/16, K/32, 8, 16] int32");
  const int64_t kernel_bits = qweight.size(2);
  TORCH_CHECK(
      kernel_bits == bits || kernel_bits == 8,
      "pangolin_gemv_cpu qweight plane count must be bits or 8, got ",
      kernel_bits);
  TORCH_CHECK(
      qzeros.size(0) == num_groups && qzeros.size(1) == (N / 32) * bits,
      "pangolin_gemv_cpu qzeros must have planar shape [groups, N/32*bits]");
  TORCH_CHECK(g_idx.size(0) == K, "pangolin_gemv_cpu g_idx must have K entries");

  const auto x_f_tensor = input.to(at::kFloat).contiguous();
  const auto scales_b_tensor = scales.to(at::kBFloat16).contiguous();
  auto zero_scale_f_tensor = at::empty({num_groups, N}, at::TensorOptions().dtype(at::kFloat).device(scales.device()));

  const float* __restrict__ x_f = x_f_tensor.data_ptr<float>();
  const at::BFloat16* __restrict__ scale_b = scales_b_tensor.data_ptr<at::BFloat16>();
  float* __restrict__ zero_scale_f = zero_scale_f_tensor.data_ptr<float>();
  const int32_t* __restrict__ qweight_ptr = qweight.data_ptr<int32_t>();
  const int32_t* __restrict__ qzeros_ptr = qzeros.data_ptr<int32_t>();
  const int32_t* __restrict__ g_idx_ptr = g_idx.data_ptr<int32_t>();

  const int64_t qzeros_stride = (N / 32) * bits;

  // Pre-compute zero * scale for every (group, column) in FP32 so the hot loop
  // loads one BF16 scale vector and one FP32 zero-scale vector per k-block.
  // Use a vectorized AVX-512 path when available; otherwise fall back to the
  // scalar decoder.
#if PLANAR_GEMV_CPU_X86
  if (cpu_supports_avx512_core()) {
    switch (bits) {
      case 3:
        compute_zero_scale_avx512<3>(qzeros_ptr, scale_b, zero_scale_f, N, static_cast<int>(num_groups), qzeros_stride);
        break;
      case 5:
        compute_zero_scale_avx512<5>(qzeros_ptr, scale_b, zero_scale_f, N, static_cast<int>(num_groups), qzeros_stride);
        break;
      case 6:
        compute_zero_scale_avx512<6>(qzeros_ptr, scale_b, zero_scale_f, N, static_cast<int>(num_groups), qzeros_stride);
        break;
      case 7:
        compute_zero_scale_avx512<7>(qzeros_ptr, scale_b, zero_scale_f, N, static_cast<int>(num_groups), qzeros_stride);
        break;
    }
  } else
#endif
  {
    at::parallel_for(
        0,
        num_groups * N,
        std::max<int64_t>(1, num_groups * N / at::get_num_threads()),
        [&](int64_t begin, int64_t end) {
          switch (bits) {
            case 3:
              for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t g = idx / N;
                const int64_t n = idx % N;
                const int z = decode_zero_code<3>(qzeros_ptr, static_cast<int>(g), static_cast<int>(n), static_cast<int>(num_groups), N, qzeros_stride);
                const float s = static_cast<float>(scale_b[idx]);
                zero_scale_f[idx] = static_cast<float>(z) * s;
              }
              break;
            case 5:
              for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t g = idx / N;
                const int64_t n = idx % N;
                const int z = decode_zero_code<5>(qzeros_ptr, static_cast<int>(g), static_cast<int>(n), static_cast<int>(num_groups), N, qzeros_stride);
                const float s = static_cast<float>(scale_b[idx]);
                zero_scale_f[idx] = static_cast<float>(z) * s;
              }
              break;
            case 6:
              for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t g = idx / N;
                const int64_t n = idx % N;
                const int z = decode_zero_code<6>(qzeros_ptr, static_cast<int>(g), static_cast<int>(n), static_cast<int>(num_groups), N, qzeros_stride);
                const float s = static_cast<float>(scale_b[idx]);
                zero_scale_f[idx] = static_cast<float>(z) * s;
              }
              break;
            case 7:
              for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t g = idx / N;
                const int64_t n = idx % N;
                const int z = decode_zero_code<7>(qzeros_ptr, static_cast<int>(g), static_cast<int>(n), static_cast<int>(num_groups), N, qzeros_stride);
                const float s = static_cast<float>(scale_b[idx]);
                zero_scale_f[idx] = static_cast<float>(z) * s;
              }
              break;
          }
        });
  }

  auto output = torch::empty({M, N}, at::TensorOptions().dtype(at::kBFloat16).device(input.device()));
  at::BFloat16* out_ptr = output.data_ptr<at::BFloat16>();

  dispatch_size_m(M, kernel_bits, x_f, qweight_ptr, scale_b, zero_scale_f, g_idx_ptr, out_ptr, K, N, num_groups);

  return output;
}

}  // namespace pangolin_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_pangolin, m) {
  m.def(
      "gemv_cpu(Tensor input, Tensor qweight, Tensor scales, Tensor qzeros, Tensor g_idx, int bits) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_pangolin, CPU, m) {
  m.impl("gemv_cpu", pangolin_cpu::pangolin_gemv_cpu);
}
