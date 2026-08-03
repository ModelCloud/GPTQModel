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
static void planar_gemv_cpu_init_features() {
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
         __builtin_cpu_supports("avx512vl");
}

// AVX2 256-bit vector helpers.
__attribute__((target("avx2,fma"))) inline __m256i load_i_256(const void* p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

__attribute__((target("avx2,fma"))) inline __m256 load_f_256(const float* p) {
  return _mm256_loadu_ps(p);
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

__attribute__((target("avx2,fma"))) inline void store_f_256(float* p, __m256 a) {
  _mm256_storeu_ps(p, a);
}

// AVX-512 512-bit vector helpers.
__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512i load_i_512(const void* p) {
  return _mm512_loadu_si512(p);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512 load_f_512(const float* p) {
  return _mm512_loadu_ps(p);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512i set1_i_512(int v) {
  return _mm512_set1_epi32(v);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512 set1_f_512(float v) {
  return _mm512_set1_ps(v);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512i srlv_512(__m512i a, __m512i b) {
  return _mm512_srlv_epi32(a, b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512i slli_512(__m512i a, int imm) {
  return _mm512_slli_epi32(a, imm);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512i and_512(__m512i a, __m512i b) {
  return _mm512_and_si512(a, b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512i or_512(__m512i a, __m512i b) {
  return _mm512_or_si512(a, b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512 cvt_i_f_512(__m512i a) {
  return _mm512_cvtepi32_ps(a);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512 fmsub_512(__m512 a, __m512 b, __m512 c) {
  return _mm512_fmsub_ps(a, b, c);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline __m512 fmadd_512(__m512 a, __m512 b, __m512 c) {
  return _mm512_fmadd_ps(a, b, c);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) inline void store_f_512(float* p, __m512 a) {
  _mm512_storeu_ps(p, a);
}

#define DEFINE_GEMV_KERNEL(suffix, target_features, LOAD_I, LOAD_F, SET1_I, SET1_F, SRLV, SLLI, AND, OR, CVT, FMSUB, FMADD, STORE_F, INT_T, FLOAT_T, VLEN) \
  template <int Bits, int SizeM> \
  __attribute__((target(target_features))) \
  void gemv_col_block_##suffix( \
      const float* x_f, \
      const int32_t* qweight, \
      const float* scale_f, \
      const float* zero_scale_f, \
      const int32_t* g_idx, \
      at::BFloat16* out, \
      int64_t col0, \
      int64_t K, \
      int64_t N, \
      int64_t num_groups) { \
    constexpr auto planes = plane_info<Bits>(); \
    const int64_t num_k_blocks = K / 32; \
    FLOAT_T acc[SizeM]; \
    for (int m = 0; m < SizeM; ++m) { \
      acc[m] = SET1_F(0.0f); \
    } \
    for (int64_t kb = 0; kb < num_k_blocks; ++kb) { \
      const int64_t row0 = kb * 32; \
      int group = g_idx[row0]; \
      if (group < 0) { \
        group += static_cast<int>(num_groups); \
      } \
      const float* scale_ptr = scale_f + static_cast<int64_t>(group) * N + col0; \
      const float* zscale_ptr = zero_scale_f + static_cast<int64_t>(group) * N + col0; \
      FLOAT_T scale_vec = LOAD_F(scale_ptr); \
      FLOAT_T zscale_vec = LOAD_F(zscale_ptr); \
      INT_T qwords[Bits]; \
      for (int pw = 0; pw < Bits; ++pw) { \
        const int64_t row = static_cast<int64_t>(kb) * Bits + pw; \
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(qweight) + row * N + col0; \
        qwords[pw] = LOAD_I(ptr); \
      } \
      for (int k = 0; k < 32; ++k) { \
        INT_T code = SET1_I(0); \
        for (int p = 0; p < 3; ++p) { \
          const int w = planes[p].w; \
          if (w == 0) { \
            break; \
          } \
          const int pack_factor = 32 / w; \
          INT_T words = qwords[planes[p].start + k / pack_factor]; \
          INT_T shift = SET1_I(w * (k % pack_factor)); \
          INT_T shifted = SRLV(words, shift); \
          INT_T masked = AND(shifted, SET1_I((1 << w) - 1)); \
          code = OR(code, SLLI(masked, planes[p].off)); \
        } \
        FLOAT_T code_f = CVT(code); \
        FLOAT_T weight = FMSUB(code_f, scale_vec, zscale_vec); \
        for (int m = 0; m < SizeM; ++m) { \
          FLOAT_T xval = SET1_F(x_f[m * K + row0 + k]); \
          acc[m] = FMADD(xval, weight, acc[m]); \
        } \
      } \
    } \
    for (int m = 0; m < SizeM; ++m) { \
      alignas(64) float tmp[VLEN]; \
      STORE_F(tmp, acc[m]); \
      for (int i = 0; i < VLEN; ++i) { \
        out[static_cast<int64_t>(m) * N + col0 + i] = at::BFloat16(tmp[i]); \
      } \
    } \
  } \
  template <int Bits, int SizeM> \
  void gemv_kernel_##suffix( \
      const float* x_f, \
      const int32_t* qweight, \
      const float* scale_f, \
      const float* zero_scale_f, \
      const int32_t* g_idx, \
      at::BFloat16* out, \
      int64_t /*M*/, \
      int64_t K, \
      int64_t N, \
      int64_t num_groups) { \
    /* No tail: N % 32 == 0 is checked by pangolin_gemv_cpu and VLEN divides 32. */ \
    const int64_t col_chunks = N / VLEN; \
    const int64_t grain = std::max<int64_t>(1, col_chunks / at::get_num_threads()); \
    at::parallel_for(0, col_chunks, grain, [&](int64_t begin, int64_t end) { \
      for (int64_t col_block = begin; col_block < end; ++col_block) { \
        gemv_col_block_##suffix<Bits, SizeM>( \
            x_f, qweight, scale_f, zero_scale_f, g_idx, out, col_block * VLEN, K, N, num_groups); \
      } \
    }); \
  }

DEFINE_GEMV_KERNEL(
    avx2,
    "avx2,fma",
    load_i_256,
    load_f_256,
    set1_i_256,
    set1_f_256,
    srlv_256,
    slli_256,
    and_256,
    or_256,
    cvt_i_f_256,
    fmsub_256,
    fmadd_256,
    store_f_256,
    __m256i,
    __m256,
    8)

DEFINE_GEMV_KERNEL(
    avx512,
    "avx512f,avx512bw,avx512vl,fma",
    load_i_512,
    load_f_512,
    set1_i_512,
    set1_f_512,
    srlv_512,
    slli_512,
    and_512,
    or_512,
    cvt_i_f_512,
    fmsub_512,
    fmadd_512,
    store_f_512,
    __m512i,
    __m512,
    16)

#endif  // PLANAR_GEMV_CPU_X86

template <int Bits, int SizeM>
void gemv_kernel_scalar(
    const float* x_f,
    const int32_t* qweight,
    const float* scale_f,
    const float* zero_scale_f,
    const int32_t* g_idx,
    at::BFloat16* out,
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
        const float scale = scale_f[static_cast<int64_t>(group) * N + n];
        const float zscale = zero_scale_f[static_cast<int64_t>(group) * N + n];
        for (int k = 0; k < 32; ++k) {
          int code = 0;
          for (int p = 0; p < 3; ++p) {
            const int w = planes[p].w;
            if (w == 0) {
              break;
            }
            const int pack_factor = 32 / w;
            const int64_t row = static_cast<int64_t>(kb) * Bits + planes[p].start + k / pack_factor;
            const uint32_t word = static_cast<uint32_t>(qweight[row * N + n]);
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
    const float* x_f,
    const int32_t* qweight,
    const float* scale_f,
    const float* zero_scale_f,
    const int32_t* g_idx,
    at::BFloat16* out,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t num_groups) {
#if PLANAR_GEMV_CPU_X86
  // AVX-512 ZMM register pressure with many accumulators causes spills that the
  // current target setup mishandles; keep the fast M=1..8 decode shapes on AVX-512
  // and route larger M through the AVX2 path.
  if (cpu_supports_avx512_core() && N % 16 == 0 && SizeM <= 8) {
    gemv_kernel_avx512<Bits, SizeM>(x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
    return;
  }
  if (cpu_supports_avx2() && N % 8 == 0) {
    gemv_kernel_avx2<Bits, SizeM>(x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
    return;
  }
#endif
  gemv_kernel_scalar<Bits, SizeM>(x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
}

template <int SizeM>
static void dispatch_bits(
    int64_t bits,
    const float* x_f,
    const int32_t* qweight,
    const float* scale_f,
    const float* zero_scale_f,
    const int32_t* g_idx,
    at::BFloat16* out,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t num_groups) {
  switch (bits) {
    case 3:
      run_gemv<3, SizeM>(x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 5:
      run_gemv<5, SizeM>(x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 6:
      run_gemv<6, SizeM>(x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 7:
      run_gemv<7, SizeM>(x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    default:
      TORCH_CHECK(false, "pangolin_gemv_cpu supports bits 3/5/6/7, got ", bits);
  }
}

static void dispatch_size_m(
    int64_t M,
    int64_t bits,
    const float* x_f,
    const int32_t* qweight,
    const float* scale_f,
    const float* zero_scale_f,
    const int32_t* g_idx,
    at::BFloat16* out,
    int64_t K,
    int64_t N,
    int64_t num_groups) {
  switch (M) {
    case 1:
      dispatch_bits<1>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 2:
      dispatch_bits<2>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 3:
      dispatch_bits<3>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 4:
      dispatch_bits<4>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 5:
      dispatch_bits<5>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 6:
      dispatch_bits<6>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 7:
      dispatch_bits<7>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 8:
      dispatch_bits<8>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 16:
      dispatch_bits<16>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
      break;
    case 32:
      dispatch_bits<32>(bits, x_f, qweight, scale_f, zero_scale_f, g_idx, out, M, K, N, num_groups);
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
      input.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2 && qzeros.dim() == 2 && g_idx.dim() == 1,
      "pangolin_gemv_cpu expects input[M,K], qweight/scales/qzeros 2D, g_idx 1D");
  TORCH_CHECK(
      input.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous() && qzeros.is_contiguous() &&
          g_idx.is_contiguous(),
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
      qweight.size(0) == (K / 32) * bits && qweight.size(1) == N,
      "pangolin_gemv_cpu qweight must have planar shape [K/32*bits, N]");
  TORCH_CHECK(
      qzeros.size(0) == num_groups && qzeros.size(1) == (N / 32) * bits,
      "pangolin_gemv_cpu qzeros must have planar shape [groups, N/32*bits]");
  TORCH_CHECK(g_idx.size(0) == K, "pangolin_gemv_cpu g_idx must have K entries");

  const auto x_f_tensor = input.to(at::kFloat).contiguous();
  const auto scales_f_tensor = scales.to(at::kFloat).contiguous();
  auto zero_scale_f_tensor = at::empty_like(scales_f_tensor);

  const float* x_f = x_f_tensor.data_ptr<float>();
  const float* scale_f = scales_f_tensor.data_ptr<float>();
  float* zero_scale_f = zero_scale_f_tensor.data_ptr<float>();
  const int32_t* qweight_ptr = qweight.data_ptr<int32_t>();
  const int32_t* qzeros_ptr = qzeros.data_ptr<int32_t>();
  const int32_t* g_idx_ptr = g_idx.data_ptr<int32_t>();

  const int64_t qzeros_stride = (N / 32) * bits;

  // Pre-compute zero * scale for every (group, column) so the hot loop only
  // loads two float vectors per k-block.
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
              const float s = scale_f[idx];
              zero_scale_f[idx] = static_cast<float>(z) * s;
            }
            break;
          case 5:
            for (int64_t idx = begin; idx < end; ++idx) {
              const int64_t g = idx / N;
              const int64_t n = idx % N;
              const int z = decode_zero_code<5>(qzeros_ptr, static_cast<int>(g), static_cast<int>(n), static_cast<int>(num_groups), N, qzeros_stride);
              const float s = scale_f[idx];
              zero_scale_f[idx] = static_cast<float>(z) * s;
            }
            break;
          case 6:
            for (int64_t idx = begin; idx < end; ++idx) {
              const int64_t g = idx / N;
              const int64_t n = idx % N;
              const int z = decode_zero_code<6>(qzeros_ptr, static_cast<int>(g), static_cast<int>(n), static_cast<int>(num_groups), N, qzeros_stride);
              const float s = scale_f[idx];
              zero_scale_f[idx] = static_cast<float>(z) * s;
            }
            break;
          case 7:
            for (int64_t idx = begin; idx < end; ++idx) {
              const int64_t g = idx / N;
              const int64_t n = idx % N;
              const int z = decode_zero_code<7>(qzeros_ptr, static_cast<int>(g), static_cast<int>(n), static_cast<int>(num_groups), N, qzeros_stride);
              const float s = scale_f[idx];
              zero_scale_f[idx] = static_cast<float>(z) * s;
            }
            break;
        }
      });

  auto output = torch::empty({M, N}, at::TensorOptions().dtype(at::kBFloat16).device(input.device()));
  at::BFloat16* out_ptr = output.data_ptr<at::BFloat16>();

  dispatch_size_m(M, bits, x_f, qweight_ptr, scale_f, zero_scale_f, g_idx_ptr, out_ptr, K, N, num_groups);

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
