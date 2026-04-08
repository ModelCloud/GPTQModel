// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define GPTQMODEL_FLOATX_X86 1
#else
#define GPTQMODEL_FLOATX_X86 0
#endif

namespace gptqmodel_floatx {

enum class ScaleMode : int64_t {
  kNone = 0,
  kMultiply = 1,
  kDivide = 2,
};

enum class TargetKind : int64_t {
  kBFloat16 = 0,
  kFloat16 = 1,
};

enum class Fp8Format : int64_t {
  kE4M3Fn = 0,
  kE5M2 = 1,
  kE4M3FnUz = 2,
  kE5M2FnUz = 3,
  kE8M0Fnu = 4,
};

enum class ScaleLayout1D {
  kNone,
  kScalar,
  kVector,
  kRepeat,
};

enum class ScaleLayout2D {
  kNone,
  kScalar,
  kFull,
  kBlock,
  kRowRepeat,
  kColRepeat,
};

struct ScaleSpec1D {
  ScaleLayout1D layout = ScaleLayout1D::kNone;
  const float* ptr = nullptr;
  int64_t length = 0;
  int64_t repeat = 1;
};

struct ScaleSpec2D {
  ScaleLayout2D layout = ScaleLayout2D::kNone;
  const float* ptr = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t scale_rows = 0;
  int64_t scale_cols = 0;
  int64_t row_repeat = 1;
  int64_t col_repeat = 1;
};

inline int64_t clamped_threads(int64_t requested) {
  const int64_t limit = 8;
  if (requested > 0) {
    return std::max<int64_t>(1, std::min<int64_t>(requested, limit));
  }
  return std::max<int64_t>(1, std::min<int64_t>(at::get_num_threads(), limit));
}

template <typename SrcT, typename DstT>
std::array<float, 256> build_fp8_table() {
  std::array<float, 256> table{};
  for (int value = 0; value < 256; ++value) {
    const float decoded =
        static_cast<float>(SrcT(static_cast<uint8_t>(value), SrcT::from_bits()));
    table[value] = static_cast<float>(DstT(decoded));
  }
  return table;
}

const std::array<float, 256>& fp8_table(Fp8Format format, TargetKind target_kind) {
  // Target-rounded tables keep the hot loops from re-quantizing decoded values.
  static const auto e4m3fn_bf16 = build_fp8_table<c10::Float8_e4m3fn, c10::BFloat16>();
  static const auto e5m2_bf16 = build_fp8_table<c10::Float8_e5m2, c10::BFloat16>();
  static const auto e4m3fnuz_bf16 = build_fp8_table<c10::Float8_e4m3fnuz, c10::BFloat16>();
  static const auto e5m2fnuz_bf16 = build_fp8_table<c10::Float8_e5m2fnuz, c10::BFloat16>();
  static const auto e8m0fnu_bf16 = build_fp8_table<c10::Float8_e8m0fnu, c10::BFloat16>();
  static const auto e4m3fn_fp16 = build_fp8_table<c10::Float8_e4m3fn, c10::Half>();
  static const auto e5m2_fp16 = build_fp8_table<c10::Float8_e5m2, c10::Half>();
  static const auto e4m3fnuz_fp16 = build_fp8_table<c10::Float8_e4m3fnuz, c10::Half>();
  static const auto e5m2fnuz_fp16 = build_fp8_table<c10::Float8_e5m2fnuz, c10::Half>();
  static const auto e8m0fnu_fp16 = build_fp8_table<c10::Float8_e8m0fnu, c10::Half>();
  switch (target_kind) {
    case TargetKind::kBFloat16:
      switch (format) {
        case Fp8Format::kE4M3Fn:
          return e4m3fn_bf16;
        case Fp8Format::kE5M2:
          return e5m2_bf16;
        case Fp8Format::kE4M3FnUz:
          return e4m3fnuz_bf16;
        case Fp8Format::kE5M2FnUz:
          return e5m2fnuz_bf16;
        case Fp8Format::kE8M0Fnu:
          return e8m0fnu_bf16;
      }
      break;
    case TargetKind::kFloat16:
      switch (format) {
        case Fp8Format::kE4M3Fn:
          return e4m3fn_fp16;
        case Fp8Format::kE5M2:
          return e5m2_fp16;
        case Fp8Format::kE4M3FnUz:
          return e4m3fnuz_fp16;
        case Fp8Format::kE5M2FnUz:
          return e5m2fnuz_fp16;
        case Fp8Format::kE8M0Fnu:
          return e8m0fnu_fp16;
      }
      break;
  }
  TORCH_CHECK(false, "Unsupported FP8 format code");
}

template <typename DstT>
std::array<float, 16> build_fp4_table() {
  static constexpr std::array<float, 16> kDecodedValues = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  std::array<float, 16> table{};
  for (size_t idx = 0; idx < table.size(); ++idx) {
    table[idx] = static_cast<float>(DstT(kDecodedValues[idx]));
  }
  return table;
}

const std::array<float, 16>& fp4_table(TargetKind target_kind) {
  static const auto bf16 = build_fp4_table<c10::BFloat16>();
  static const auto fp16 = build_fp4_table<c10::Half>();
  switch (target_kind) {
    case TargetKind::kBFloat16:
      return bf16;
    case TargetKind::kFloat16:
      return fp16;
  }
  TORCH_CHECK(false, "Unsupported target dtype for FP4 table");
}

ScaleSpec1D make_scale_spec_1d(
    const c10::optional<at::Tensor>& scale_opt,
    int64_t result_len,
    int64_t axis,
    bool axis_is_none) {
  (void)axis;
  (void)axis_is_none;
  ScaleSpec1D spec;
  if (!scale_opt.has_value() || !scale_opt->defined()) {
    return spec;
  }

  const at::Tensor& scale = *scale_opt;
  TORCH_CHECK(scale.device().is_cpu(), "scale tensor must reside on CPU");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale tensor must be float32");
  TORCH_CHECK(scale.is_contiguous(), "scale tensor must be contiguous");

  spec.ptr = scale.const_data_ptr<float>();
  if (scale.ndimension() == 0) {
    spec.layout = ScaleLayout1D::kScalar;
    return spec;
  }

  TORCH_CHECK(scale.ndimension() == 1, "1D dequantization only supports scalar or 1D scale tensors");
  spec.length = scale.numel();
  if (spec.length == result_len) {
    spec.layout = ScaleLayout1D::kVector;
    return spec;
  }
  TORCH_CHECK(result_len % spec.length == 0, "scale tensor shape incompatible with 1D output");
  spec.layout = ScaleLayout1D::kRepeat;
  spec.repeat = result_len / spec.length;
  return spec;
}

ScaleSpec2D make_scale_spec_2d(
    const c10::optional<at::Tensor>& scale_opt,
    int64_t rows,
    int64_t cols,
    int64_t axis,
    bool axis_is_none) {
  ScaleSpec2D spec;
  if (!scale_opt.has_value() || !scale_opt->defined()) {
    return spec;
  }

  const at::Tensor& scale = *scale_opt;
  TORCH_CHECK(scale.device().is_cpu(), "scale tensor must reside on CPU");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale tensor must be float32");
  TORCH_CHECK(scale.is_contiguous(), "scale tensor must be contiguous");

  spec.ptr = scale.const_data_ptr<float>();
  if (scale.ndimension() == 0) {
    spec.layout = ScaleLayout2D::kScalar;
    return spec;
  }

  if (scale.ndimension() == 1) {
    const int64_t count = scale.size(0);
    const int64_t resolved_axis = axis_is_none ? 0 : axis;
    if (resolved_axis == 0) {
      TORCH_CHECK(rows % count == 0, "row scale tensor shape incompatible with output");
      spec.layout = ScaleLayout2D::kRowRepeat;
      spec.scale_rows = count;
      spec.row_repeat = rows / count;
      return spec;
    }
    TORCH_CHECK(resolved_axis == 1, "axis must be 0 or 1 for 2D scale tensors");
    TORCH_CHECK(cols % count == 0, "column scale tensor shape incompatible with output");
    spec.layout = ScaleLayout2D::kColRepeat;
    spec.scale_cols = count;
    spec.col_repeat = cols / count;
    return spec;
  }

  TORCH_CHECK(scale.ndimension() == 2, "2D dequantization only supports scale tensors up to rank 2");
  spec.scale_rows = scale.size(0);
  spec.scale_cols = scale.size(1);
  if (spec.scale_rows == rows && spec.scale_cols == cols) {
    spec.layout = ScaleLayout2D::kFull;
    spec.cols = cols;
    return spec;
  }

  TORCH_CHECK(
      rows % spec.scale_rows == 0 && cols % spec.scale_cols == 0,
      "block scale tensor shape incompatible with output");
  spec.layout = ScaleLayout2D::kBlock;
  spec.row_repeat = rows / spec.scale_rows;
  spec.col_repeat = cols / spec.scale_cols;
  spec.cols = cols;
  return spec;
}

inline float scale_value_1d(const ScaleSpec1D& spec, int64_t idx) {
  switch (spec.layout) {
    case ScaleLayout1D::kNone:
      return 1.0f;
    case ScaleLayout1D::kScalar:
      return spec.ptr[0];
    case ScaleLayout1D::kVector:
      return spec.ptr[idx];
    case ScaleLayout1D::kRepeat:
      return spec.ptr[idx / spec.repeat];
  }
  return 1.0f;
}

inline float scale_value_2d(const ScaleSpec2D& spec, int64_t row, int64_t col) {
  switch (spec.layout) {
    case ScaleLayout2D::kNone:
      return 1.0f;
    case ScaleLayout2D::kScalar:
      return spec.ptr[0];
    case ScaleLayout2D::kFull:
      return spec.ptr[row * spec.cols + col];
    case ScaleLayout2D::kBlock:
      return spec.ptr[(row / spec.row_repeat) * spec.scale_cols + (col / spec.col_repeat)];
    case ScaleLayout2D::kRowRepeat:
      return spec.ptr[row / spec.row_repeat];
    case ScaleLayout2D::kColRepeat:
      return spec.ptr[col / spec.col_repeat];
  }
  return 1.0f;
}

template <typename T>
inline void store_scalar(T* dst, float value);

template <>
inline void store_scalar<c10::Half>(c10::Half* dst, float value) {
  *dst = c10::Half(value);
}

template <>
inline void store_scalar<c10::BFloat16>(c10::BFloat16* dst, float value) {
  *dst = c10::BFloat16(value);
}

template <typename T>
inline void apply_scale_and_store_scalar(
    T* dst,
    const float* values,
    int64_t count,
    ScaleMode scale_mode,
    const ScaleSpec2D& spec,
    int64_t row,
    int64_t col_base) {
  for (int64_t i = 0; i < count; ++i) {
    const T rounded_value = T(values[i]);
    float value = static_cast<float>(rounded_value);
    if (scale_mode != ScaleMode::kNone) {
      const T rounded_scale = T(scale_value_2d(spec, row, col_base + i));
      const float scale = static_cast<float>(rounded_scale);
      value = scale_mode == ScaleMode::kMultiply ? value * scale : value / scale;
    }
    store_scalar(dst + i, value);
  }
}

template <typename T>
inline void apply_scale_and_store_scalar_1d(
    T* dst,
    float value,
    ScaleMode scale_mode,
    const ScaleSpec1D& spec,
    int64_t idx) {
  const T rounded_value = T(value);
  value = static_cast<float>(rounded_value);
  if (scale_mode != ScaleMode::kNone) {
    const T rounded_scale = T(scale_value_1d(spec, idx));
    const float scale = static_cast<float>(rounded_scale);
    value = scale_mode == ScaleMode::kMultiply ? value * scale : value / scale;
  }
  store_scalar(dst + idx, value);
}

#if GPTQMODEL_FLOATX_X86 && (defined(__GNUC__) || defined(__clang__))
inline bool env_flag_enabled(const char* name) {
  // Allow tests and compatibility debugging to force the scalar fallback.
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
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

inline bool cpu_supports_avx2() {
  if (env_flag_enabled("GPTQMODEL_FLOATX_CPU_DISABLE_AVX2")) {
    return false;
  }
  return __builtin_cpu_supports("avx2");
}

inline bool cpu_supports_f16c() {
  if (env_flag_enabled("GPTQMODEL_FLOATX_CPU_DISABLE_AVX2")) {
    return false;
  }
  return __builtin_cpu_supports("f16c");
}

__attribute__((target("avx2")))
inline __m256 load_fp8x8_to_ps(const uint8_t* src, const float* table) {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
  const __m256i indices = _mm256_cvtepu8_epi32(raw);
  return _mm256_i32gather_ps(table, indices, 4);
}

__attribute__((target("avx2")))
inline void load_fp8x16_to_ps(
    const uint8_t* src,
    const float* table,
    __m256* values_lo,
    __m256* values_hi) {
  // Two gathers amortize loop overhead on the common 16-aligned benchmark shapes.
  *values_lo = load_fp8x8_to_ps(src, table);
  *values_hi = load_fp8x8_to_ps(src + 8, table);
}

__attribute__((target("avx2")))
inline void load_fp4x16_to_ps(
    const uint8_t* src,
    const float* table,
    __m256* values_lo,
    __m256* values_hi) {
  // Decode 8 packed bytes into 16 logical FP4 values in column order.
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
  const __m128i lo_nibbles = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
  const __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
  const __m128i interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
  *values_lo = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(interleaved), 4);
  *values_hi = _mm256_i32gather_ps(table, _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved, 8)), 4);
}

__attribute__((target("avx2")))
inline void fill_scale8(
    float* dst,
    const ScaleSpec2D& spec,
    int64_t row,
    int64_t col_base) {
  for (int i = 0; i < 8; ++i) {
    dst[i] = scale_value_2d(spec, row, col_base + i);
  }
}

__attribute__((target("avx2")))
inline __m256 round_ps_to_bf16_ps(__m256 values) {
  // Match the Python reference by rounding operands to bf16 before scaling.
  const __m256i bits = _mm256_castps_si256(values);
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
  const __m256i rounding = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb);
  const __m256i rounded = _mm256_add_epi32(bits, rounding);
  const __m256i bf16_bits = _mm256_slli_epi32(_mm256_srli_epi32(rounded, 16), 16);
  return _mm256_castsi256_ps(bf16_bits);
}

__attribute__((target("avx2,f16c")))
inline __m256 round_ps_to_fp16_ps(__m256 values) {
  // F16C lets us round to fp16 and back to float32 without leaving SIMD.
  const __m128i fp16 =
      _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return _mm256_cvtph_ps(fp16);
}

__attribute__((target("avx2")))
inline __m256 load_scale8_ps(
    const ScaleSpec2D& spec,
    int64_t row,
    int64_t col_base) {
  // Most benchmarked scale layouts are constant or contiguous across eight lanes,
  // so specialize those cases and only fall back to per-lane expansion when needed.
  switch (spec.layout) {
    case ScaleLayout2D::kNone:
      return _mm256_set1_ps(1.0f);
    case ScaleLayout2D::kScalar:
      return _mm256_set1_ps(spec.ptr[0]);
    case ScaleLayout2D::kFull:
      return _mm256_loadu_ps(spec.ptr + row * spec.cols + col_base);
    case ScaleLayout2D::kRowRepeat:
      return _mm256_set1_ps(spec.ptr[row / spec.row_repeat]);
    case ScaleLayout2D::kColRepeat: {
      if (spec.col_repeat == 1) {
        return _mm256_loadu_ps(spec.ptr + col_base);
      }
      if ((col_base / spec.col_repeat) == ((col_base + 7) / spec.col_repeat)) {
        return _mm256_set1_ps(spec.ptr[col_base / spec.col_repeat]);
      }
      alignas(32) float scales[8];
      fill_scale8(scales, spec, row, col_base);
      return _mm256_load_ps(scales);
    }
    case ScaleLayout2D::kBlock: {
      const int64_t block_row = row / spec.row_repeat;
      if (spec.col_repeat == 1) {
        return _mm256_loadu_ps(spec.ptr + block_row * spec.scale_cols + col_base);
      }
      if ((col_base / spec.col_repeat) == ((col_base + 7) / spec.col_repeat)) {
        return _mm256_set1_ps(spec.ptr[block_row * spec.scale_cols + (col_base / spec.col_repeat)]);
      }
      alignas(32) float scales[8];
      fill_scale8(scales, spec, row, col_base);
      return _mm256_load_ps(scales);
    }
  }
  return _mm256_set1_ps(1.0f);
}

__attribute__((target("avx2")))
inline void store_bf16x8(c10::BFloat16* dst, __m256 values);

__attribute__((target("avx2,f16c")))
inline void store_fp16x8(c10::Half* dst, __m256 values);

__attribute__((target("avx2")))
inline void apply_scale_and_store_bf16x8(
    c10::BFloat16* dst,
    __m256 values,
    ScaleMode scale_mode,
    const ScaleSpec2D& spec,
    int64_t row,
    int64_t col_base) {
  // Values arrive pre-rounded from the lookup tables; only scale operands need rounding here.
  if (scale_mode != ScaleMode::kNone) {
    __m256 scales = round_ps_to_bf16_ps(load_scale8_ps(spec, row, col_base));
    values = scale_mode == ScaleMode::kMultiply ? _mm256_mul_ps(values, scales) : _mm256_div_ps(values, scales);
  }
  store_bf16x8(dst, values);
}

__attribute__((target("avx2")))
inline void apply_scale_and_store_bf16x8_const(
    c10::BFloat16* dst,
    __m256 values,
    ScaleMode scale_mode,
    __m256 rounded_scale) {
  if (scale_mode != ScaleMode::kNone) {
    values = scale_mode == ScaleMode::kMultiply ? _mm256_mul_ps(values, rounded_scale) : _mm256_div_ps(values, rounded_scale);
  }
  store_bf16x8(dst, values);
}

__attribute__((target("avx2")))
inline void store_bf16x8(c10::BFloat16* dst, __m256 values) {
  const __m256i bits = _mm256_castps_si256(values);
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
  const __m256i rounding = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb);
  const __m256i rounded = _mm256_add_epi32(bits, rounding);
  const __m256i bf16 = _mm256_srli_epi32(rounded, 16);
  const __m128i lo = _mm256_castsi256_si128(bf16);
  const __m128i hi = _mm256_extracti128_si256(bf16, 1);
  const __m128i packed = _mm_packus_epi32(lo, hi);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), packed);
}

__attribute__((target("avx2,f16c")))
inline void store_fp16x8(c10::Half* dst, __m256 values) {
  const __m128i packed = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), packed);
}

__attribute__((target("avx2,f16c")))
inline void apply_scale_and_store_fp16x8(
    c10::Half* dst,
    __m256 values,
    ScaleMode scale_mode,
    const ScaleSpec2D& spec,
    int64_t row,
    int64_t col_base) {
  // Values arrive pre-rounded from the lookup tables; only scale operands need rounding here.
  if (scale_mode != ScaleMode::kNone) {
    __m256 scales = round_ps_to_fp16_ps(load_scale8_ps(spec, row, col_base));
    values = scale_mode == ScaleMode::kMultiply ? _mm256_mul_ps(values, scales) : _mm256_div_ps(values, scales);
  }
  store_fp16x8(dst, values);
}

__attribute__((target("avx2,f16c")))
inline void apply_scale_and_store_fp16x8_const(
    c10::Half* dst,
    __m256 values,
    ScaleMode scale_mode,
    __m256 rounded_scale) {
  if (scale_mode != ScaleMode::kNone) {
    values = scale_mode == ScaleMode::kMultiply ? _mm256_mul_ps(values, rounded_scale) : _mm256_div_ps(values, rounded_scale);
  }
  store_fp16x8(dst, values);
}

__attribute__((target("avx2")))
void dequantize_fp8_row_avx2_bf16(
    const uint8_t* src_row,
    c10::BFloat16* dst_row,
    int64_t cols,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t row) {
  int64_t col = 0;
  for (; col + 16 <= cols; col += 16) {
    __m256 values_lo;
    __m256 values_hi;
    load_fp8x16_to_ps(src_row + col, table.data(), &values_lo, &values_hi);
    apply_scale_and_store_bf16x8(dst_row + col, values_lo, scale_mode, spec, row, col);
    apply_scale_and_store_bf16x8(dst_row + col + 8, values_hi, scale_mode, spec, row, col + 8);
  }
  for (; col + 8 <= cols; col += 8) {
    __m256 values = load_fp8x8_to_ps(src_row + col, table.data());
    apply_scale_and_store_bf16x8(dst_row + col, values, scale_mode, spec, row, col);
  }
  if (col < cols) {
    alignas(32) float tail[8] = {};
    const int64_t tail_count = cols - col;
    for (int64_t i = 0; i < tail_count; ++i) {
      tail[i] = table[src_row[col + i]];
    }
    apply_scale_and_store_scalar(dst_row + col, tail, tail_count, scale_mode, spec, row, col);
  }
}

__attribute__((target("avx2")))
void dequantize_fp8_row_avx2_bf16_block_scale(
    const uint8_t* src_row,
    c10::BFloat16* dst_row,
    int64_t cols,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t row) {
  const int64_t block_row = row / spec.row_repeat;
  int64_t col = 0;
  while (col < cols) {
    const int64_t block_col = col / spec.col_repeat;
    const int64_t block_end = std::min<int64_t>(cols, (block_col + 1) * spec.col_repeat);
    const float rounded_scale = static_cast<float>(c10::BFloat16(spec.ptr[block_row * spec.scale_cols + block_col]));
    const __m256 scale_vec = _mm256_set1_ps(rounded_scale);

    // Real FP8 checkpoints reuse one scale across an entire block, so keep it hot for the whole span.
    for (; col + 16 <= block_end; col += 16) {
      __m256 values_lo;
      __m256 values_hi;
      load_fp8x16_to_ps(src_row + col, table.data(), &values_lo, &values_hi);
      apply_scale_and_store_bf16x8_const(dst_row + col, values_lo, scale_mode, scale_vec);
      apply_scale_and_store_bf16x8_const(dst_row + col + 8, values_hi, scale_mode, scale_vec);
    }
    for (; col + 8 <= block_end; col += 8) {
      __m256 values = load_fp8x8_to_ps(src_row + col, table.data());
      apply_scale_and_store_bf16x8_const(dst_row + col, values, scale_mode, scale_vec);
    }
    if (col < block_end) {
      alignas(32) float tail[8] = {};
      const int64_t tail_count = block_end - col;
      for (int64_t i = 0; i < tail_count; ++i) {
        float value = table[src_row[col + i]];
        if (scale_mode != ScaleMode::kNone) {
          value = scale_mode == ScaleMode::kMultiply ? value * rounded_scale : value / rounded_scale;
        }
        tail[i] = value;
      }
      for (int64_t i = 0; i < tail_count; ++i) {
        store_scalar(dst_row + col + i, tail[i]);
      }
      col = block_end;
    }
  }
}

__attribute__((target("avx2,f16c")))
void dequantize_fp8_row_avx2_fp16(
    const uint8_t* src_row,
    c10::Half* dst_row,
    int64_t cols,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t row) {
  int64_t col = 0;
  for (; col + 16 <= cols; col += 16) {
    __m256 values_lo;
    __m256 values_hi;
    load_fp8x16_to_ps(src_row + col, table.data(), &values_lo, &values_hi);
    apply_scale_and_store_fp16x8(dst_row + col, values_lo, scale_mode, spec, row, col);
    apply_scale_and_store_fp16x8(dst_row + col + 8, values_hi, scale_mode, spec, row, col + 8);
  }
  for (; col + 8 <= cols; col += 8) {
    __m256 values = load_fp8x8_to_ps(src_row + col, table.data());
    apply_scale_and_store_fp16x8(dst_row + col, values, scale_mode, spec, row, col);
  }
  if (col < cols) {
    alignas(32) float tail[8] = {};
    const int64_t tail_count = cols - col;
    for (int64_t i = 0; i < tail_count; ++i) {
      tail[i] = table[src_row[col + i]];
    }
    apply_scale_and_store_scalar(dst_row + col, tail, tail_count, scale_mode, spec, row, col);
  }
}

__attribute__((target("avx2,f16c")))
void dequantize_fp8_row_avx2_fp16_block_scale(
    const uint8_t* src_row,
    c10::Half* dst_row,
    int64_t cols,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t row) {
  const int64_t block_row = row / spec.row_repeat;
  int64_t col = 0;
  while (col < cols) {
    const int64_t block_col = col / spec.col_repeat;
    const int64_t block_end = std::min<int64_t>(cols, (block_col + 1) * spec.col_repeat);
    const float rounded_scale = static_cast<float>(c10::Half(spec.ptr[block_row * spec.scale_cols + block_col]));
    const __m256 scale_vec = _mm256_set1_ps(rounded_scale);

    // Real FP8 checkpoints reuse one scale across an entire block, so keep it hot for the whole span.
    for (; col + 16 <= block_end; col += 16) {
      __m256 values_lo;
      __m256 values_hi;
      load_fp8x16_to_ps(src_row + col, table.data(), &values_lo, &values_hi);
      apply_scale_and_store_fp16x8_const(dst_row + col, values_lo, scale_mode, scale_vec);
      apply_scale_and_store_fp16x8_const(dst_row + col + 8, values_hi, scale_mode, scale_vec);
    }
    for (; col + 8 <= block_end; col += 8) {
      __m256 values = load_fp8x8_to_ps(src_row + col, table.data());
      apply_scale_and_store_fp16x8_const(dst_row + col, values, scale_mode, scale_vec);
    }
    if (col < block_end) {
      alignas(32) float tail[8] = {};
      const int64_t tail_count = block_end - col;
      for (int64_t i = 0; i < tail_count; ++i) {
        float value = table[src_row[col + i]];
        if (scale_mode != ScaleMode::kNone) {
          value = scale_mode == ScaleMode::kMultiply ? value * rounded_scale : value / rounded_scale;
        }
        tail[i] = value;
      }
      for (int64_t i = 0; i < tail_count; ++i) {
        store_scalar(dst_row + col + i, tail[i]);
      }
      col = block_end;
    }
  }
}
#else
inline bool cpu_supports_avx2() {
  return false;
}

inline bool cpu_supports_f16c() {
  return false;
}
#endif

template <typename T>
void dequantize_fp8_scalar(
    const uint8_t* src,
    T* dst,
    int64_t rows,
    int64_t cols,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t threads) {
  const int64_t grain = std::max<int64_t>(1, rows / clamped_threads(threads));
  at::parallel_for(0, rows, grain, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const uint8_t* src_row = src + row * cols;
      T* dst_row = dst + row * cols;
      for (int64_t col = 0; col < cols; ++col) {
        const float value[1] = {table[src_row[col]]};
        apply_scale_and_store_scalar(dst_row + col, value, 1, scale_mode, spec, row, col);
      }
    }
  });
}

void dequantize_fp8_2d(
    const at::Tensor& source,
    at::Tensor& output,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t threads) {
  const int64_t rows = source.size(0);
  const int64_t cols = source.size(1);
  const uint8_t* src = reinterpret_cast<const uint8_t*>(source.const_data_ptr());

  if (output.scalar_type() == at::kBFloat16) {
    c10::BFloat16* dst = output.data_ptr<c10::BFloat16>();
#if GPTQMODEL_FLOATX_X86 && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
      const int64_t grain = std::max<int64_t>(1, rows / clamped_threads(threads));
      at::parallel_for(0, rows, grain, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          if (scale_mode != ScaleMode::kNone && spec.layout == ScaleLayout2D::kBlock && spec.col_repeat >= 16) {
            dequantize_fp8_row_avx2_bf16_block_scale(
                src + row * cols,
                dst + row * cols,
                cols,
                table,
                spec,
                scale_mode,
                row);
          } else {
            dequantize_fp8_row_avx2_bf16(
                src + row * cols,
                dst + row * cols,
                cols,
                table,
                spec,
                scale_mode,
                row);
          }
        }
      });
      return;
    }
#endif
    dequantize_fp8_scalar(src, dst, rows, cols, table, spec, scale_mode, threads);
    return;
  }

  c10::Half* dst = output.data_ptr<c10::Half>();
#if GPTQMODEL_FLOATX_X86 && (defined(__GNUC__) || defined(__clang__))
  if (cpu_supports_avx2() && cpu_supports_f16c()) {
    const int64_t grain = std::max<int64_t>(1, rows / clamped_threads(threads));
    at::parallel_for(0, rows, grain, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        if (scale_mode != ScaleMode::kNone && spec.layout == ScaleLayout2D::kBlock && spec.col_repeat >= 16) {
          dequantize_fp8_row_avx2_fp16_block_scale(
              src + row * cols,
              dst + row * cols,
              cols,
              table,
              spec,
              scale_mode,
              row);
        } else {
          dequantize_fp8_row_avx2_fp16(
              src + row * cols,
              dst + row * cols,
              cols,
              table,
              spec,
              scale_mode,
              row);
        }
      }
    });
    return;
  }
#endif
  dequantize_fp8_scalar(src, dst, rows, cols, table, spec, scale_mode, threads);
}

template <typename T>
void dequantize_fp4_scalar(
    const uint8_t* src,
    T* dst,
    int64_t rows,
    int64_t packed_cols,
    const std::array<float, 16>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t threads) {
  const int64_t cols = packed_cols * 2;
  const int64_t grain = std::max<int64_t>(1, rows / clamped_threads(threads));
  at::parallel_for(0, rows, grain, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const uint8_t* src_row = src + row * packed_cols;
      T* dst_row = dst + row * cols;
      for (int64_t packed_col = 0; packed_col < packed_cols; ++packed_col) {
        const uint8_t byte = src_row[packed_col];
        const int64_t col = packed_col * 2;
        const float values[2] = {
            table[byte & 0x0F],
            table[(byte >> 4) & 0x0F],
        };
        apply_scale_and_store_scalar(dst_row + col, values, 2, scale_mode, spec, row, col);
      }
    }
  });
}

template <typename T>
void dequantize_fp4_vectorized(
    const uint8_t* src,
    T* dst,
    int64_t rows,
    int64_t packed_cols,
    const std::array<float, 16>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t threads) {
#if GPTQMODEL_FLOATX_X86 && (defined(__GNUC__) || defined(__clang__))
  auto dequantize_fp4_row_avx2_bf16 =
      [&table, &spec, scale_mode](const uint8_t* src_row, c10::BFloat16* dst_row, int64_t cols, int64_t row)
      __attribute__((target("avx2"))) {
        alignas(32) float values[8];
        int64_t col = 0;
        for (; col + 16 <= cols; col += 16) {
          __m256 vec_lo;
          __m256 vec_hi;
          load_fp4x16_to_ps(src_row + (col / 2), table.data(), &vec_lo, &vec_hi);
          apply_scale_and_store_bf16x8(dst_row + col, vec_lo, scale_mode, spec, row, col);
          apply_scale_and_store_bf16x8(dst_row + col + 8, vec_hi, scale_mode, spec, row, col + 8);
        }
        for (; col + 8 <= cols; col += 8) {
          // FP4 unpack stays scalar per nibble, then hands the scaled arithmetic to SIMD.
          for (int64_t lane = 0; lane < 8; ++lane) {
            const int64_t logical_col = col + lane;
            const uint8_t byte = src_row[logical_col / 2];
            const uint8_t nibble = (logical_col & 1)
                ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                : static_cast<uint8_t>(byte & 0x0F);
            values[lane] = table[nibble];
          }
          __m256 vec = _mm256_load_ps(values);
          apply_scale_and_store_bf16x8(dst_row + col, vec, scale_mode, spec, row, col);
        }
        if (col < cols) {
          for (int64_t logical_col = col; logical_col < cols; ++logical_col) {
            const uint8_t byte = src_row[logical_col / 2];
            const uint8_t nibble = (logical_col & 1)
                ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                : static_cast<uint8_t>(byte & 0x0F);
            float value = table[nibble];
            if (scale_mode != ScaleMode::kNone) {
              const float scale = scale_value_2d(spec, row, logical_col);
              value = scale_mode == ScaleMode::kMultiply ? value * scale : value / scale;
            }
            store_scalar(dst_row + logical_col, value);
          }
        }
      };

  auto dequantize_fp4_row_avx2_fp16 =
      [&table, &spec, scale_mode](const uint8_t* src_row, c10::Half* dst_row, int64_t cols, int64_t row)
      __attribute__((target("avx2,f16c"))) {
        alignas(32) float values[8];
        int64_t col = 0;
        for (; col + 16 <= cols; col += 16) {
          __m256 vec_lo;
          __m256 vec_hi;
          load_fp4x16_to_ps(src_row + (col / 2), table.data(), &vec_lo, &vec_hi);
          apply_scale_and_store_fp16x8(dst_row + col, vec_lo, scale_mode, spec, row, col);
          apply_scale_and_store_fp16x8(dst_row + col + 8, vec_hi, scale_mode, spec, row, col + 8);
        }
        for (; col + 8 <= cols; col += 8) {
          // FP4 unpack stays scalar per nibble, then hands the scaled arithmetic to SIMD.
          for (int64_t lane = 0; lane < 8; ++lane) {
            const int64_t logical_col = col + lane;
            const uint8_t byte = src_row[logical_col / 2];
            const uint8_t nibble = (logical_col & 1)
                ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                : static_cast<uint8_t>(byte & 0x0F);
            values[lane] = table[nibble];
          }
          __m256 vec = _mm256_load_ps(values);
          apply_scale_and_store_fp16x8(dst_row + col, vec, scale_mode, spec, row, col);
        }
        if (col < cols) {
          for (int64_t logical_col = col; logical_col < cols; ++logical_col) {
            const uint8_t byte = src_row[logical_col / 2];
            const uint8_t nibble = (logical_col & 1)
                ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                : static_cast<uint8_t>(byte & 0x0F);
            float value = table[nibble];
            if (scale_mode != ScaleMode::kNone) {
              const float scale = scale_value_2d(spec, row, logical_col);
              value = scale_mode == ScaleMode::kMultiply ? value * scale : value / scale;
            }
            store_scalar(dst_row + logical_col, value);
          }
        }
      };

  if (cpu_supports_avx2()) {
    const int64_t cols = packed_cols * 2;
    const int64_t grain = std::max<int64_t>(1, rows / clamped_threads(threads));
    if constexpr (std::is_same_v<T, c10::BFloat16>) {
      at::parallel_for(0, rows, grain, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          dequantize_fp4_row_avx2_bf16(src + row * packed_cols, dst + row * cols, cols, row);
        }
      });
      return;
    }
    if constexpr (std::is_same_v<T, c10::Half>) {
      if (cpu_supports_f16c()) {
        at::parallel_for(0, rows, grain, [&](int64_t begin, int64_t end) {
          for (int64_t row = begin; row < end; ++row) {
            dequantize_fp4_row_avx2_fp16(src + row * packed_cols, dst + row * cols, cols, row);
          }
        });
        return;
      }
    }
  }
#endif
  dequantize_fp4_scalar(src, dst, rows, packed_cols, table, spec, scale_mode, threads);
}

at::Tensor empty_target_like(const at::Tensor& source, TargetKind target_kind) {
  auto options = at::TensorOptions().device(at::kCPU);
  return at::empty_like(
      source,
      options.dtype(target_kind == TargetKind::kBFloat16 ? at::kBFloat16 : at::kHalf));
}

at::Tensor empty_target_for_fp4(const at::Tensor& source, TargetKind target_kind) {
  auto sizes = source.sizes().vec();
  TORCH_CHECK(!sizes.empty(), "FP4 source tensor must have at least one dimension");
  sizes.back() *= 2;
  return at::empty(
      sizes,
      at::TensorOptions().device(at::kCPU).dtype(
          target_kind == TargetKind::kBFloat16 ? at::kBFloat16 : at::kHalf));
}

at::Tensor dequantize_fp8_cpu(
    const at::Tensor& source,
    const c10::optional<at::Tensor>& scale,
    int64_t scale_mode_value,
    int64_t axis,
    bool axis_is_none,
    int64_t target_dtype_value,
    int64_t format_value,
    int64_t threads) {
  TORCH_CHECK(source.device().is_cpu(), "FP8 source tensor must reside on CPU");
  TORCH_CHECK(source.ndimension() == 1 || source.ndimension() == 2, "FP8 fast path only supports 1D or 2D tensors");
  TORCH_CHECK(source.element_size() == 1, "FP8 fast path expects one byte per source element");

  const ScaleMode scale_mode = static_cast<ScaleMode>(scale_mode_value);
  const TargetKind target_kind = static_cast<TargetKind>(target_dtype_value);
  const Fp8Format format = static_cast<Fp8Format>(format_value);

  at::Tensor src = source.contiguous();
  at::Tensor output = empty_target_like(src, target_kind);
  const auto& table = fp8_table(format, target_kind);

  if (src.ndimension() == 1) {
    const int64_t length = src.size(0);
    const ScaleSpec1D spec = make_scale_spec_1d(scale, length, axis, axis_is_none);
    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.const_data_ptr());
    if (output.scalar_type() == at::kBFloat16) {
      c10::BFloat16* dst = output.data_ptr<c10::BFloat16>();
      const int64_t grain = std::max<int64_t>(1, length / clamped_threads(threads));
      at::parallel_for(0, length, grain, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          apply_scale_and_store_scalar_1d(
              dst,
              table[src_ptr[idx]],
              scale_mode,
              spec,
              idx);
        }
      });
    } else {
      c10::Half* dst = output.data_ptr<c10::Half>();
      const int64_t grain = std::max<int64_t>(1, length / clamped_threads(threads));
      at::parallel_for(0, length, grain, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          apply_scale_and_store_scalar_1d(
              dst,
              table[src_ptr[idx]],
              scale_mode,
              spec,
              idx);
        }
      });
    }
    return output;
  }

  const ScaleSpec2D spec = make_scale_spec_2d(scale, src.size(0), src.size(1), axis, axis_is_none);
  dequantize_fp8_2d(src, output, table, spec, scale_mode, threads);
  return output;
}

at::Tensor dequantize_fp4_cpu(
    const at::Tensor& source,
    const c10::optional<at::Tensor>& scale,
    int64_t scale_mode_value,
    int64_t axis,
    bool axis_is_none,
    int64_t target_dtype_value,
    int64_t threads) {
  TORCH_CHECK(source.device().is_cpu(), "FP4 source tensor must reside on CPU");
  TORCH_CHECK(source.ndimension() == 1 || source.ndimension() == 2, "FP4 fast path only supports 1D or 2D tensors");
  TORCH_CHECK(source.element_size() == 1, "FP4 fast path expects packed one-byte storage");

  const ScaleMode scale_mode = static_cast<ScaleMode>(scale_mode_value);
  const TargetKind target_kind = static_cast<TargetKind>(target_dtype_value);

  at::Tensor src = source.contiguous();
  at::Tensor output = empty_target_for_fp4(src, target_kind);
  const auto& table = fp4_table(target_kind);

  if (src.ndimension() == 1) {
    const int64_t packed = src.size(0);
    const int64_t length = packed * 2;
    const ScaleSpec1D spec = make_scale_spec_1d(scale, length, axis, axis_is_none);
    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.const_data_ptr());
    if (output.scalar_type() == at::kBFloat16) {
      c10::BFloat16* dst = output.data_ptr<c10::BFloat16>();
      const int64_t grain = std::max<int64_t>(1, length / clamped_threads(threads));
      at::parallel_for(0, length, grain, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          const uint8_t byte = src_ptr[idx / 2];
          const uint8_t nibble = (idx & 1) ? static_cast<uint8_t>((byte >> 4) & 0x0F) : static_cast<uint8_t>(byte & 0x0F);
          apply_scale_and_store_scalar_1d(dst, table[nibble], scale_mode, spec, idx);
        }
      });
    } else {
      c10::Half* dst = output.data_ptr<c10::Half>();
      const int64_t grain = std::max<int64_t>(1, length / clamped_threads(threads));
      at::parallel_for(0, length, grain, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          const uint8_t byte = src_ptr[idx / 2];
          const uint8_t nibble = (idx & 1) ? static_cast<uint8_t>((byte >> 4) & 0x0F) : static_cast<uint8_t>(byte & 0x0F);
          apply_scale_and_store_scalar_1d(dst, table[nibble], scale_mode, spec, idx);
        }
      });
    }
    return output;
  }

  const int64_t rows = src.size(0);
  const int64_t packed_cols = src.size(1);
  const ScaleSpec2D spec = make_scale_spec_2d(scale, rows, packed_cols * 2, axis, axis_is_none);
  const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.const_data_ptr());
  if (output.scalar_type() == at::kBFloat16) {
    c10::BFloat16* dst = output.data_ptr<c10::BFloat16>();
    dequantize_fp4_vectorized(src_ptr, dst, rows, packed_cols, table, spec, scale_mode, threads);
  } else {
    c10::Half* dst = output.data_ptr<c10::Half>();
    dequantize_fp4_vectorized(src_ptr, dst, rows, packed_cols, table, spec, scale_mode, threads);
  }
  return output;
}

} // namespace gptqmodel_floatx

TORCH_LIBRARY(gptqmodel_floatx, m) {
  m.def(
      "dequantize_fp8_cpu(Tensor src, Tensor? scale, int scale_mode, int axis, bool axis_is_none, int target_dtype, int format_code, int threads) -> Tensor");
  m.def(
      "dequantize_fp4_cpu(Tensor src, Tensor? scale, int scale_mode, int axis, bool axis_is_none, int target_dtype, int threads) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_floatx, CPU, m) {
  m.impl("dequantize_fp8_cpu", TORCH_FN(gptqmodel_floatx::dequantize_fp8_cpu));
  m.impl("dequantize_fp4_cpu", TORCH_FN(gptqmodel_floatx::dequantize_fp4_cpu));
}
