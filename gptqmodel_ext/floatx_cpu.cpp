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

template <typename T>
std::array<float, 256> build_fp8_table() {
  std::array<float, 256> table{};
  for (int value = 0; value < 256; ++value) {
    table[value] = static_cast<float>(T(static_cast<uint8_t>(value), T::from_bits()));
  }
  return table;
}

const std::array<float, 256>& fp8_table(Fp8Format format) {
  static const auto e4m3fn = build_fp8_table<c10::Float8_e4m3fn>();
  static const auto e5m2 = build_fp8_table<c10::Float8_e5m2>();
  static const auto e4m3fnuz = build_fp8_table<c10::Float8_e4m3fnuz>();
  static const auto e5m2fnuz = build_fp8_table<c10::Float8_e5m2fnuz>();
  static const auto e8m0fnu = build_fp8_table<c10::Float8_e8m0fnu>();
  switch (format) {
    case Fp8Format::kE4M3Fn:
      return e4m3fn;
    case Fp8Format::kE5M2:
      return e5m2;
    case Fp8Format::kE4M3FnUz:
      return e4m3fnuz;
    case Fp8Format::kE5M2FnUz:
      return e5m2fnuz;
    case Fp8Format::kE8M0Fnu:
      return e8m0fnu;
  }
  TORCH_CHECK(false, "Unsupported FP8 format code");
}

const std::array<float, 16>& fp4_table() {
  static const std::array<float, 16> table = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  return table;
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
inline bool cpu_supports_avx2() {
  return __builtin_cpu_supports("avx2");
}

inline bool cpu_supports_f16c() {
  return __builtin_cpu_supports("f16c");
}

__attribute__((target("avx2")))
inline __m256 load_fp8x8_to_ps(const uint8_t* src, const float* table) {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
  const __m256i indices = _mm256_cvtepu8_epi32(raw);
  return _mm256_i32gather_ps(table, indices, 4);
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

__attribute__((target("avx2")))
void dequantize_fp8_row_avx2_bf16(
    const uint8_t* src_row,
    c10::BFloat16* dst_row,
    int64_t cols,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t row) {
  alignas(32) float scales[8];
  int64_t col = 0;
  for (; col + 8 <= cols; col += 8) {
    __m256 values = load_fp8x8_to_ps(src_row + col, table.data());
    if (scale_mode == ScaleMode::kNone) {
      store_bf16x8(dst_row + col, values);
      continue;
    }
    _mm256_store_ps(scales, values);
    apply_scale_and_store_scalar(dst_row + col, scales, 8, scale_mode, spec, row, col);
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
void dequantize_fp8_row_avx2_fp16(
    const uint8_t* src_row,
    c10::Half* dst_row,
    int64_t cols,
    const std::array<float, 256>& table,
    const ScaleSpec2D& spec,
    ScaleMode scale_mode,
    int64_t row) {
  alignas(32) float scales[8];
  int64_t col = 0;
  for (; col + 8 <= cols; col += 8) {
    __m256 values = load_fp8x8_to_ps(src_row + col, table.data());
    if (scale_mode == ScaleMode::kNone) {
      store_fp16x8(dst_row + col, values);
      continue;
    }
    _mm256_store_ps(scales, values);
    apply_scale_and_store_scalar(dst_row + col, scales, 8, scale_mode, spec, row, col);
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
        float value = table[src_row[col]];
        if (scale_mode != ScaleMode::kNone) {
          const float scale = scale_value_2d(spec, row, col);
          value = scale_mode == ScaleMode::kMultiply ? value * scale : value / scale;
        }
        store_scalar(dst_row + col, value);
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
          dequantize_fp8_row_avx2_bf16(
              src + row * cols,
              dst + row * cols,
              cols,
              table,
              spec,
              scale_mode,
              row);
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
        dequantize_fp8_row_avx2_fp16(
            src + row * cols,
            dst + row * cols,
            cols,
            table,
            spec,
            scale_mode,
            row);
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
        float lo = table[byte & 0x0F];
        float hi = table[(byte >> 4) & 0x0F];
        if (scale_mode != ScaleMode::kNone) {
          const float scale_lo = scale_value_2d(spec, row, col);
          const float scale_hi = scale_value_2d(spec, row, col + 1);
          lo = scale_mode == ScaleMode::kMultiply ? lo * scale_lo : lo / scale_lo;
          hi = scale_mode == ScaleMode::kMultiply ? hi * scale_hi : hi / scale_hi;
        }
        store_scalar(dst_row + col, lo);
        store_scalar(dst_row + col + 1, hi);
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
        for (; col + 8 <= cols; col += 8) {
          for (int64_t lane = 0; lane < 8; ++lane) {
            const int64_t logical_col = col + lane;
            const uint8_t byte = src_row[logical_col / 2];
            const uint8_t nibble = (logical_col & 1)
                ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                : static_cast<uint8_t>(byte & 0x0F);
            values[lane] = table[nibble];
          }
          __m256 vec = _mm256_load_ps(values);
          if (scale_mode == ScaleMode::kNone) {
            store_bf16x8(dst_row + col, vec);
          } else {
            apply_scale_and_store_scalar(dst_row + col, values, 8, scale_mode, spec, row, col);
          }
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
        for (; col + 8 <= cols; col += 8) {
          for (int64_t lane = 0; lane < 8; ++lane) {
            const int64_t logical_col = col + lane;
            const uint8_t byte = src_row[logical_col / 2];
            const uint8_t nibble = (logical_col & 1)
                ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                : static_cast<uint8_t>(byte & 0x0F);
            values[lane] = table[nibble];
          }
          __m256 vec = _mm256_load_ps(values);
          if (scale_mode == ScaleMode::kNone) {
            store_fp16x8(dst_row + col, vec);
          } else {
            apply_scale_and_store_scalar(dst_row + col, values, 8, scale_mode, spec, row, col);
          }
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
  const auto& table = fp8_table(format);

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
  const auto& table = fp4_table();

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
