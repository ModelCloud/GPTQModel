// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Fused YAQA corrected-tile construction for the factored CPU recurrence.
//
// Mirrors gptqmodel_ext/qvq/qvq_yaqa_cuda.cu for the host path: given
// precomputed left_transformed_error and right_transformed_error matrices, and
// the strict lower-triangular output feedback, compute the corrected tile
// stack for one anti-diagonal:
//
//   corrected = source + bias
//               + left(16, K) @ output_feedback(K, 16)
//               + left(16, 16)
//               + right(16, 16)
//
// where K shrinks as the anti-diagonal walks from bottom-right toward top-left.

#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#define QVQ_YAQA_CPU_X86 1
#include <immintrin.h>
#else
#define QVQ_YAQA_CPU_X86 0
#endif

namespace qvq_cpu {
namespace {

constexpr int64_t kTile = 16;

#if QVQ_YAQA_CPU_X86
__attribute__((constructor))
static void qvq_yaqa_cpu_init_cpu_features() {
  __builtin_cpu_init();
}

inline bool cpu_has_avx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
         __builtin_cpu_supports("avx512dq");
}

inline bool cpu_has_fma() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

at::Tensor qvq_yaqa_feedback_cpu(
    const at::Tensor& source,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& output_feedback,
    int64_t first_input_block,
    int64_t first_output_block,
    int64_t count,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(source.device().is_cpu(), "YAQA source must be CPU");
  TORCH_CHECK(
      source.scalar_type() == at::kFloat && source.dim() == 2 && source.is_contiguous(),
      "YAQA source must be contiguous FP32 [input, output]");
  for (const auto* tensor : {&left, &right}) {
    TORCH_CHECK(
        tensor->device() == source.device() && tensor->scalar_type() == at::kFloat &&
            tensor->sizes() == source.sizes() && tensor->is_contiguous(),
        "YAQA transformed errors must match the contiguous FP32 source");
  }
  TORCH_CHECK(
      output_feedback.device() == source.device() && output_feedback.scalar_type() == at::kFloat &&
          output_feedback.dim() == 2 && output_feedback.size(0) == source.size(1) &&
          output_feedback.size(1) == source.size(1) && output_feedback.is_contiguous(),
      "YAQA output feedback must be contiguous FP32 [output, output]");
  TORCH_CHECK(
      source.size(0) % kTile == 0 && source.size(1) % kTile == 0,
      "YAQA fused feedback requires dimensions divisible by 16");
  TORCH_CHECK(
      count >= 1 && first_input_block >= 0 && first_output_block >= count - 1 &&
          first_input_block + count <= source.size(0) / kTile &&
          first_output_block < source.size(1) / kTile,
      "YAQA anti-diagonal geometry is invalid");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->device() == source.device() && bias->scalar_type() == at::kFloat &&
            bias->sizes() == source.sizes() && bias->is_contiguous(),
        "YAQA bias must match the contiguous FP32 source");
  }

  const int64_t in_features = source.size(0);
  const int64_t out_features = source.size(1);

  at::Tensor corrected = at::empty({count, kTile, kTile}, source.options());
  at::Tensor cross_tile = at::empty({kTile, kTile}, source.options());

  const float* __restrict__ source_ptr = source.const_data_ptr<float>();
  const float* __restrict__ left_ptr = left.const_data_ptr<float>();
  const float* __restrict__ right_ptr = right.const_data_ptr<float>();
  const float* __restrict__ bias_ptr = bias.has_value() ? bias->const_data_ptr<float>() : nullptr;
  float* __restrict__ corrected_ptr = corrected.mutable_data_ptr<float>();

  for (int64_t tile = 0; tile < count; ++tile) {
    const int64_t input_start = (first_input_block + tile) * kTile;
    const int64_t output_start = (first_output_block - tile) * kTile;
    const int64_t k = out_features - output_start;

    at::Tensor left_slice = left.narrow(0, input_start, kTile).narrow(1, output_start, k);
    at::Tensor output_fb_slice =
        output_feedback.narrow(0, output_start, k).narrow(1, output_start, kTile);

    at::mm_out(cross_tile, left_slice, output_fb_slice);
    const float* __restrict__ cross_ptr = cross_tile.const_data_ptr<float>();

    for (int64_t row = 0; row < kTile; ++row) {
      const int64_t matrix_row = input_start + row;
      for (int64_t col = 0; col < kTile; ++col) {
        const int64_t matrix_col = output_start + col;
        const int64_t src_index = matrix_row * out_features + matrix_col;
        float value = source_ptr[src_index];
        if (bias_ptr != nullptr) {
          value += bias_ptr[src_index];
        }
        value += cross_ptr[row * kTile + col];
        value += left_ptr[src_index];
        value += right_ptr[src_index];
        corrected_ptr[(tile * kTile + row) * kTile + col] = value;
      }
    }
  }

  return corrected;
}

// Fused in-place factored cache update, mirroring the batched cuBLAS update in
// gptqmodel_ext/qvq/qvq_yaqa_cuda.cu. For every tile of one anti-diagonal:
//
//   left[:, o0:o0+16]  -= input_feedback[i0:i0+16, :].T @ Q
//   right[i0:i0+16, :] -= Q @ output_feedback[o0:o0+16, :]
//
// Both destinations are disjoint across the anti-diagonal, so the rank-16
// updates need no temporaries, no indexed scatters, and no atomics.

void left_update_rows_scalar(
    float* left_columns,
    const float* input_feedback_block,
    const float* q,
    int64_t in_features,
    int64_t out_features,
    int64_t row_begin,
    int64_t row_end) {
  for (int64_t i = row_begin; i < row_end; ++i) {
    float* __restrict__ dst = left_columns + i * out_features;
    // Accumulate the rank-16 update on its own before touching the cache tile:
    // the update is far smaller than the running cache value, so folding it in
    // once keeps the rounding error at the scale of the update.
    float update[kTile] = {0.0f};
    for (int64_t k = 0; k < kTile; ++k) {
      const float scale = input_feedback_block[k * in_features + i];
      const float* __restrict__ q_row = q + k * kTile;
      for (int64_t col = 0; col < kTile; ++col) {
        update[col] += scale * q_row[col];
      }
    }
    for (int64_t col = 0; col < kTile; ++col) {
      dst[col] -= update[col];
    }
  }
}

void right_update_columns_scalar(
    float* right_rows,
    const float* output_feedback_block,
    const float* q,
    int64_t out_features,
    int64_t col_begin,
    int64_t col_end) {
  for (int64_t row = 0; row < kTile; ++row) {
    float* __restrict__ dst = right_rows + row * out_features;
    const float* __restrict__ q_row = q + row * kTile;
    for (int64_t j = col_begin; j < col_end; ++j) {
      float update = 0.0f;
      for (int64_t c = 0; c < kTile; ++c) {
        update += q_row[c] * output_feedback_block[c * out_features + j];
      }
      dst[j] -= update;
    }
  }
}

#if QVQ_YAQA_CPU_X86
__attribute__((target("avx512f,avx512vl,avx512dq")))
void left_update_rows_avx512(
    float* left_columns,
    const float* input_feedback_block,
    const float* q,
    int64_t in_features,
    int64_t out_features,
    int64_t row_begin,
    int64_t row_end) {
  // One 16-wide vector holds a full destination tile row, so the whole rank-16
  // update is 16 fused multiply-adds against broadcast feedback, folded into
  // the cache tile once at the end to keep rounding at the update's scale.
  __m512 q_rows[kTile];
  for (int64_t k = 0; k < kTile; ++k) {
    q_rows[k] = _mm512_loadu_ps(q + k * kTile);
  }
  for (int64_t i = row_begin; i < row_end; ++i) {
    float* dst = left_columns + i * out_features;
    __m512 update = _mm512_setzero_ps();
    for (int64_t k = 0; k < kTile; ++k) {
      update = _mm512_fmadd_ps(
          _mm512_set1_ps(input_feedback_block[k * in_features + i]), q_rows[k], update);
    }
    _mm512_storeu_ps(dst, _mm512_sub_ps(_mm512_loadu_ps(dst), update));
  }
}

__attribute__((target("avx512f,avx512vl,avx512dq")))
void right_update_columns_avx512(
    float* right_rows,
    const float* output_feedback_block,
    const float* q,
    int64_t out_features,
    int64_t col_begin,
    int64_t col_end) {
  for (int64_t row = 0; row < kTile; ++row) {
    float* dst = right_rows + row * out_features;
    const float* q_row = q + row * kTile;
    int64_t j = col_begin;
    for (; j + 16 <= col_end; j += 16) {
      __m512 update = _mm512_setzero_ps();
      for (int64_t c = 0; c < kTile; ++c) {
        update = _mm512_fmadd_ps(
            _mm512_set1_ps(q_row[c]),
            _mm512_loadu_ps(output_feedback_block + c * out_features + j),
            update);
      }
      _mm512_storeu_ps(dst + j, _mm512_sub_ps(_mm512_loadu_ps(dst + j), update));
    }
    for (; j < col_end; ++j) {
      float update = 0.0f;
      for (int64_t c = 0; c < kTile; ++c) {
        update += q_row[c] * output_feedback_block[c * out_features + j];
      }
      dst[j] -= update;
    }
  }
}

__attribute__((target("avx2,fma")))
void left_update_rows_avx2(
    float* left_columns,
    const float* input_feedback_block,
    const float* q,
    int64_t in_features,
    int64_t out_features,
    int64_t row_begin,
    int64_t row_end) {
  __m256 q_low[kTile];
  __m256 q_high[kTile];
  for (int64_t k = 0; k < kTile; ++k) {
    q_low[k] = _mm256_loadu_ps(q + k * kTile);
    q_high[k] = _mm256_loadu_ps(q + k * kTile + 8);
  }
  for (int64_t i = row_begin; i < row_end; ++i) {
    float* dst = left_columns + i * out_features;
    __m256 update_low = _mm256_setzero_ps();
    __m256 update_high = _mm256_setzero_ps();
    for (int64_t k = 0; k < kTile; ++k) {
      const __m256 scale = _mm256_set1_ps(input_feedback_block[k * in_features + i]);
      update_low = _mm256_fmadd_ps(scale, q_low[k], update_low);
      update_high = _mm256_fmadd_ps(scale, q_high[k], update_high);
    }
    _mm256_storeu_ps(dst, _mm256_sub_ps(_mm256_loadu_ps(dst), update_low));
    _mm256_storeu_ps(dst + 8, _mm256_sub_ps(_mm256_loadu_ps(dst + 8), update_high));
  }
}

__attribute__((target("avx2,fma")))
void right_update_columns_avx2(
    float* right_rows,
    const float* output_feedback_block,
    const float* q,
    int64_t out_features,
    int64_t col_begin,
    int64_t col_end) {
  for (int64_t row = 0; row < kTile; ++row) {
    float* dst = right_rows + row * out_features;
    const float* q_row = q + row * kTile;
    int64_t j = col_begin;
    for (; j + 8 <= col_end; j += 8) {
      __m256 update = _mm256_setzero_ps();
      for (int64_t c = 0; c < kTile; ++c) {
        update = _mm256_fmadd_ps(
            _mm256_set1_ps(q_row[c]),
            _mm256_loadu_ps(output_feedback_block + c * out_features + j),
            update);
      }
      _mm256_storeu_ps(dst + j, _mm256_sub_ps(_mm256_loadu_ps(dst + j), update));
    }
    for (; j < col_end; ++j) {
      float update = 0.0f;
      for (int64_t c = 0; c < kTile; ++c) {
        update += q_row[c] * output_feedback_block[c * out_features + j];
      }
      dst[j] -= update;
    }
  }
}
#endif  // QVQ_YAQA_CPU_X86

void qvq_yaqa_feedback_update_cpu(
    at::Tensor& left,
    at::Tensor& right,
    const at::Tensor& input_feedback,
    const at::Tensor& output_feedback,
    const at::Tensor& reconstructed,
    int64_t first_input_block,
    int64_t first_output_block,
    int64_t count) {
  TORCH_CHECK(
      left.device().is_cpu() && left.scalar_type() == at::kFloat && left.dim() == 2 &&
          left.is_contiguous(),
      "YAQA left cache must be contiguous CPU FP32 [input, output]");
  TORCH_CHECK(
      right.device() == left.device() && right.scalar_type() == at::kFloat &&
          right.sizes() == left.sizes() && right.is_contiguous(),
      "YAQA right cache must match the contiguous FP32 left cache");
  const int64_t in_features = left.size(0);
  const int64_t out_features = left.size(1);
  TORCH_CHECK(
      input_feedback.device() == left.device() && input_feedback.scalar_type() == at::kFloat &&
          input_feedback.sizes() == at::IntArrayRef({in_features, in_features}) &&
          input_feedback.is_contiguous(),
      "YAQA input feedback must be contiguous FP32 [input, input]");
  TORCH_CHECK(
      output_feedback.device() == left.device() && output_feedback.scalar_type() == at::kFloat &&
          output_feedback.sizes() == at::IntArrayRef({out_features, out_features}) &&
          output_feedback.is_contiguous(),
      "YAQA output feedback must be contiguous FP32 [output, output]");
  TORCH_CHECK(
      reconstructed.device() == left.device() && reconstructed.scalar_type() == at::kFloat &&
          reconstructed.dim() == 3 && reconstructed.size(0) == count &&
          reconstructed.size(1) == kTile && reconstructed.size(2) == kTile &&
          reconstructed.is_contiguous(),
      "YAQA reconstruction must be contiguous FP32 [count, 16, 16]");
  TORCH_CHECK(
      in_features % kTile == 0 && out_features % kTile == 0,
      "YAQA fused update requires dimensions divisible by 16");
  TORCH_CHECK(
      count >= 1 && first_input_block >= 0 && first_output_block >= count - 1 &&
          first_input_block + count <= in_features / kTile &&
          first_output_block < out_features / kTile,
      "YAQA update anti-diagonal geometry is invalid");

  float* const left_ptr = left.mutable_data_ptr<float>();
  float* const right_ptr = right.mutable_data_ptr<float>();
  const float* const input_feedback_ptr = input_feedback.const_data_ptr<float>();
  const float* const output_feedback_ptr = output_feedback.const_data_ptr<float>();
  const float* const reconstructed_ptr = reconstructed.const_data_ptr<float>();

#if QVQ_YAQA_CPU_X86
  const bool use_avx512 = cpu_has_avx512();
  const bool use_avx2 = !use_avx512 && cpu_has_fma();
#endif

  // Left update: every (tile, input row) pair owns 16 contiguous destination
  // floats, so the work splits cleanly over rows without synchronization.
  const int64_t left_work = count * in_features;
  const int64_t left_grain = std::max<int64_t>(256, left_work / (4 * at::get_num_threads()));
  at::parallel_for(0, left_work, left_grain, [&](int64_t begin, int64_t end) {
    int64_t index = begin;
    while (index < end) {
      const int64_t tile = index / in_features;
      const int64_t row_begin = index - tile * in_features;
      const int64_t row_end = std::min(end - tile * in_features, in_features);
      const int64_t input_start = (first_input_block + tile) * kTile;
      const int64_t output_start = (first_output_block - tile) * kTile;
      float* left_columns = left_ptr + output_start;
      const float* input_feedback_block = input_feedback_ptr + input_start * in_features;
      const float* q = reconstructed_ptr + tile * kTile * kTile;
#if QVQ_YAQA_CPU_X86
      if (use_avx512) {
        left_update_rows_avx512(
            left_columns, input_feedback_block, q, in_features, out_features, row_begin, row_end);
      } else if (use_avx2) {
        left_update_rows_avx2(
            left_columns, input_feedback_block, q, in_features, out_features, row_begin, row_end);
      } else {
        left_update_rows_scalar(
            left_columns, input_feedback_block, q, in_features, out_features, row_begin, row_end);
      }
#else
      left_update_rows_scalar(
          left_columns, input_feedback_block, q, in_features, out_features, row_begin, row_end);
#endif
      index = tile * in_features + row_end;
    }
  });

  // Right update: every (tile, output column) pair owns 16 destination rows,
  // so the work splits over 16-wide column chunks of the anti-diagonal.
  const int64_t column_chunks = out_features / kTile;
  const int64_t right_work = count * column_chunks;
  const int64_t right_grain = std::max<int64_t>(1, right_work / (4 * at::get_num_threads()));
  at::parallel_for(0, right_work, right_grain, [&](int64_t begin, int64_t end) {
    int64_t index = begin;
    while (index < end) {
      const int64_t tile = index / column_chunks;
      const int64_t chunk_begin = index - tile * column_chunks;
      const int64_t chunk_end = std::min(end - tile * column_chunks, column_chunks);
      const int64_t input_start = (first_input_block + tile) * kTile;
      const int64_t output_start = (first_output_block - tile) * kTile;
      float* right_rows = right_ptr + input_start * out_features;
      const float* output_feedback_block = output_feedback_ptr + output_start * out_features;
      const float* q = reconstructed_ptr + tile * kTile * kTile;
      const int64_t col_begin = chunk_begin * kTile;
      const int64_t col_end = chunk_end * kTile;
#if QVQ_YAQA_CPU_X86
      if (use_avx512) {
        right_update_columns_avx512(
            right_rows, output_feedback_block, q, out_features, col_begin, col_end);
      } else if (use_avx2) {
        right_update_columns_avx2(
            right_rows, output_feedback_block, q, out_features, col_begin, col_end);
      } else {
        right_update_columns_scalar(
            right_rows, output_feedback_block, q, out_features, col_begin, col_end);
      }
#else
      right_update_columns_scalar(
          right_rows, output_feedback_block, q, out_features, col_begin, col_end);
#endif
      index = tile * column_chunks + chunk_end;
    }
  });
}

}  // namespace
}  // namespace qvq_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("yaqa_feedback(Tensor source, Tensor left, Tensor right, Tensor output_feedback, "
        "int first_input_block, int first_output_block, int count, Tensor? bias) -> Tensor");
  m.def("yaqa_feedback_update_(Tensor(a!) left, Tensor(b!) right, Tensor input_feedback, "
        "Tensor output_feedback, Tensor reconstructed, int first_input_block, "
        "int first_output_block, int count) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("yaqa_feedback", qvq_cpu::qvq_yaqa_feedback_cpu);
  m.impl("yaqa_feedback_update_", qvq_cpu::qvq_yaqa_feedback_update_cpu);
}
