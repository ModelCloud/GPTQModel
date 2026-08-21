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

#include <cstdint>

namespace qvq_cpu {
namespace {

constexpr int64_t kTile = 16;

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

}  // namespace
}  // namespace qvq_cpu

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("yaqa_feedback(Tensor source, Tensor left, Tensor right, Tensor output_feedback, "
        "int first_input_block, int first_output_block, int count, Tensor? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("yaqa_feedback", qvq_cpu::qvq_yaqa_feedback_cpu);
}
