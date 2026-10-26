// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/library.h>

#include "vptq.h"

namespace {

at::Tensor vptq_dequantize_dispatch(
    const at::Tensor& indices,
    const at::Tensor& centroids,
    const c10::optional<at::Tensor>& res_centroids,
    const c10::optional<at::Tensor>& outlier_indices,
    const c10::optional<at::Tensor>& outlier_centroids,
    const c10::optional<at::Tensor>& inverse_perm,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& weight_bias,
    int64_t group_size,
    int64_t in_features,
    int64_t out_features,
    int64_t vector_len,
    int64_t outlier_size,
    int64_t outlier_vector_len,
    int64_t num_centroids,
    int64_t num_res_centroids,
    int64_t index_bits,
    int64_t res_bits,
    int64_t norm_dim) {
  return gptqmodel::vptq::vptq_dequantize_cuda(
      indices,
      centroids,
      res_centroids,
      outlier_indices,
      outlier_centroids,
      inverse_perm,
      weight_scale,
      weight_bias,
      group_size,
      in_features,
      out_features,
      vector_len,
      outlier_size,
      outlier_vector_len,
      num_centroids,
      num_res_centroids,
      index_bits,
      res_bits,
      norm_dim);
}

at::Tensor vptq_linear_dispatch(
    const at::Tensor& x,
    const at::Tensor& indices,
    const at::Tensor& centroids,
    const c10::optional<at::Tensor>& res_centroids,
    const c10::optional<at::Tensor>& outlier_indices,
    const c10::optional<at::Tensor>& outlier_centroids,
    const c10::optional<at::Tensor>& inverse_perm,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& weight_bias,
    const c10::optional<at::Tensor>& bias,
    int64_t group_size,
    int64_t in_features,
    int64_t out_features,
    int64_t vector_len,
    int64_t outlier_size,
    int64_t outlier_vector_len,
    int64_t num_centroids,
    int64_t num_res_centroids,
    int64_t index_bits,
    int64_t res_bits,
    int64_t norm_dim) {
  return gptqmodel::vptq::vptq_linear_cuda(
      x,
      indices,
      centroids,
      res_centroids,
      outlier_indices,
      outlier_centroids,
      inverse_perm,
      weight_scale,
      weight_bias,
      bias,
      group_size,
      in_features,
      out_features,
      vector_len,
      outlier_size,
      outlier_vector_len,
      num_centroids,
      num_res_centroids,
      index_bits,
      res_bits,
      norm_dim);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_vptq, m) {
  m.def(
      "vptq_dequantize("
      "Tensor indices, Tensor centroids, Tensor? res_centroids, "
      "Tensor? outlier_indices, Tensor? outlier_centroids, Tensor? inverse_perm, "
      "Tensor? weight_scale, Tensor? weight_bias, "
      "int group_size, int in_features, int out_features, int vector_len, "
      "int outlier_size, int outlier_vector_len, int num_centroids, "
      "int num_res_centroids, int index_bits, int res_bits, int norm_dim"
      ") -> Tensor");
  m.def(
      "vptq_linear("
      "Tensor x, Tensor indices, Tensor centroids, Tensor? res_centroids, "
      "Tensor? outlier_indices, Tensor? outlier_centroids, Tensor? inverse_perm, "
      "Tensor? weight_scale, Tensor? weight_bias, Tensor? bias, "
      "int group_size, int in_features, int out_features, int vector_len, "
      "int outlier_size, int outlier_vector_len, int num_centroids, "
      "int num_res_centroids, int index_bits, int res_bits, int norm_dim"
      ") -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_vptq, CUDA, m) {
  m.impl("vptq_dequantize", &vptq_dequantize_dispatch);
  m.impl("vptq_linear", &vptq_linear_dispatch);
}
