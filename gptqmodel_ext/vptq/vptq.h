// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace gptqmodel::vptq {

at::Tensor vptq_dequantize_cuda(
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
    int64_t norm_dim);

at::Tensor vptq_linear_cuda(
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
    int64_t norm_dim);

}  // namespace gptqmodel::vptq
