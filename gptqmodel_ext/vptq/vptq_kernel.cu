// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// Adapted from https://github.com/microsoft/VPTQ

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <type_traits>

#include "vptq.h"

namespace gptqmodel::vptq {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t* p, int64_t i) {
  return static_cast<float>(p[i]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t rounded(float v) {
  return static_cast<scalar_t>(v);
}

template <typename weight_t>
__device__ __forceinline__ float reconstructed_weight(
    int row,
    int channel,
    const int32_t* indices,
    const weight_t* centroids,
    const weight_t* res_centroids,
    const int32_t* outlier_indices,
    const weight_t* outlier_centroids,
    const int32_t* inverse_perm,
    const weight_t* weight_scale,
    const weight_t* weight_bias,
    int group_size,
    int in_features,
    int out_features,
    int vector_len,
    int outlier_size,
    int outlier_vector_len,
    int num_centroids,
    int num_res_centroids,
    int index_bits,
    int res_bits,
    int norm_dim,
    int packed_words,
    int outlier_num_indices,
    int num_indices,
    bool has_residual,
    bool has_outlier,
    bool has_perm,
    bool has_norm) {
  const int qchannel = has_perm ? inverse_perm[channel] : channel;
  const int qrow = row;
  float value = 0.0f;
  if (has_outlier && qchannel < outlier_size) {
    const int vector = qrow / outlier_vector_len;
    const int component = qrow % outlier_vector_len;
    const int code = outlier_indices[vector * outlier_size + qchannel];
    value = load_float(outlier_centroids, code * outlier_vector_len + component);
  } else {
    const int main_channel = qchannel - (has_outlier ? outlier_size : 0);
    const int group = main_channel / group_size;
    const int within_group = main_channel - group * group_size;
    const int vector = qrow / vector_len;
    const int component = qrow % vector_len;
    const int64_t packed_row = (static_cast<int64_t>(group) * num_indices + vector) * packed_words;
    const int bits_per_index = index_bits + res_bits;
    const int bit = within_group * bits_per_index;
    const int word = bit >> 5;
    const int shift = bit & 31;
    const uint64_t lo = static_cast<uint32_t>(indices[packed_row + word]);
    const uint64_t hi = word + 1 < packed_words
        ? static_cast<uint32_t>(indices[packed_row + word + 1])
        : 0ULL;
    const uint64_t merged = (lo | (hi << 32)) >> shift;
    const uint32_t mask = bits_per_index == 32 ? 0xffffffffu : ((1u << bits_per_index) - 1u);
    const uint32_t packed_index = static_cast<uint32_t>(merged) & mask;
    const uint32_t main_index = packed_index & ((1u << index_bits) - 1u);
    value = load_float(centroids, (static_cast<int64_t>(group) * num_centroids + main_index) * vector_len + component);
    if (has_residual) {
      const uint32_t residual_index = (packed_index >> index_bits) & ((1u << res_bits) - 1u);
      const float residual = load_float(res_centroids, (static_cast<int64_t>(group) * num_res_centroids + residual_index) * vector_len + component);
      value = static_cast<float>(rounded<weight_t>(value + residual));
    }
  }

  if (has_norm) {
    const int norm_index = norm_dim == 0 ? channel : row;
    const float scale = load_float(weight_scale, norm_index);
    const float bias = load_float(weight_bias, norm_index);
    value = static_cast<float>(rounded<weight_t>(value * scale));
    value = static_cast<float>(rounded<weight_t>(value + bias));
  }
  return value;
}

template <typename scalar_t, typename weight_t>
__global__ void vptq_dequantize_kernel(
    scalar_t* output,
    const int32_t* indices,
    const weight_t* centroids,
    const weight_t* res_centroids,
    const int32_t* outlier_indices,
    const weight_t* outlier_centroids,
    const int32_t* inverse_perm,
    const weight_t* weight_scale,
    const weight_t* weight_bias,
    int group_size,
    int in_features,
    int out_features,
    int vector_len,
    int outlier_size,
    int outlier_vector_len,
    int num_centroids,
    int num_res_centroids,
    int index_bits,
    int res_bits,
    int norm_dim,
    int packed_words,
    int outlier_num_indices,
    int num_indices,
    bool has_residual,
    bool has_outlier,
    bool has_perm,
    bool has_norm) {
  const int64_t total = static_cast<int64_t>(out_features) * in_features;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int row = linear / in_features;
    const int channel = linear - static_cast<int64_t>(row) * in_features;
    const float w = reconstructed_weight(
        row, channel, indices, centroids, res_centroids, outlier_indices,
        outlier_centroids, inverse_perm, weight_scale, weight_bias,
        group_size, in_features, out_features, vector_len, outlier_size,
        outlier_vector_len, num_centroids, num_res_centroids, index_bits,
        res_bits, norm_dim, packed_words, outlier_num_indices, num_indices,
        has_residual, has_outlier, has_perm, has_norm);
    output[linear] = rounded<scalar_t>(w);
  }
}

template <typename scalar_t, typename weight_t>
__global__ void vptq_linear_kernel(
    const scalar_t* x,
    scalar_t* output,
    const int32_t* indices,
    const weight_t* centroids,
    const weight_t* res_centroids,
    const int32_t* outlier_indices,
    const weight_t* outlier_centroids,
    const int32_t* inverse_perm,
    const weight_t* weight_scale,
    const weight_t* weight_bias,
    const scalar_t* bias,
    int batch,
    int group_size,
    int in_features,
    int out_features,
    int vector_len,
    int outlier_size,
    int outlier_vector_len,
    int num_centroids,
    int num_res_centroids,
    int index_bits,
    int res_bits,
    int norm_dim,
    int packed_words,
    int outlier_num_indices,
    int num_indices,
    bool has_residual,
    bool has_outlier,
    bool has_perm,
    bool has_norm,
    bool has_bias) {
  const int linear = blockIdx.x;
  const int b = linear / out_features;
  const int row = linear - b * out_features;
  // BF16's short mantissa makes output rounding more sensitive to accumulation
  // order. Use FP64 accumulation there; FP16 and FP32 stay on the faster FP32
  // path.
  using acc_t = std::conditional_t<std::is_same_v<scalar_t, c10::BFloat16>, double, float>;
  acc_t sum = 0.0;
  for (int channel = threadIdx.x; channel < in_features; channel += blockDim.x) {
      const float w = reconstructed_weight(
          row, channel, indices, centroids, res_centroids, outlier_indices,
          outlier_centroids, inverse_perm, weight_scale, weight_bias,
          group_size, in_features, out_features, vector_len, outlier_size,
          outlier_vector_len, num_centroids, num_res_centroids, index_bits,
          res_bits, norm_dim, packed_words, outlier_num_indices, num_indices,
          has_residual, has_outlier, has_perm, has_norm);
      if constexpr (std::is_same_v<acc_t, double>) {
        sum = fma(
            static_cast<double>(load_float(x, static_cast<int64_t>(b) * in_features + channel)),
            static_cast<double>(w),
            sum);
      } else {
        sum = fmaf(load_float(x, static_cast<int64_t>(b) * in_features + channel), w, sum);
      }
  }
  // Reduce within each warp, then combine the warp totals. Bias is added
  // after the dot product, matching the reference linear operation.
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ acc_t warp_sums[kThreads / 32];
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    acc_t total = threadIdx.x < (kThreads / 32) ? warp_sums[threadIdx.x] : 0.0;
    for (int offset = 16; offset > 0; offset >>= 1)
      total += __shfl_down_sync(0xffffffff, total, offset);
    if (threadIdx.x == 0) {
      if (has_bias) total += load_float(bias, row);
      output[linear] = rounded<scalar_t>(static_cast<float>(total));
    }
  }
}

// Follow VPTQ's GEMV mapping: one CTA computes a whole output code vector,
// with each thread traversing input channels and accumulating all components.
// This decodes each packed index once for the vector instead of once per output.
template <typename scalar_t, typename weight_t>
__global__ void vptq_linear_vec4_kernel(
    const scalar_t* x,
    scalar_t* output,
    const int32_t* indices,
    const weight_t* centroids,
    const weight_t* res_centroids,
    const int32_t* inverse_perm,
    const weight_t* weight_scale,
    const weight_t* weight_bias,
    const scalar_t* bias,
    int batch,
    int group_size,
    int in_features,
    int out_features,
    int outlier_size,
    int num_centroids,
    int num_res_centroids,
    int index_bits,
    int res_bits,
    int packed_words,
    int num_indices,
    int norm_dim,
    bool has_residual,
    bool has_perm,
    bool has_norm,
    bool has_bias) {
  const int b = blockIdx.y;
  const int first_row = blockIdx.x * 4;
  const int vector = first_row / 4;
  using acc_t = float;
  acc_t sums[4] = {0.0, 0.0, 0.0, 0.0};

  for (int channel = threadIdx.x; channel < in_features; channel += blockDim.x) {
    const int qchannel = has_perm ? inverse_perm[channel] : channel;
    if (qchannel < outlier_size) continue;  // Generic kernel handles outliers.
    const int main_channel = qchannel - outlier_size;
    const int group = main_channel / group_size;
    const int within_group = main_channel - group * group_size;
    const int64_t packed_row =
        (static_cast<int64_t>(group) * num_indices + vector) * packed_words;
    const int bits_per_index = index_bits + res_bits;
    const int bit = within_group * bits_per_index;
    const int word = bit >> 5;
    const int shift = bit & 31;
    const uint64_t lo = static_cast<uint32_t>(indices[packed_row + word]);
    const uint64_t hi = word + 1 < packed_words
        ? static_cast<uint32_t>(indices[packed_row + word + 1])
        : 0ULL;
    const uint32_t mask = bits_per_index == 32
        ? 0xffffffffu
        : ((1u << bits_per_index) - 1u);
    const uint32_t packed_index = static_cast<uint32_t>((lo | (hi << 32)) >> shift) & mask;
    const uint32_t main_index = packed_index & ((1u << index_bits) - 1u);
    const uint32_t residual_index = has_residual
        ? ((packed_index >> index_bits) & ((1u << res_bits) - 1u))
        : 0;
    const float activation = load_float(x, static_cast<int64_t>(b) * in_features + channel);

#pragma unroll
    for (int component = 0; component < 4; ++component) {
      const int row = first_row + component;
      if (row >= out_features) continue;
      float weight = load_float(centroids,
          (static_cast<int64_t>(group) * num_centroids + main_index) * 4 + component);
      if (has_residual) {
        const float residual = load_float(res_centroids,
            (static_cast<int64_t>(group) * num_res_centroids + residual_index) * 4 + component);
        weight = static_cast<float>(rounded<weight_t>(weight + residual));
      }
      if (has_norm) {
        const int norm_index = norm_dim == 0 ? channel : row;
        weight = static_cast<float>(rounded<weight_t>(
            weight * load_float(weight_scale, norm_index)));
        weight = static_cast<float>(rounded<weight_t>(
            weight + load_float(weight_bias, norm_index)));
      }
      sums[component] = fmaf(activation, weight, sums[component]);
    }
  }

  __shared__ acc_t warp_sums[4][kThreads / 32];
#pragma unroll
  for (int component = 0; component < 4; ++component) {
    acc_t sum = sums[component];
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((threadIdx.x & 31) == 0)
      warp_sums[component][threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      float total = threadIdx.x < (kThreads / 32)
          ? warp_sums[component][threadIdx.x]
          : 0.0;
      for (int offset = 16; offset > 0; offset >>= 1)
        total += __shfl_down_sync(0xffffffff, total, offset);
      const int row = first_row + component;
      if (threadIdx.x == 0 && row < out_features) {
        if (has_bias) total += load_float(bias, row);
        output[static_cast<int64_t>(b) * out_features + row] =
            rounded<scalar_t>(static_cast<float>(total));
      }
    }
  }
}

template <typename scalar_t, typename weight_t>
void launch_dequantize(
    at::Tensor& output, const at::Tensor& indices, const at::Tensor& centroids,
    const c10::optional<at::Tensor>& res_centroids,
    const c10::optional<at::Tensor>& outlier_indices,
    const c10::optional<at::Tensor>& outlier_centroids,
    const c10::optional<at::Tensor>& inverse_perm,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& weight_bias,
    int64_t group_size, int64_t in_features, int64_t out_features,
    int64_t vector_len, int64_t outlier_size, int64_t outlier_vector_len,
    int64_t num_centroids, int64_t num_res_centroids, int64_t index_bits,
    int64_t res_bits, int64_t norm_dim) {
  const int64_t n = in_features * out_features;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  vptq_dequantize_kernel<scalar_t, weight_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<scalar_t>(), indices.data_ptr<int32_t>(),
      centroids.data_ptr<weight_t>(), res_centroids.has_value() ? res_centroids->data_ptr<weight_t>() : nullptr,
      outlier_indices.has_value() ? outlier_indices->data_ptr<int32_t>() : nullptr,
      outlier_centroids.has_value() ? outlier_centroids->data_ptr<weight_t>() : nullptr,
      inverse_perm.has_value() ? inverse_perm->data_ptr<int32_t>() : nullptr,
      weight_scale.has_value() ? weight_scale->data_ptr<weight_t>() : nullptr,
      weight_bias.has_value() ? weight_bias->data_ptr<weight_t>() : nullptr,
      group_size, in_features, out_features, vector_len, outlier_size,
      outlier_vector_len, num_centroids, num_res_centroids, index_bits,
      res_bits, norm_dim, indices.size(2), outlier_indices.has_value() ? outlier_indices->size(1) : 0,
      (out_features + vector_len - 1) / vector_len, res_centroids.has_value(),
      outlier_indices.has_value(), inverse_perm.has_value(), weight_scale.has_value());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename weight_t>
void launch_linear(
    const at::Tensor& x, at::Tensor& output, const at::Tensor& indices,
    const at::Tensor& centroids, const c10::optional<at::Tensor>& res_centroids,
    const c10::optional<at::Tensor>& outlier_indices,
    const c10::optional<at::Tensor>& outlier_centroids,
    const c10::optional<at::Tensor>& inverse_perm,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& weight_bias,
    const c10::optional<at::Tensor>& bias, int64_t batch, int64_t group_size,
    int64_t in_features, int64_t out_features, int64_t vector_len,
    int64_t outlier_size, int64_t outlier_vector_len, int64_t num_centroids,
    int64_t num_res_centroids, int64_t index_bits, int64_t res_bits,
    int64_t norm_dim) {
  const bool vectorized = vector_len == 4 && !outlier_indices.has_value();
  if (vectorized) {
    const dim3 blocks((out_features + 3) / 4, batch);
    vptq_linear_vec4_kernel<scalar_t, weight_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), indices.data_ptr<int32_t>(),
      centroids.data_ptr<weight_t>(), res_centroids.has_value() ? res_centroids->data_ptr<weight_t>() : nullptr,
      inverse_perm.has_value() ? inverse_perm->data_ptr<int32_t>() : nullptr,
      weight_scale.has_value() ? weight_scale->data_ptr<weight_t>() : nullptr,
      weight_bias.has_value() ? weight_bias->data_ptr<weight_t>() : nullptr,
      bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr, batch, group_size,
      in_features, out_features, outlier_size, num_centroids, num_res_centroids,
      index_bits, res_bits, indices.size(2),
      (out_features + vector_len - 1) / vector_len, norm_dim, res_centroids.has_value(),
      inverse_perm.has_value(), weight_scale.has_value(), bias.has_value());
  } else {
    const int64_t n = batch * out_features;
    const int blocks = static_cast<int>(n);
    vptq_linear_kernel<scalar_t, weight_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), indices.data_ptr<int32_t>(),
      centroids.data_ptr<weight_t>(), res_centroids.has_value() ? res_centroids->data_ptr<weight_t>() : nullptr,
      outlier_indices.has_value() ? outlier_indices->data_ptr<int32_t>() : nullptr,
      outlier_centroids.has_value() ? outlier_centroids->data_ptr<weight_t>() : nullptr,
      inverse_perm.has_value() ? inverse_perm->data_ptr<int32_t>() : nullptr,
      weight_scale.has_value() ? weight_scale->data_ptr<weight_t>() : nullptr,
      weight_bias.has_value() ? weight_bias->data_ptr<weight_t>() : nullptr,
      bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr, batch, group_size,
      in_features, out_features, vector_len, outlier_size, outlier_vector_len,
      num_centroids, num_res_centroids, index_bits, res_bits, norm_dim,
      indices.size(2), outlier_indices.has_value() ? outlier_indices->size(1) : 0,
      (out_features + vector_len - 1) / vector_len, res_centroids.has_value(),
      outlier_indices.has_value(), inverse_perm.has_value(), weight_scale.has_value(),
      bias.has_value());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor vptq_dequantize_cuda(
    const at::Tensor& indices, const at::Tensor& centroids,
    const c10::optional<at::Tensor>& res_centroids,
    const c10::optional<at::Tensor>& outlier_indices,
    const c10::optional<at::Tensor>& outlier_centroids,
    const c10::optional<at::Tensor>& inverse_perm,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& weight_bias, int64_t group_size,
    int64_t in_features, int64_t out_features, int64_t vector_len,
    int64_t outlier_size, int64_t outlier_vector_len, int64_t num_centroids,
    int64_t num_res_centroids, int64_t index_bits, int64_t res_bits,
    int64_t norm_dim) {
  TORCH_CHECK(indices.is_cuda() && indices.is_contiguous() && indices.dim() == 3 &&
                  indices.scalar_type() == at::kInt,
              "VPTQ CUDA indices must be contiguous 3D int32 on CUDA");
  TORCH_CHECK(centroids.is_cuda() && centroids.device() == indices.device() &&
                  centroids.is_contiguous(),
              "VPTQ CUDA centroids must be contiguous and on the indices device");
  TORCH_CHECK(in_features > 0 && out_features > 0 && group_size > 0 && vector_len > 0,
              "VPTQ CUDA feature and group sizes must be positive");
  TORCH_CHECK(index_bits > 0 && res_bits >= 0 && index_bits + res_bits <= 32,
              "VPTQ CUDA packed-index metadata is invalid");
  TORCH_CHECK(outlier_size >= 0 && outlier_size <= in_features,
              "VPTQ CUDA outlier size is outside the input feature range");
  const auto check_optional = [&](const c10::optional<at::Tensor>& tensor,
                                  const char* name,
                                  c10::optional<at::ScalarType> expected_dtype = c10::nullopt) {
    if (!tensor.has_value()) return;
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == indices.device() && tensor->is_contiguous(),
                "VPTQ CUDA ", name, " must be contiguous and on the indices device");
    if (expected_dtype.has_value()) {
      TORCH_CHECK(tensor->scalar_type() == *expected_dtype,
                  "VPTQ CUDA ", name, " has an unexpected dtype");
    }
  };
  check_optional(res_centroids, "residual centroids", centroids.scalar_type());
  check_optional(outlier_indices, "outlier indices", at::kInt);
  check_optional(outlier_centroids, "outlier centroids", centroids.scalar_type());
  check_optional(inverse_perm, "inverse permutation", at::kInt);
  check_optional(weight_scale, "weight scale", centroids.scalar_type());
  check_optional(weight_bias, "weight bias", centroids.scalar_type());
  TORCH_CHECK(res_centroids.has_value() == (res_bits > 0),
              "VPTQ CUDA residual centroids and residual index bits must agree");
  TORCH_CHECK(outlier_indices.has_value() == outlier_centroids.has_value(),
              "VPTQ CUDA outlier indices and centroids must be provided together");
  TORCH_CHECK(weight_scale.has_value() == weight_bias.has_value(),
              "VPTQ CUDA normalization scale and bias must be provided together");
  TORCH_CHECK(outlier_indices.has_value() == (outlier_size > 0),
              "VPTQ CUDA outlier buffers and outlier size must agree");

  c10::cuda::CUDAGuard guard(indices.device());
  auto out = at::empty({out_features, in_features}, centroids.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, centroids.scalar_type(), "vptq_dequantize", [&] {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "vptq_dequantize_out", [&] {
      launch_dequantize<scalar_t, scalar_t>(out, indices, centroids, res_centroids,
          outlier_indices, outlier_centroids, inverse_perm, weight_scale, weight_bias,
          group_size, in_features, out_features, vector_len, outlier_size,
          outlier_vector_len, num_centroids, num_res_centroids, index_bits, res_bits, norm_dim);
    });
  });
  return out;
}

at::Tensor vptq_linear_cuda(
    const at::Tensor& x, const at::Tensor& indices, const at::Tensor& centroids,
    const c10::optional<at::Tensor>& res_centroids,
    const c10::optional<at::Tensor>& outlier_indices,
    const c10::optional<at::Tensor>& outlier_centroids,
    const c10::optional<at::Tensor>& inverse_perm,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& weight_bias,
    const c10::optional<at::Tensor>& bias, int64_t group_size,
    int64_t in_features, int64_t out_features, int64_t vector_len,
    int64_t outlier_size, int64_t outlier_vector_len, int64_t num_centroids,
    int64_t num_res_centroids, int64_t index_bits, int64_t res_bits,
    int64_t norm_dim) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "VPTQ CUDA expects contiguous CUDA activations");
  TORCH_CHECK(x.dim() >= 1, "VPTQ CUDA activations must have at least one dimension");
  TORCH_CHECK(in_features > 0 && out_features > 0, "VPTQ CUDA feature sizes must be positive");
  TORCH_CHECK(x.size(-1) == in_features, "VPTQ CUDA activation width mismatch");
  TORCH_CHECK(x.scalar_type() == centroids.scalar_type(), "VPTQ CUDA activation/codebook dtypes must match");
  TORCH_CHECK(indices.is_cuda() && indices.device() == x.device() && indices.is_contiguous(),
              "VPTQ CUDA indices must be contiguous and on the activation device");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.dim() == 3,
              "VPTQ CUDA indices must be a contiguous 3D int32 tensor");
  TORCH_CHECK(centroids.is_cuda() && centroids.device() == x.device() && centroids.is_contiguous(),
              "VPTQ CUDA centroids must be contiguous and on the activation device");
  TORCH_CHECK(group_size > 0 && vector_len > 0 && index_bits > 0 && res_bits >= 0,
              "VPTQ CUDA packed-index metadata is invalid");
  TORCH_CHECK(index_bits + res_bits <= 32, "VPTQ CUDA packed indices may use at most 32 bits");
  TORCH_CHECK(outlier_size >= 0 && outlier_size <= in_features,
              "VPTQ CUDA outlier size is outside the input feature range");
  TORCH_CHECK(indices.size(1) >= (out_features + vector_len - 1) / vector_len,
              "VPTQ CUDA indices do not cover all output vectors");
  TORCH_CHECK(indices.size(2) >= (group_size * (index_bits + res_bits) + 31) / 32,
              "VPTQ CUDA packed-index rows are too short for the declared bit width");

  const auto check_optional = [&](const c10::optional<at::Tensor>& tensor,
                                  const char* name,
                                  c10::optional<at::ScalarType> expected_dtype = c10::nullopt) {
    if (!tensor.has_value()) return;
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == x.device() && tensor->is_contiguous(),
                "VPTQ CUDA ", name, " must be contiguous and on the activation device");
    if (expected_dtype.has_value()) {
      TORCH_CHECK(tensor->scalar_type() == *expected_dtype,
                  "VPTQ CUDA ", name, " has an unexpected dtype");
    }
  };
  check_optional(res_centroids, "residual centroids", x.scalar_type());
  check_optional(outlier_indices, "outlier indices", at::kInt);
  check_optional(outlier_centroids, "outlier centroids", x.scalar_type());
  check_optional(inverse_perm, "inverse permutation", at::kInt);
  check_optional(weight_scale, "weight scale", x.scalar_type());
  check_optional(weight_bias, "weight bias", x.scalar_type());
  check_optional(bias, "bias", x.scalar_type());
  TORCH_CHECK(res_centroids.has_value() == (res_bits > 0),
              "VPTQ CUDA residual centroids and residual index bits must agree");
  TORCH_CHECK(outlier_indices.has_value() == outlier_centroids.has_value(),
              "VPTQ CUDA outlier indices and centroids must be provided together");
  TORCH_CHECK(weight_scale.has_value() == weight_bias.has_value(),
              "VPTQ CUDA normalization scale and bias must be provided together");
  TORCH_CHECK(outlier_indices.has_value() == (outlier_size > 0),
              "VPTQ CUDA outlier buffers and outlier size must agree");
  if (bias.has_value()) {
    TORCH_CHECK(bias->numel() == out_features, "VPTQ CUDA bias width mismatch");
  }
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t batch = x.numel() / in_features;
  auto output = at::empty({batch, out_features}, x.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "vptq_linear", [&] {
    launch_linear<scalar_t, scalar_t>(x, output, indices, centroids, res_centroids,
        outlier_indices, outlier_centroids, inverse_perm, weight_scale, weight_bias,
        bias, batch, group_size, in_features, out_features, vector_len, outlier_size,
        outlier_vector_len, num_centroids, num_res_centroids, index_bits, res_bits, norm_dim);
  });
  auto shape = x.sizes().vec();
  shape.back() = out_features;
  return output.view(shape);
}

}  // namespace gptqmodel::vptq
