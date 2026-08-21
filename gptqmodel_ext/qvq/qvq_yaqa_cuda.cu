// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Fused YAQA corrected-tile construction for the factored CUDA recurrence.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <torch/library.h>
#include <torch/types.h>

#include <vector>

namespace {

constexpr int kTile = 16;
constexpr int kThreads = kTile * kTile;
static_assert(sizeof(float*) == sizeof(int64_t), "YAQA grouped pointers require 64-bit device addresses");

__global__ void qvq_yaqa_grouped_pointers_kernel(
    const float* left,
    const float* output_feedback,
    float* cross,
    float** a_array,
    float** b_array,
    float** c_array,
    int first_input_block,
    int first_output_block,
    int in_features,
    int out_features,
    int count,
    int families) {
  const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = count * families;
  if (item >= total) {
    return;
  }
  const int family = item / count;
  const int tile = item - family * count;
  const int input_start = (first_input_block + tile) * kTile;
  const int output_start = (first_output_block - tile) * kTile;
  const int64_t matrix_stride = static_cast<int64_t>(in_features) * out_features;
  const int64_t family_offset = static_cast<int64_t>(family) * matrix_stride;
  // Row-major C=P@L is column-major C.T=L.T@P.T.
  b_array[item] = const_cast<float*>(
      left + family_offset + static_cast<int64_t>(input_start) * out_features + output_start);
  a_array[item] = const_cast<float*>(
      output_feedback + static_cast<int64_t>(output_start) * out_features + output_start);
  c_array[item] = cross + static_cast<int64_t>(item) * kThreads;
}

__global__ void qvq_yaqa_update_grouped_pointers_kernel(
    float* left,
    float* right,
    const float* input_feedback,
    const float* output_feedback,
    const float* reconstructed,
    float** a_array,
    float** b_array,
    float** c_array,
    int first_input_block,
    int first_output_block,
    int in_features,
    int out_features,
    int count,
    int families) {
  const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = count * families;
  if (item >= total) {
    return;
  }
  const int family = item / count;
  const int tile = item - family * count;
  const int input_start = (first_input_block + tile) * kTile;
  const int output_start = (first_output_block - tile) * kTile;
  const int64_t matrix_stride = static_cast<int64_t>(in_features) * out_features;
  const int64_t family_offset = static_cast<int64_t>(family) * matrix_stride;
  const float* tile_reconstruction = reconstructed + static_cast<int64_t>(item) * kThreads;

  // Row-major P[:, j] -= L_I[i, :].T @ Q is column-major
  // P[:, j].T -= Q.T @ L_I[i, :].
  a_array[item] = const_cast<float*>(tile_reconstruction);
  b_array[item] = const_cast<float*>(
      input_feedback + static_cast<int64_t>(input_start) * in_features);
  c_array[item] = left + family_offset + output_start;

  // Row-major R[i, :] -= Q @ L_O[j, :] is column-major
  // R[i, :].T -= L_O[j, :].T @ Q.T.
  const int right_index = total + item;
  a_array[right_index] = const_cast<float*>(
      output_feedback + static_cast<int64_t>(output_start) * out_features);
  b_array[right_index] = const_cast<float*>(tile_reconstruction);
  c_array[right_index] = right + family_offset + static_cast<int64_t>(input_start) * out_features;
}

__global__ __launch_bounds__(kThreads) void qvq_yaqa_feedback_epilogue_kernel(
    const float* __restrict__ source,
    const float* __restrict__ left,
    const float* __restrict__ right,
    const float* __restrict__ cross,
    const float* __restrict__ bias,
    float* __restrict__ corrected,
    int first_input_block,
    int first_output_block,
    int in_features,
    int out_features,
    int count) {
  const int item = static_cast<int>(blockIdx.x);
  const int family = item / count;
  const int tile = item - family * count;
  const int lane = static_cast<int>(threadIdx.x);
  const int row = lane / kTile;
  const int col = lane % kTile;
  const int input_start = (first_input_block + tile) * kTile;
  const int output_start = (first_output_block - tile) * kTile;
  const int64_t matrix_stride = static_cast<int64_t>(in_features) * out_features;
  const int64_t family_offset = static_cast<int64_t>(family) * matrix_stride;

  const int64_t matrix_index = static_cast<int64_t>(input_start + row) * out_features + output_start + col;
  float value = source[matrix_index];
  if (bias != nullptr) {
    value += bias[matrix_index];
  }
  value += cross[static_cast<int64_t>(item) * kThreads + lane];
  value += left[family_offset + matrix_index];
  value += right[family_offset + matrix_index];
  corrected[static_cast<int64_t>(item) * kThreads + lane] = value;
}

at::Tensor qvq_yaqa_feedback_cuda(
    const at::Tensor& source,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& output_feedback,
    int64_t first_input_block,
    int64_t first_output_block,
    int64_t count,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(source.is_cuda(), "YAQA source must be CUDA");
  TORCH_CHECK(source.scalar_type() == at::kFloat && source.dim() == 2 && source.is_contiguous(),
              "YAQA source must be contiguous FP32 [input, output]");
  TORCH_CHECK(left.device() == source.device() && left.scalar_type() == at::kFloat && left.is_contiguous() &&
                  (left.sizes() == source.sizes() ||
                   (left.dim() == 3 && left.size(1) == source.size(0) && left.size(2) == source.size(1))),
              "YAQA left transformed error must be contiguous FP32 [input, output] or [family, input, output]");
  TORCH_CHECK(right.device() == source.device() && right.scalar_type() == at::kFloat &&
                  right.sizes() == left.sizes() && right.is_contiguous(),
              "YAQA right transformed error must match the left transformed error");
  TORCH_CHECK(output_feedback.device() == source.device() && output_feedback.scalar_type() == at::kFloat &&
                  output_feedback.dim() == 2 && output_feedback.size(0) == source.size(1) &&
                  output_feedback.size(1) == source.size(1) && output_feedback.is_contiguous(),
              "YAQA output feedback must be contiguous FP32 [output, output]");
  TORCH_CHECK(source.size(0) % kTile == 0 && source.size(1) % kTile == 0,
              "YAQA fused feedback requires dimensions divisible by 16");
  TORCH_CHECK(count >= 1 && first_input_block >= 0 && first_output_block >= count - 1 &&
                  first_input_block + count <= source.size(0) / kTile &&
                  first_output_block < source.size(1) / kTile,
              "YAQA anti-diagonal geometry is invalid");
  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == source.device() && bias->scalar_type() == at::kFloat &&
                    bias->sizes() == source.sizes() && bias->is_contiguous(),
                "YAQA bias must match the contiguous FP32 source");
  }

  const c10::cuda::CUDAGuard device_guard(source.device());
  const int64_t families = left.dim() == 3 ? left.size(0) : 1;
  const int64_t total = families * count;
  at::Tensor corrected = left.dim() == 3
      ? at::empty({families, count, kTile, kTile}, source.options())
      : at::empty({count, kTile, kTile}, source.options());
  at::Tensor cross = at::empty_like(corrected);
  at::Tensor pointer_storage = at::empty({3, total}, source.options().dtype(at::kLong));
  auto pointers = reinterpret_cast<float**>(pointer_storage.mutable_data_ptr<int64_t>());
  auto a_array = pointers;
  auto b_array = pointers + total;
  auto c_array = pointers + 2 * total;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device());
  qvq_yaqa_grouped_pointers_kernel<<<static_cast<unsigned int>((total + 127) / 128), 128, 0, stream>>>(
      left.const_data_ptr<float>(),
      output_feedback.const_data_ptr<float>(),
      cross.mutable_data_ptr<float>(),
      a_array,
      b_array,
      c_array,
      static_cast<int>(first_input_block),
      static_cast<int>(first_output_block),
      static_cast<int>(source.size(0)),
      static_cast<int>(source.size(1)),
      static_cast<int>(count),
      static_cast<int>(families));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int group_count = static_cast<int>(total);
  std::vector<cublasOperation_t> operations(group_count, CUBLAS_OP_N);
  std::vector<int> m(group_count, kTile);
  std::vector<int> n(group_count, kTile);
  std::vector<int> k(group_count);
  std::vector<int> lda(group_count, static_cast<int>(source.size(1)));
  std::vector<int> ldb(group_count, static_cast<int>(source.size(1)));
  std::vector<int> ldc(group_count, kTile);
  std::vector<int> group_sizes(group_count, 1);
  std::vector<float> alpha(group_count, 1.0f);
  std::vector<float> beta(group_count, 0.0f);
  for (int item = 0; item < group_count; ++item) {
    const int tile = item % static_cast<int>(count);
    k[item] = static_cast<int>(source.size(1)) -
        (static_cast<int>(first_output_block) - tile) * kTile;
  }
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasPointerMode_t previous_pointer_mode;
  TORCH_CHECK(cublasGetPointerMode(handle, &previous_pointer_mode) == CUBLAS_STATUS_SUCCESS,
              "failed to read YAQA cuBLAS pointer mode");
  TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS, "failed to bind YAQA cuBLAS stream");
  TORCH_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_SUCCESS,
              "failed to set YAQA cuBLAS pointer mode");
  const cublasStatus_t grouped_status = cublasSgemmGroupedBatched(
      handle,
      operations.data(),
      operations.data(),
      m.data(),
      n.data(),
      k.data(),
      alpha.data(),
      a_array,
      lda.data(),
      b_array,
      ldb.data(),
      beta.data(),
      c_array,
      ldc.data(),
      group_count,
      group_sizes.data());
  TORCH_CHECK(cublasSetPointerMode(handle, previous_pointer_mode) == CUBLAS_STATUS_SUCCESS,
              "failed to restore YAQA cuBLAS pointer mode");
  TORCH_CHECK(grouped_status == CUBLAS_STATUS_SUCCESS, "YAQA grouped FP32 GEMM failed");

  qvq_yaqa_feedback_epilogue_kernel<<<static_cast<unsigned int>(total), kThreads, 0, stream>>>(
      source.const_data_ptr<float>(),
      left.const_data_ptr<float>(),
      right.const_data_ptr<float>(),
      cross.const_data_ptr<float>(),
      bias.has_value() ? bias->const_data_ptr<float>() : nullptr,
      corrected.mutable_data_ptr<float>(),
      static_cast<int>(first_input_block),
      static_cast<int>(first_output_block),
      static_cast<int>(source.size(0)),
      static_cast<int>(source.size(1)),
      static_cast<int>(count));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return corrected;
}

void qvq_yaqa_feedback_update_cuda(
    at::Tensor& left,
    at::Tensor& right,
    const at::Tensor& input_feedback,
    const at::Tensor& output_feedback,
    const at::Tensor& reconstructed,
    int64_t first_input_block,
    int64_t first_output_block,
    int64_t count) {
  TORCH_CHECK(left.is_cuda() && left.scalar_type() == at::kFloat &&
                  (left.dim() == 2 || left.dim() == 3) && left.is_contiguous(),
              "YAQA left cache must be contiguous CUDA FP32 [input, output] or [family, input, output]");
  TORCH_CHECK(right.device() == left.device() && right.scalar_type() == at::kFloat &&
                  right.sizes() == left.sizes() && right.is_contiguous(),
              "YAQA right cache must match the contiguous FP32 left cache");
  const int64_t families = left.dim() == 3 ? left.size(0) : 1;
  const int64_t in_features = left.size(left.dim() - 2);
  const int64_t out_features = left.size(left.dim() - 1);
  TORCH_CHECK(input_feedback.device() == left.device() && input_feedback.scalar_type() == at::kFloat &&
                  input_feedback.sizes() == at::IntArrayRef({in_features, in_features}) &&
                  input_feedback.is_contiguous(),
              "YAQA input feedback must be contiguous FP32 [input, input]");
  TORCH_CHECK(output_feedback.device() == left.device() && output_feedback.scalar_type() == at::kFloat &&
                  output_feedback.sizes() == at::IntArrayRef({out_features, out_features}) &&
                  output_feedback.is_contiguous(),
              "YAQA output feedback must be contiguous FP32 [output, output]");
  TORCH_CHECK(reconstructed.device() == left.device() && reconstructed.scalar_type() == at::kFloat &&
                  ((left.dim() == 2 && reconstructed.dim() == 3 && reconstructed.size(0) == count) ||
                   (left.dim() == 3 && reconstructed.dim() == 4 && reconstructed.size(0) == families &&
                    reconstructed.size(1) == count)) &&
                  reconstructed.size(reconstructed.dim() - 2) == kTile &&
                  reconstructed.size(reconstructed.dim() - 1) == kTile &&
                  reconstructed.is_contiguous(),
              "YAQA reconstruction must be contiguous FP32 [count, 16, 16] or [family, count, 16, 16]");
  TORCH_CHECK(in_features % kTile == 0 && out_features % kTile == 0,
              "YAQA fused update requires dimensions divisible by 16");
  TORCH_CHECK(count >= 1 && first_input_block >= 0 && first_output_block >= count - 1 &&
                  first_input_block + count <= in_features / kTile &&
                  first_output_block < out_features / kTile,
              "YAQA update anti-diagonal geometry is invalid");

  const c10::cuda::CUDAGuard device_guard(left.device());
  const int64_t total = families * count;
  at::Tensor pointer_storage = at::empty({6, total}, left.options().dtype(at::kLong));
  auto pointers = reinterpret_cast<float**>(pointer_storage.mutable_data_ptr<int64_t>());
  auto a_array = pointers;
  auto b_array = pointers + 2 * total;
  auto c_array = pointers + 4 * total;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(left.get_device());
  qvq_yaqa_update_grouped_pointers_kernel<<<static_cast<unsigned int>((total + 127) / 128), 128, 0, stream>>>(
      left.mutable_data_ptr<float>(),
      right.mutable_data_ptr<float>(),
      input_feedback.const_data_ptr<float>(),
      output_feedback.const_data_ptr<float>(),
      reconstructed.const_data_ptr<float>(),
      a_array,
      b_array,
      c_array,
      static_cast<int>(first_input_block),
      static_cast<int>(first_output_block),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(count),
      static_cast<int>(families));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int matrix_count = static_cast<int>(total);
  const float alpha = -1.0f;
  const float beta = 1.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasPointerMode_t previous_pointer_mode;
  TORCH_CHECK(cublasGetPointerMode(handle, &previous_pointer_mode) == CUBLAS_STATUS_SUCCESS,
              "failed to read YAQA update cuBLAS pointer mode");
  TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
              "failed to bind YAQA update cuBLAS stream");
  TORCH_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_SUCCESS,
              "failed to set YAQA update cuBLAS pointer mode");
  const cublasStatus_t left_status = cublasSgemmBatched(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      kTile,
      static_cast<int>(in_features),
      kTile,
      &alpha,
      a_array,
      kTile,
      b_array,
      static_cast<int>(in_features),
      &beta,
      c_array,
      static_cast<int>(out_features),
      matrix_count);
  const cublasStatus_t right_status = cublasSgemmBatched(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<int>(out_features),
      kTile,
      kTile,
      &alpha,
      a_array + total,
      static_cast<int>(out_features),
      b_array + total,
      kTile,
      &beta,
      c_array + total,
      static_cast<int>(out_features),
      matrix_count);
  TORCH_CHECK(cublasSetPointerMode(handle, previous_pointer_mode) == CUBLAS_STATUS_SUCCESS,
              "failed to restore YAQA update cuBLAS pointer mode");
  TORCH_CHECK(left_status == CUBLAS_STATUS_SUCCESS && right_status == CUBLAS_STATUS_SUCCESS,
              "YAQA batched FP32 cache update failed");
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("yaqa_feedback(Tensor source, Tensor left, Tensor right, Tensor output_feedback, "
        "int first_input_block, int first_output_block, int count, Tensor? bias) -> Tensor");
  m.def("yaqa_feedback_update_(Tensor(a!) left, Tensor(b!) right, Tensor input_feedback, "
        "Tensor output_feedback, Tensor reconstructed, int first_input_block, "
        "int first_output_block, int count) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("yaqa_feedback", &qvq_yaqa_feedback_cuda);
  m.impl("yaqa_feedback_update_", &qvq_yaqa_feedback_update_cuda);
}
