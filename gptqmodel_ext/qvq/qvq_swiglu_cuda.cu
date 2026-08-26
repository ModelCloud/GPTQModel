// SPDX-License-Identifier: Apache-2.0
// Fused analytical Smooth-SwiGLU statistics and group-scale solve.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>
#include <tuple>

namespace {

template <typename T> __device__ __forceinline__ float as_float(T v);
template <> __device__ __forceinline__ float as_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float as_float<half>(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float as_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template <typename Weight>
__global__ void swiglu_channel_stats_kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    const Weight* __restrict__ up_weight, const Weight* __restrict__ down_weight,
    float* __restrict__ a, float* __restrict__ b,
    float* __restrict__ gate_salience, float* __restrict__ up_salience, float* __restrict__ down_salience,
    int tokens, int intermediate, int hidden, int output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= intermediate) return;
  float alpha = 0.0f, beta = 0.0f, gamma = 0.0f, up_energy = 0.0f, down_energy = 0.0f;
  for (int t = 0; t < tokens; ++t) {
    const float g = gate[static_cast<int64_t>(t) * intermediate + i];
    const float u = up[static_cast<int64_t>(t) * intermediate + i];
    const float sigmoid = 1.0f / (1.0f + expf(-g));
    const float silu = g * sigmoid;
    const float h = silu * u;
    const float derivative = sigmoid * (1.0f + g * (1.0f - sigmoid));
    alpha += silu * silu;
    beta += h * h;
    const float gate_term = u * derivative;
    gamma += gate_term * gate_term;
  }
  for (int j = 0; j < hidden; ++j)
    up_energy += as_float(up_weight[static_cast<int64_t>(i) * hidden + j]) * as_float(up_weight[static_cast<int64_t>(i) * hidden + j]);
  for (int j = 0; j < output; ++j)
    down_energy += as_float(down_weight[static_cast<int64_t>(j) * intermediate + i]) * as_float(down_weight[static_cast<int64_t>(j) * intermediate + i]);
  a[i] = (alpha / static_cast<float>(tokens)) * (up_energy / static_cast<float>(hidden));
  b[i] = (beta / static_cast<float>(tokens)) * down_energy;
  gate_salience[i] = (gamma / static_cast<float>(tokens)) * down_energy;
  up_salience[i] = (alpha / static_cast<float>(tokens)) * down_energy;
  down_salience[i] = beta / static_cast<float>(tokens);
}

__global__ void swiglu_group_solve_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ scales, float* __restrict__ proxy,
    int intermediate, int group_size, int groups, float scale_min, float scale_max) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups) return;
  float sum_a = 0.0f, sum_b = 0.0f;
  const int start = group * group_size;
  const int stop = min(start + group_size, intermediate);
  for (int i = start; i < stop; ++i) { sum_a += a[i]; sum_b += b[i]; }
  const float eps = 1.1920928955078125e-7f;
  float s = powf(fmaxf(sum_b, eps) / fmaxf(sum_a, eps), 0.25f);
  s = fminf(fmaxf(s, scale_min), scale_max);
  for (int i = start; i < stop; ++i) scales[i] = s;
  proxy[group] = sum_a * s * s + sum_b / (s * s);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> swiglu_proxy_scales_cuda(
    const at::Tensor& gate, const at::Tensor& up, const at::Tensor& up_weight,
    const at::Tensor& down_weight, int64_t group_size, double scale_min, double scale_max) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda() && up_weight.is_cuda() && down_weight.is_cuda(), "Smooth-SwiGLU CUDA op requires CUDA tensors");
  TORCH_CHECK(gate.device() == up.device() && gate.device() == up_weight.device() && gate.device() == down_weight.device(), "Smooth-SwiGLU CUDA tensors must share a device");
  TORCH_CHECK(gate.scalar_type() == at::kFloat && up.scalar_type() == at::kFloat, "gate/up activations must be float32");
  TORCH_CHECK(up_weight.scalar_type() == down_weight.scalar_type(), "up/down weights must share a dtype");
  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous() && up_weight.is_contiguous() && down_weight.is_contiguous(), "Smooth-SwiGLU CUDA tensors must be contiguous");
  TORCH_CHECK(gate.dim() == 2 && up.sizes() == gate.sizes(), "gate/up must be [tokens, intermediate]");
  TORCH_CHECK(up_weight.dim() == 2 && down_weight.dim() == 2 && up_weight.size(0) == gate.size(1) && down_weight.size(1) == gate.size(1), "invalid projection geometry");
  TORCH_CHECK(group_size > 0 && scale_min > 0 && scale_min <= scale_max, "invalid group or scale bounds");
  const int tokens = static_cast<int>(gate.size(0));
  const int intermediate = static_cast<int>(gate.size(1));
  const int hidden = static_cast<int>(up_weight.size(1));
  const int output = static_cast<int>(down_weight.size(0));
  const int groups = (intermediate + static_cast<int>(group_size) - 1) / static_cast<int>(group_size);
  auto options = gate.options().dtype(at::kFloat);
  auto a = at::empty({intermediate}, options);
  auto b = at::empty({intermediate}, options);
  auto gate_salience = at::empty({intermediate}, options);
  auto up_salience = at::empty({intermediate}, options);
  auto down_salience = at::empty({intermediate}, options);
  auto scales = at::empty({intermediate}, options);
  auto proxy = at::empty({groups}, options);
  const c10::cuda::CUDAGuard device_guard(gate.device());
  const auto stream = at::cuda::getCurrentCUDAStream(gate.get_device());
  const dim3 blocks((intermediate + 255) / 256);
  const dim3 threads(256);
#define LAUNCH(W) swiglu_channel_stats_kernel<W><<<blocks, threads, 0, stream>>>(reinterpret_cast<const float*>(gate.const_data_ptr()), reinterpret_cast<const float*>(up.const_data_ptr()), reinterpret_cast<const W*>(up_weight.const_data_ptr()), reinterpret_cast<const W*>(down_weight.const_data_ptr()), reinterpret_cast<float*>(a.mutable_data_ptr()), reinterpret_cast<float*>(b.mutable_data_ptr()), reinterpret_cast<float*>(gate_salience.mutable_data_ptr()), reinterpret_cast<float*>(up_salience.mutable_data_ptr()), reinterpret_cast<float*>(down_salience.mutable_data_ptr()), tokens, intermediate, hidden, output)
  if (up_weight.scalar_type() == at::kHalf) LAUNCH(half);
  else if (up_weight.scalar_type() == at::kBFloat16) LAUNCH(nv_bfloat16);
  else LAUNCH(float);
#undef LAUNCH
  swiglu_group_solve_kernel<<<(groups + 255) / 256, 256, 0, stream>>>(reinterpret_cast<const float*>(a.const_data_ptr()), reinterpret_cast<const float*>(b.const_data_ptr()), reinterpret_cast<float*>(scales.mutable_data_ptr()), reinterpret_cast<float*>(proxy.mutable_data_ptr()), intermediate, static_cast<int>(group_size), groups, static_cast<float>(scale_min), static_cast<float>(scale_max));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {scales, proxy, gate_salience, up_salience, down_salience};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  m.def("swiglu_proxy_scales(Tensor gate, Tensor up, Tensor up_weight, Tensor down_weight, int group_size, float scale_min, float scale_max) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CUDA, m) {
  m.impl("swiglu_proxy_scales", &swiglu_proxy_scales_cuda);
}
