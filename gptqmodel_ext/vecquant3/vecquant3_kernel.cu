// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kBlockWidth = 256;
constexpr int kBlockHeight = 24;
constexpr int kThreads = kBlockWidth;
constexpr int kAccumulationFloat32 = 0;
constexpr int kAccumulationInput = 1;

template <typename scalar_t>
struct scalar_traits;

template <>
struct scalar_traits<half> {
  using scalar2_t = half2;
  using torch_t = at::Half;

  __device__ __forceinline__ static half from_float(float value) {
    return __float2half(value);
  }

  __device__ __forceinline__ static half from_int(int value) {
    return __int2half_rn(value);
  }

  __device__ __forceinline__ static half2 make2(half value) {
    return __half2half2(value);
  }

  __device__ __forceinline__ static half2 make2(half low, half high) {
    return __halves2half2(low, high);
  }

  __device__ __forceinline__ static half mul(half a, half b) {
    return __hmul(a, b);
  }

  __device__ __forceinline__ static half2 mul2(half2 a, half2 b) {
    return __hmul2(a, b);
  }

  __device__ __forceinline__ static half2 fma2(half2 a, half2 b, half2 c) {
    return __hfma2(a, b, c);
  }

  __device__ __forceinline__ static float to_float(half value) {
    return __half2float(value);
  }
};

template <>
struct scalar_traits<__nv_bfloat16> {
  using scalar2_t = __nv_bfloat162;
  using torch_t = c10::BFloat16;

  __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }

  __device__ __forceinline__ static __nv_bfloat16 from_int(int value) {
    return __float2bfloat16(static_cast<float>(value));
  }

  __device__ __forceinline__ static __nv_bfloat162 make2(__nv_bfloat16 value) {
    return __bfloat162bfloat162(value);
  }

  __device__ __forceinline__ static __nv_bfloat162 make2(__nv_bfloat16 low,
                                                         __nv_bfloat16 high) {
    return __halves2bfloat162(low, high);
  }

  __device__ __forceinline__ static __nv_bfloat16 mul(__nv_bfloat16 a,
                                                      __nv_bfloat16 b) {
    return __hmul(a, b);
  }

  __device__ __forceinline__ static __nv_bfloat162 mul2(__nv_bfloat162 a,
                                                        __nv_bfloat162 b) {
    return __hmul2(a, b);
  }

  __device__ __forceinline__ static __nv_bfloat162 fma2(__nv_bfloat162 a,
                                                        __nv_bfloat162 b,
                                                        __nv_bfloat162 c) {
    return __hfma2(a, b, c);
  }

  __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
};

__device__ __forceinline__ unsigned int as_unsigned(int value) {
  return *reinterpret_cast<unsigned int *>(&value);
}

__device__ __forceinline__ int unpack_int3_zero(const int *__restrict__ qzeros,
                                                int group,
                                                int col,
                                                int qzeros_stride) {
  const int block = col >> 5;
  const int idx = col & 31;
  const int *base = qzeros + group * qzeros_stride + block * 3;
  const unsigned int word0 = as_unsigned(base[0]);
  const unsigned int word1 = as_unsigned(base[1]);
  const unsigned int word2 = as_unsigned(base[2]);

  if (idx <= 9) {
    return (word0 >> (idx * 3)) & 0x7;
  }
  if (idx == 10) {
    return ((word0 >> 30) & 0x3) | ((word1 << 2) & 0x4);
  }
  if (idx <= 20) {
    return (word1 >> (1 + 3 * (idx - 11))) & 0x7;
  }
  if (idx == 21) {
    return ((word1 >> 31) & 0x1) | ((word2 << 1) & 0x6);
  }
  return (word2 >> (2 + 3 * (idx - 22))) & 0x7;
}

template <typename scalar_t, int GroupSize, bool FloatAccum>
__device__ __forceinline__ void accumulate_pair(
    unsigned int packed_pair,
    int absolute_half2_k,
    int relative_half2_k,
    int col,
    int width,
    int qzeros_stride,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ blockvec,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ scales,
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ deq2,
    int deq_offset,
    float &acc_float,
    typename scalar_traits<scalar_t>::scalar2_t &acc_input) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;

  const int group = (absolute_half2_k * 2) / GroupSize;
  const scalar_t scale = scales[group * width + col];
  const int zero = unpack_int3_zero(qzeros, group, col, qzeros_stride);
  const scalar2_t scale2 = traits::make2(scale);
  const scalar2_t zero2 =
      traits::make2(traits::mul(traits::from_float(-static_cast<float>(zero)), scale));
  const scalar2_t deq = deq2[(packed_pair & 0x3f) * 32 + deq_offset];
  const scalar2_t weight = traits::fma2(deq, scale2, zero2);
  if constexpr (FloatAccum) {
    const scalar2_t product = traits::mul2(weight, blockvec[relative_half2_k]);
    acc_float += traits::to_float(product.x) + traits::to_float(product.y);
  } else {
    acc_input = traits::fma2(weight, blockvec[relative_half2_k], acc_input);
  }
}

template <typename scalar_t, int GroupSize, bool WithLora, bool FloatAccum>
__global__ void vecquant3_gptq_gemv_kernel(
    const typename scalar_traits<scalar_t>::scalar2_t *__restrict__ vec,
    const int *__restrict__ qweight,
    const scalar_t *__restrict__ scales,
    const int *__restrict__ qzeros,
    const scalar_t *__restrict__ down,
    const scalar_t *__restrict__ up,
    float *__restrict__ out,
    int qweight_rows,
    int width,
    int qzeros_stride,
    int rank) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;

  const int blockwidth2 = kBlockWidth / 2;
  const int row = kBlockHeight * blockIdx.x;
  const int col = kBlockWidth * blockIdx.y + threadIdx.x;
  if (col >= width) {
    return;
  }

  __shared__ scalar2_t blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2) {
    blockvec[threadIdx.x] = vec[blockIdx.x * blockwidth2 + threadIdx.x];
  }

  __shared__ scalar2_t deq2[64][32];
  int val = threadIdx.x / 32;
  const int off = threadIdx.x % 32;
  for (; val < 64; val += kBlockWidth / 32) {
    deq2[val][off] = traits::make2(traits::from_int(val & 0x7),
                                   traits::from_int(val >> 3));
  }

  __syncthreads();

  int i = width * row + col;
  int k = 0;
  const int absolute_half2_base = blockIdx.x * blockwidth2;
  float acc_float = 0.0f;
  scalar2_t acc_input = traits::make2(traits::from_float(0.0f));

  while (k < blockwidth2 && (row + ((k * 3) >> 4)) < qweight_rows) {
    unsigned int tmp1 = as_unsigned(qweight[i]);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 24, absolute_half2_base + k + 4, k + 4, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    i += width;
    unsigned int tmp2 = as_unsigned(qweight[i]);
    unsigned int tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp, absolute_half2_base + k + 5, k + 5, col, width, qzeros_stride,
        blockvec, qzeros, scales, &deq2[0][0], off, acc_float, acc_input);
    tmp2 >>= 4;
    k += 6;

    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp2 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp2 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp2 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp2 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    i += width;
    tmp1 = as_unsigned(qweight[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp, absolute_half2_base + k + 4, k + 4, col, width, qzeros_stride,
        blockvec, qzeros, scales, &deq2[0][0], off, acc_float, acc_input);
    tmp1 >>= 2;
    k += 5;

    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 0, absolute_half2_base + k + 0, k + 0, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 6, absolute_half2_base + k + 1, k + 1, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 12, absolute_half2_base + k + 2, k + 2, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 18, absolute_half2_base + k + 3, k + 3, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    accumulate_pair<scalar_t, GroupSize, FloatAccum>(
        tmp1 >> 24, absolute_half2_base + k + 4, k + 4, col, width,
        qzeros_stride, blockvec, qzeros, scales, &deq2[0][0], off, acc_float,
        acc_input);
    i += width;
    k += 5;
  }

  float acc;
  if constexpr (FloatAccum) {
    acc = acc_float;
  } else {
    acc = traits::to_float(acc_input.x) + traits::to_float(acc_input.y);
  }

  if constexpr (WithLora) {
    if (blockIdx.x == 0) {
      for (int r = 0; r < rank; ++r) {
        acc += traits::to_float(traits::mul(down[r], up[r * width + col]));
      }
    }
  }

  atomicAdd(&out[col], acc);
}

void validate_common_inputs(const torch::Tensor &vec,
                            const torch::Tensor &qweight,
                            const torch::Tensor &scales,
                            const torch::Tensor &qzeros,
                            int64_t group_size,
                            int64_t accumulation_type) {
  TORCH_CHECK(vec.is_cuda(), "vecquant3 gemv requires CUDA vec");
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && qzeros.is_cuda(),
              "vecquant3 gemv inputs must be CUDA tensors");
  TORCH_CHECK(vec.scalar_type() == torch::kFloat16 ||
                  vec.scalar_type() == torch::kBFloat16,
              "vecquant3 gemv requires fp16 or bf16 vec");
  TORCH_CHECK(scales.scalar_type() == vec.scalar_type(),
              "vecquant3 gemv scales dtype must match vec dtype");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "vecquant3 gemv requires int32 qweight");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
              "vecquant3 gemv requires int32 qzeros");
  TORCH_CHECK(accumulation_type == kAccumulationFloat32 ||
                  accumulation_type == kAccumulationInput,
              "vecquant3 gemv accumulation_type must be 0 (float32) or 1 (input dtype)");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "vecquant3 gemv group_size must be one of 32, 64, 128");
  TORCH_CHECK(vec.numel() == (qweight.size(0) / 3) * 32,
              "vecquant3 gemv vec length must match qweight int3 packing");
  TORCH_CHECK(vec.numel() % 256 == 0,
              "vecquant3 gemv fast path requires in_features divisible by 256");
  TORCH_CHECK(qweight.dim() == 2 && qweight.size(0) % kBlockHeight == 0,
              "vecquant3 gemv qweight must be [in_features // 32 * 3, out_features] with in_features divisible by 256");
  TORCH_CHECK(scales.dim() == 2,
              "vecquant3 gemv scales must be [num_groups, out_features]");
  TORCH_CHECK(qzeros.dim() == 2,
              "vecquant3 gemv qzeros must be [num_groups, out_features // 32 * 3]");
  TORCH_CHECK(scales.size(1) == qweight.size(1),
              "vecquant3 gemv scales out_features mismatch");
  TORCH_CHECK(qweight.size(1) % 32 == 0,
              "vecquant3 gemv out_features must be divisible by 32");
  TORCH_CHECK(qzeros.size(0) == scales.size(0),
              "vecquant3 gemv qzeros/scales num_groups mismatch");
  TORCH_CHECK(qzeros.size(1) == (qweight.size(1) / 32) * 3,
              "vecquant3 gemv qzeros packed width mismatch");
  TORCH_CHECK(scales.size(0) == vec.numel() / group_size,
              "vecquant3 gemv scales num_groups must equal in_features / group_size");
  TORCH_CHECK(vec.is_contiguous() && qweight.is_contiguous() &&
                  scales.is_contiguous() && qzeros.is_contiguous(),
              "vecquant3 gemv inputs must be contiguous");
}

template <typename scalar_t, bool WithLora, bool FloatAccum>
torch::Tensor launch_vecquant3_gptq_gemv_typed(torch::Tensor vec,
                                               torch::Tensor qweight,
                                               torch::Tensor scales,
                                               torch::Tensor qzeros,
                                               torch::Tensor down,
                                               torch::Tensor up,
                                               int64_t group_size) {
  using traits = scalar_traits<scalar_t>;
  using scalar2_t = typename traits::scalar2_t;
  using torch_t = typename traits::torch_t;
  const c10::cuda::CUDAGuard device_guard(vec.device());
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
    const auto *props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major >= 8,
                "vecquant3 gemv bf16 requires CUDA compute capability >= 8.0");
  }

  auto out = torch::empty({qweight.size(1)},
                          torch::TensorOptions().device(vec.device()).dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream(vec.device().index());
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(), 0,
                                 out.numel() * sizeof(float), stream));

  const int qweight_rows = static_cast<int>(qweight.size(0));
  const int width = static_cast<int>(qweight.size(1));
  const int qzeros_stride = static_cast<int>(qzeros.size(1));
  const int rank = WithLora ? static_cast<int>(down.numel()) : 0;
  dim3 blocks((qweight_rows + kBlockHeight - 1) / kBlockHeight,
              (width + kBlockWidth - 1) / kBlockWidth);
  dim3 threads(kThreads);

  const auto *vec_ptr =
      reinterpret_cast<const scalar2_t *>(vec.data_ptr<torch_t>());
  const auto *scale_ptr =
      reinterpret_cast<const scalar_t *>(scales.data_ptr<torch_t>());
  const auto *down_ptr =
      WithLora ? reinterpret_cast<const scalar_t *>(down.data_ptr<torch_t>()) : nullptr;
  const auto *up_ptr =
      WithLora ? reinterpret_cast<const scalar_t *>(up.data_ptr<torch_t>()) : nullptr;

  if (group_size == 32) {
    vecquant3_gptq_gemv_kernel<scalar_t, 32, WithLora, FloatAccum>
        <<<blocks, threads, 0, stream>>>(
        vec_ptr, qweight.data_ptr<int>(), scale_ptr, qzeros.data_ptr<int>(),
        down_ptr, up_ptr, out.data_ptr<float>(), qweight_rows, width,
        qzeros_stride, rank);
  } else if (group_size == 64) {
    vecquant3_gptq_gemv_kernel<scalar_t, 64, WithLora, FloatAccum>
        <<<blocks, threads, 0, stream>>>(
        vec_ptr, qweight.data_ptr<int>(), scale_ptr, qzeros.data_ptr<int>(),
        down_ptr, up_ptr, out.data_ptr<float>(), qweight_rows, width,
        qzeros_stride, rank);
  } else {
    vecquant3_gptq_gemv_kernel<scalar_t, 128, WithLora, FloatAccum>
        <<<blocks, threads, 0, stream>>>(
        vec_ptr, qweight.data_ptr<int>(), scale_ptr, qzeros.data_ptr<int>(),
        down_ptr, up_ptr, out.data_ptr<float>(), qweight_rows, width,
        qzeros_stride, rank);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <bool WithLora>
torch::Tensor launch_vecquant3_gptq_gemv(torch::Tensor vec,
                                         torch::Tensor qweight,
                                         torch::Tensor scales,
                                         torch::Tensor qzeros,
                                         torch::Tensor down,
                                         torch::Tensor up,
                                         int64_t group_size,
                                         int64_t accumulation_type) {
  validate_common_inputs(vec, qweight, scales, qzeros, group_size,
                         accumulation_type);
  if constexpr (WithLora) {
    TORCH_CHECK(down.is_cuda() && up.is_cuda(),
                "vecquant3 gemv_lora requires CUDA LoRA tensors");
    TORCH_CHECK(down.scalar_type() == vec.scalar_type() &&
                    up.scalar_type() == vec.scalar_type(),
                "vecquant3 gemv_lora LoRA dtype must match vec dtype");
    TORCH_CHECK(down.dim() == 1,
                "vecquant3 gemv_lora down must be a rank vector");
    TORCH_CHECK(up.dim() == 2 && up.size(0) == down.numel() &&
                    up.size(1) == qweight.size(1),
                "vecquant3 gemv_lora up must be [rank, out_features]");
    TORCH_CHECK(down.is_contiguous() && up.is_contiguous(),
                "vecquant3 gemv_lora LoRA tensors must be contiguous");
  }

  const bool use_float_accum = accumulation_type == kAccumulationFloat32;
  if (vec.scalar_type() == torch::kFloat16) {
    if (use_float_accum) {
      return launch_vecquant3_gptq_gemv_typed<half, WithLora, true>(
          vec, qweight, scales, qzeros, down, up, group_size);
    }
    return launch_vecquant3_gptq_gemv_typed<half, WithLora, false>(
        vec, qweight, scales, qzeros, down, up, group_size);
  }

  if (use_float_accum) {
    return launch_vecquant3_gptq_gemv_typed<__nv_bfloat16, WithLora, true>(
        vec, qweight, scales, qzeros, down, up, group_size);
  }
  return launch_vecquant3_gptq_gemv_typed<__nv_bfloat16, WithLora, false>(
      vec, qweight, scales, qzeros, down, up, group_size);
}

}  // namespace

torch::Tensor vecquant3_gptq_gemv_cuda(torch::Tensor vec,
                                       torch::Tensor qweight,
                                       torch::Tensor scales,
                                       torch::Tensor qzeros,
                                       int64_t group_size,
                                       int64_t accumulation_type) {
  return launch_vecquant3_gptq_gemv<false>(vec.reshape({-1}), qweight, scales,
                                           qzeros, torch::Tensor(),
                                           torch::Tensor(), group_size,
                                           accumulation_type);
}

torch::Tensor vecquant3_gptq_gemv_lora_cuda(torch::Tensor vec,
                                            torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            torch::Tensor down,
                                            torch::Tensor up,
                                            int64_t group_size,
                                            int64_t accumulation_type) {
  return launch_vecquant3_gptq_gemv<true>(vec.reshape({-1}), qweight, scales,
                                          qzeros, down.reshape({-1}), up,
                                          group_size, accumulation_type);
}
