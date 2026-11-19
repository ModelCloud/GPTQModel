// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// One-launch BF16 -> FP16 pre-scale and FP16/FP32 -> BF16 re-scale for EXL3.
//
// The eager torch sequence (per row: amax -> frexp(amax / bound) ->
// power-of-two scale -> x / scale -> fp16, then GEMM, then fp32 -> * scale ->
// bf16) is a dozen tiny launches.  At decode batch size those launches cost
// more than the GEMM.  These kernels perform the same arithmetic with the
// same rounding in one launch each:
//
//   prescale:  amax over the row, scale = 2**frexp_exponent(amax / bound) when
//              amax > bound (1.0 otherwise), x * 2**-exponent in FP32, then a
//              single RTNE cast to FP16.
//   rescale:   ((FP16 or FP32 result -> FP32) - bias) * scale + bias with
//              three separate RN FP32 roundings, then one RTNE cast to BF16.
//
// The division by a power of two is exact, so the fused results are
// bit-identical to the eager reference.  Everything stays on the device; no
// value is read back to the host.

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
// ROCm builds the same extension; map the CUDA spellings onto HIP's, as
// gptqmodel_ext/paro/rotation.cuh does for its shared header.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

using __nv_bfloat16 = __hip_bfloat16;

// HIP has no scalar ``__float2bfloat16_rn``: the only scalar spelling is
// ``__float2bfloat16``, which converts through clang's ``__bf16`` and therefore
// rounds to nearest-even, the same rounding as CUDA's ``__float2bfloat16_rn``.
// Map the CUDA name onto it so the kernel bodies below stay identical on both
// platforms, exactly as rotation.cuh maps ``__floats2bfloat162_rn``.
static __device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float value)
{
    return __float2bfloat16(value);
}
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "bf16_scale.h"

namespace
{

constexpr int kThreads = 256;

template <typename T>
__device__ __forceinline__ float bias_to_float(T value);

template <>
__device__ __forceinline__ float bias_to_float<at::Half>(at::Half value)
{
    return __half2float(*reinterpret_cast<const __half*>(&value));
}

template <>
__device__ __forceinline__ float bias_to_float<at::BFloat16>(at::BFloat16 value)
{
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&value));
}

template <>
__device__ __forceinline__ float bias_to_float<float>(float value)
{
    return value;
}

template <typename T>
__device__ __forceinline__ float result_to_float(T value);

template <>
__device__ __forceinline__ float result_to_float<at::Half>(at::Half value)
{
    return __half2float(*reinterpret_cast<const __half*>(&value));
}

template <>
__device__ __forceinline__ float result_to_float<float>(float value)
{
    return value;
}

__global__ __launch_bounds__(kThreads)
void bf16_prescale_kernel
(
    const __nv_bfloat16* __restrict__ x,
    __half* __restrict__ out,
    float* __restrict__ scale_out,
    int64_t n_cols,
    float bound
)
{
    const int64_t row = blockIdx.x;
    const __nv_bfloat16* __restrict__ x_row = x + row * n_cols;
    __half* __restrict__ out_row = out + row * n_cols;
    const int t = threadIdx.x;

    float local_amax = 0.0f;
    int local_nan = 0;
    for (int64_t col = t; col < n_cols; col += kThreads)
    {
        const float value = __bfloat162float(x_row[col]);
        // fmaxf drops NaN; torch's inf-norm returns NaN for such a row, so
        // track it separately and restore it before the scale decision.
        local_nan |= static_cast<int>(isnan(value));
        local_amax = fmaxf(local_amax, fabsf(value));
    }

    __shared__ float s_amax[kThreads];
    __shared__ int s_nan[kThreads];
    __shared__ float s_inv_scale;
    s_amax[t] = local_amax;
    s_nan[t] = local_nan;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (t < stride)
        {
            s_amax[t] = fmaxf(s_amax[t], s_amax[t + stride]);
            s_nan[t] |= s_nan[t + stride];
        }
        __syncthreads();
    }

    if (t == 0)
    {
        float amax = s_amax[0];
        if (s_nan[0])
        {
            amax = __int_as_float(0x7fc00000);  // quiet NaN
        }
        // ``frexp(amax / bound)``: for the normal quotient of a wide row the
        // exponent is the unbiased exponent field minus 126.  Rows inside the
        // bound (and non-finite rows, matching torch.frexp's exponent 0) keep
        // an exact scale of 1.0.
        const bool wide = (amax > bound) && (amax < __int_as_float(0x7f800000));
        float scale = 1.0f;
        float inv_scale = 1.0f;
        if (wide)
        {
            const float quotient = __fdiv_rn(amax, bound);
            const int exponent = (__float_as_int(quotient) >> 23 & 0xFF) - 126;
            scale = __int_as_float((exponent + 127) << 23);
            inv_scale = __int_as_float((127 - exponent) << 23);
        }
        scale_out[row] = scale;
        s_inv_scale = inv_scale;
    }
    __syncthreads();

    const float inv_scale = s_inv_scale;
    for (int64_t col = t; col < n_cols; col += kThreads)
    {
        const float value = __bfloat162float(x_row[col]);
        out_row[col] = __float2half_rn(__fmul_rn(value, inv_scale));
    }
}

template <typename ResultT, typename BiasT>
__global__ __launch_bounds__(kThreads)
void bf16_rescale_kernel
(
    const ResultT* __restrict__ result,
    const float* __restrict__ scale,
    const BiasT* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t n_cols,
    int64_t result_stride,
    int64_t out_stride,
    bool has_bias
)
{
    const int64_t row = blockIdx.x;
    const int64_t col = static_cast<int64_t>(blockIdx.y) * kThreads + threadIdx.x;
    if (col >= n_cols)
    {
        return;
    }
    const float row_scale = scale[row];
    float value = result_to_float<ResultT>(result[row * result_stride + col]);
    if (has_bias)
    {
        const float bias_value = bias_to_float<BiasT>(bias[col]);
        value = __fadd_rn(__fmul_rn(__fsub_rn(value, bias_value), row_scale), bias_value);
    }
    else
    {
        value = __fmul_rn(value, row_scale);
    }
    out[row * out_stride + col] = __float2bfloat16_rn(value);
}

}  // namespace

namespace
{

template <typename ResultT>
void launch_bf16_rescale(
    const at::Tensor& result,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& bias,
    at::Tensor& out,
    int64_t n_cols,
    int64_t rows,
    int64_t result_stride,
    cudaStream_t stream)
{
    const ResultT* result_ptr = reinterpret_cast<const ResultT*>(result.data_ptr());
    const float* scale_ptr = scale.data_ptr<float>();
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    const dim3 grid(rows, (n_cols + kThreads - 1) / kThreads);

    if (!bias.has_value())
    {
        bf16_rescale_kernel<ResultT, float><<<grid, kThreads, 0, stream>>>(
            result_ptr, scale_ptr, nullptr, out_ptr, n_cols, result_stride, n_cols, false);
        return;
    }

    const at::Tensor& b = bias.value();
    switch (b.scalar_type())
    {
        case at::kHalf:
            bf16_rescale_kernel<ResultT, at::Half><<<grid, kThreads, 0, stream>>>(
                result_ptr, scale_ptr, b.data_ptr<at::Half>(), out_ptr, n_cols, result_stride, n_cols, true);
            break;
        case at::kBFloat16:
            bf16_rescale_kernel<ResultT, at::BFloat16><<<grid, kThreads, 0, stream>>>(
                result_ptr, scale_ptr, b.data_ptr<at::BFloat16>(), out_ptr, n_cols, result_stride, n_cols, true);
            break;
        case at::kFloat:
            bf16_rescale_kernel<ResultT, float><<<grid, kThreads, 0, stream>>>(
                result_ptr, scale_ptr, b.data_ptr<float>(), out_ptr, n_cols, result_stride, n_cols, true);
            break;
        default:
            TORCH_CHECK(false, "bf16_rescale bias must be float16, bfloat16 or float32");
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> bf16_prescale(const at::Tensor& x, double bound)
{
    TORCH_CHECK(x.is_cuda(), "bf16_prescale requires a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "bf16_prescale requires bfloat16");
    TORCH_CHECK(x.dim() >= 1 && x.numel() > 0, "bf16_prescale requires a non-empty tensor");
    TORCH_CHECK(x.is_contiguous(), "bf16_prescale requires a contiguous tensor");

    const int64_t n_cols = x.size(-1);
    const int64_t rows = x.numel() / n_cols;
    at::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kHalf));
    at::Tensor scale = at::empty({rows, 1}, x.options().dtype(at::kFloat));

    const c10::cuda::CUDAGuard guard(x.device());
    bf16_prescale_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        n_cols,
        static_cast<float>(bound));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, scale};
}

at::Tensor bf16_rescale(
    const at::Tensor& result,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& bias,
    int64_t n_cols)
{
    TORCH_CHECK(result.is_cuda(), "bf16_rescale requires a CUDA tensor");
    TORCH_CHECK(
        result.scalar_type() == at::kHalf || result.scalar_type() == at::kFloat,
        "bf16_rescale requires a float16 or float32 result");
    TORCH_CHECK(result.dim() >= 1 && result.numel() > 0, "bf16_rescale requires a non-empty result");
    TORCH_CHECK(result.is_contiguous(), "bf16_rescale requires a contiguous result");
    TORCH_CHECK(n_cols > 0 && result.size(-1) >= n_cols, "bf16_rescale output width is out of range");
    TORCH_CHECK(scale.is_cuda() && scale.device() == result.device(), "bf16_rescale scale is on the wrong device");
    TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.is_contiguous(), "bf16_rescale requires a contiguous float32 scale");
    TORCH_CHECK(scale.numel() == result.numel() / result.size(-1), "bf16_rescale scale row count mismatch");
    if (bias.has_value())
    {
        const at::Tensor& b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.device() == result.device(), "bf16_rescale bias is on the wrong device");
        TORCH_CHECK(b.dim() == 1 && b.is_contiguous() && b.numel() >= n_cols, "bf16_rescale bias geometry mismatch");
        TORCH_CHECK(b.is_floating_point(), "bf16_rescale bias must be floating point");
    }

    std::vector<int64_t> out_shape(result.sizes().begin(), result.sizes().end() - 1);
    out_shape.push_back(n_cols);
    at::Tensor out = at::empty(out_shape, result.options().dtype(at::kBFloat16));

    const int64_t rows = result.numel() / result.size(-1);
    const int64_t result_stride = result.size(-1);

    const c10::cuda::CUDAGuard guard(result.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (result.scalar_type() == at::kHalf)
    {
        launch_bf16_rescale<at::Half>(
            result, scale, bias, out, n_cols, rows, result_stride, stream);
    }
    else
    {
        launch_bf16_rescale<float>(
            result, scale, bias, out, n_cols, rows, result_stride, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
