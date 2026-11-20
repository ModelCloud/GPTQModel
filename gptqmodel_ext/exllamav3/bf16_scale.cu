// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// One-launch FP16/BF16 input protection and output re-scale for EXL3.
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
//   rescale:   unbiased result * scale + bias in FP32, then one RTNE cast
//              to BF16. The original overload also subtracts the bias from
//              inputs that include it before scaling.
//
// The division by a power of two is exact, so the fused results are
// bit-identical to the eager reference.  Everything stays on the device; no
// value is read back to the host. With BF16 inputs and checkpoint suh, the
// bound, scale and epilogue use FP64 to cover products beyond FP32's range;
// the native GEMM still consumes FP16 and accumulates in FP32.

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

// Preserve separate IEEE rounding in both the ordinary and wide BF16 path.
__device__ __forceinline__ float scale_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ double scale_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ float scale_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ double scale_add(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ float scale_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ double scale_sub(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ float scale_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ double scale_div(double a, double b) { return __ddiv_rn(a, b); }

template <typename InputT, typename ScaleT>
__global__ __launch_bounds__(kThreads)
void bf16_prescale_kernel
(
    const InputT* __restrict__ x,
    const __half* __restrict__ suh,
    __half* __restrict__ out,
    ScaleT* __restrict__ scale_out,
    int64_t n_cols,
    ScaleT bound
)
{
    const int64_t row = blockIdx.x;
    const InputT* __restrict__ x_row = x + row * n_cols;
    __half* __restrict__ out_row = out + row * n_cols;
    const int t = threadIdx.x;

    ScaleT local_amax = 0;
    int local_nan = 0;
    for (int64_t col = t; col < n_cols; col += kThreads)
    {
        const float value = bias_to_float<InputT>(x_row[col]);
        // fmaxf drops NaN; torch's inf-norm returns NaN for such a row, so
        // track it separately and restore it before the scale decision.
        local_nan |= static_cast<int>(isnan(value));
        local_amax = fmax(local_amax, static_cast<ScaleT>(fabsf(value)));
        if (suh)
        {
            const ScaleT scaled = scale_mul(static_cast<ScaleT>(value), static_cast<ScaleT>(__half2float(suh[col])));
            local_nan |= static_cast<int>(isnan(scaled));
            local_amax = fmax(local_amax, fabs(scaled));
        }
    }

    __shared__ ScaleT s_amax[kThreads];
    __shared__ int s_nan[kThreads];
    __shared__ ScaleT s_inv_scale;
    s_amax[t] = local_amax;
    s_nan[t] = local_nan;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (t < stride)
        {
            s_amax[t] = fmax(s_amax[t], s_amax[t + stride]);
            s_nan[t] |= s_nan[t + stride];
        }
        __syncthreads();
    }

    if (t == 0)
    {
        ScaleT amax = s_amax[0];
        if (s_nan[0])
        {
            amax = __int_as_float(0x7fc00000);  // quiet NaN
        }
        // BF16 checkpoint products and their power-of-two scales may exceed
        // FP32 even though cancellation leaves a finite final projection.
        const bool wide = (amax > bound) && isfinite(amax);
        ScaleT scale = 1;
        ScaleT inv_scale = 1;
        if (wide)
        {
            int exponent = 0;
            frexp(scale_div(amax, bound), &exponent);
            scale = ldexp(static_cast<ScaleT>(1), exponent);
            inv_scale = ldexp(static_cast<ScaleT>(1), -exponent);
        }
        scale_out[row] = scale;
        s_inv_scale = inv_scale;
    }
    __syncthreads();

    const ScaleT inv_scale = s_inv_scale;
    for (int64_t col = t; col < n_cols; col += kThreads)
    {
        const float value = bias_to_float<InputT>(x_row[col]);
        out_row[col] = __float2half_rn(static_cast<float>(scale_mul(static_cast<ScaleT>(value), inv_scale)));
    }
}

template <typename ResultT, typename BiasT, typename OutputT, typename ScaleT>
__global__ __launch_bounds__(kThreads)
void bf16_rescale_kernel
(
    const ResultT* __restrict__ result,
    const ScaleT* __restrict__ scale,
    const BiasT* __restrict__ bias,
    OutputT* __restrict__ out,
    int64_t n_cols,
    int64_t result_stride,
    int64_t out_stride,
    bool result_has_bias
)
{
    const int64_t row = blockIdx.x;
    const int64_t col = static_cast<int64_t>(blockIdx.y) * kThreads + threadIdx.x;
    if (col >= n_cols)
    {
        return;
    }
    const ScaleT row_scale = scale[row];
    ScaleT value = result_to_float<ResultT>(result[row * result_stride + col]);
    if (bias)
    {
        const ScaleT bias_value = bias_to_float<BiasT>(bias[col]);
        if (result_has_bias)
            value = scale_sub(value, bias_value);
        value = scale_add(scale_mul(value, row_scale), bias_value);
    }
    else
    {
        value = scale_mul(value, row_scale);
    }
    out[row * out_stride + col] = static_cast<OutputT>(value);
}

}  // namespace

namespace
{

template <typename ResultT, typename OutputT, typename ScaleT>
void launch_bf16_rescale_typed(
    const at::Tensor& result,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& bias,
    at::Tensor& out,
    int64_t n_cols,
    int64_t rows,
    int64_t result_stride,
    bool result_has_bias,
    cudaStream_t stream)
{
    const ResultT* result_ptr = reinterpret_cast<const ResultT*>(result.data_ptr());
    const ScaleT* scale_ptr = scale.data_ptr<ScaleT>();
    OutputT* out_ptr = out.data_ptr<OutputT>();
    const dim3 grid(rows, (n_cols + kThreads - 1) / kThreads);

    if (!bias.has_value())
    {
        bf16_rescale_kernel<ResultT, float, OutputT, ScaleT><<<grid, kThreads, 0, stream>>>(
            result_ptr, scale_ptr, nullptr, out_ptr, n_cols, result_stride, n_cols, false);
        return;
    }

    const at::Tensor& b = bias.value();
    switch (b.scalar_type())
    {
        case at::kHalf:
            bf16_rescale_kernel<ResultT, at::Half, OutputT, ScaleT><<<grid, kThreads, 0, stream>>>(
                result_ptr, scale_ptr, b.data_ptr<at::Half>(), out_ptr, n_cols, result_stride, n_cols, result_has_bias);
            break;
        case at::kBFloat16:
            bf16_rescale_kernel<ResultT, at::BFloat16, OutputT, ScaleT><<<grid, kThreads, 0, stream>>>(
                result_ptr, scale_ptr, b.data_ptr<at::BFloat16>(), out_ptr, n_cols, result_stride, n_cols, result_has_bias);
            break;
        case at::kFloat:
            bf16_rescale_kernel<ResultT, float, OutputT, ScaleT><<<grid, kThreads, 0, stream>>>(
                result_ptr, scale_ptr, b.data_ptr<float>(), out_ptr, n_cols, result_stride, n_cols, result_has_bias);
            break;
        default:
            TORCH_CHECK(false, "bf16_rescale bias must be float16, bfloat16 or float32");
    }
}

template <typename ResultT, typename OutputT>
void launch_bf16_rescale(
    const at::Tensor& result, const at::Tensor& scale, const std::optional<at::Tensor>& bias,
    at::Tensor& out, int64_t n_cols, int64_t rows, int64_t result_stride,
    bool result_has_bias, cudaStream_t stream)
{
    if (scale.scalar_type() == at::kDouble)
        launch_bf16_rescale_typed<ResultT, OutputT, double>(
            result, scale, bias, out, n_cols, rows, result_stride, result_has_bias, stream);
    else
        launch_bf16_rescale_typed<ResultT, OutputT, float>(
            result, scale, bias, out, n_cols, rows, result_stride, result_has_bias, stream);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> bf16_prescale(
    const at::Tensor& x, double bound, const std::optional<at::Tensor>& suh)
{
    TORCH_CHECK(x.is_cuda(), "bf16_prescale requires a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "bf16_prescale requires float16 or bfloat16");
    TORCH_CHECK(x.dim() >= 1 && x.numel() > 0, "bf16_prescale requires a non-empty tensor");
    TORCH_CHECK(x.is_contiguous(), "bf16_prescale requires a contiguous tensor");

    const int64_t n_cols = x.size(-1);
    if (suh.has_value())
    {
        TORCH_CHECK(suh->device() == x.device(), "bf16_prescale suh is on the wrong device");
        TORCH_CHECK(suh->scalar_type() == at::kHalf && suh->is_contiguous(),
                    "bf16_prescale requires contiguous float16 suh");
        TORCH_CHECK(suh->dim() == 1 && suh->numel() >= n_cols, "bf16_prescale suh width mismatch");
    }
    const int64_t rows = x.numel() / n_cols;
    at::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kHalf));
    const bool wide_scale = x.scalar_type() == at::kBFloat16 && suh.has_value();
    at::Tensor scale = at::empty({rows, 1}, x.options().dtype(wide_scale ? at::kDouble : at::kFloat));

    const c10::cuda::CUDAGuard guard(x.device());
    const __half* suh_ptr = suh.has_value()
        ? reinterpret_cast<const __half*>(suh->data_ptr<at::Half>()) : nullptr;
    AT_DISPATCH_SWITCH(x.scalar_type(), "bf16_prescale",
        AT_DISPATCH_CASE(at::kHalf, [&] {
            bf16_prescale_kernel<at::Half, float><<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<at::Half>(), suh_ptr, reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                scale.data_ptr<float>(), n_cols, static_cast<float>(bound));
        })
        AT_DISPATCH_CASE(at::kBFloat16, [&] {
            if (wide_scale)
                bf16_prescale_kernel<at::BFloat16, double><<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<at::BFloat16>(), suh_ptr, reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    scale.data_ptr<double>(), n_cols, bound);
            else
                bf16_prescale_kernel<at::BFloat16, float><<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<at::BFloat16>(), suh_ptr, reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    scale.data_ptr<float>(), n_cols, static_cast<float>(bound));
        })
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, scale};
}

at::Tensor bf16_rescale(
    const at::Tensor& result,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& bias,
    int64_t n_cols,
    bool result_has_bias,
    at::ScalarType output_dtype)
{
    TORCH_CHECK(result.is_cuda(), "bf16_rescale requires a CUDA tensor");
    TORCH_CHECK(
        result.scalar_type() == at::kHalf || result.scalar_type() == at::kFloat,
        "bf16_rescale requires a float16 or float32 result");
    TORCH_CHECK(result.dim() >= 1 && result.numel() > 0, "bf16_rescale requires a non-empty result");
    TORCH_CHECK(result.is_contiguous(), "bf16_rescale requires a contiguous result");
    TORCH_CHECK(n_cols > 0 && result.size(-1) >= n_cols, "bf16_rescale output width is out of range");
    TORCH_CHECK(scale.is_cuda() && scale.device() == result.device(), "bf16_rescale scale is on the wrong device");
    TORCH_CHECK((scale.scalar_type() == at::kFloat || scale.scalar_type() == at::kDouble) && scale.is_contiguous(),
                "bf16_rescale requires a contiguous float32 or float64 scale");
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
    TORCH_CHECK(output_dtype == at::kHalf || output_dtype == at::kBFloat16,
                "bf16_rescale requires float16 or bfloat16 output");
    at::Tensor out = at::empty(out_shape, result.options().dtype(output_dtype));

    const int64_t rows = result.numel() / result.size(-1);
    const int64_t result_stride = result.size(-1);

    const c10::cuda::CUDAGuard guard(result.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_SWITCH(output_dtype, "bf16_rescale_output",
        AT_DISPATCH_CASE(at::kHalf, [&] {
            if (result.scalar_type() == at::kHalf)
                launch_bf16_rescale<at::Half, at::Half>(result, scale, bias, out, n_cols, rows, result_stride, result_has_bias, stream);
            else
                launch_bf16_rescale<float, at::Half>(result, scale, bias, out, n_cols, rows, result_stride, result_has_bias, stream);
        })
        AT_DISPATCH_CASE(at::kBFloat16, [&] {
            if (result.scalar_type() == at::kHalf)
                launch_bf16_rescale<at::Half, at::BFloat16>(result, scale, bias, out, n_cols, rows, result_stride, result_has_bias, stream);
            else
                launch_bf16_rescale<float, at::BFloat16>(result, scale, bias, out, n_cols, rows, result_stride, result_has_bias, stream);
        })
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
