// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace gptqmodel_gptq_block {

constexpr int kWidth = 128;

// One CTA owns one output row. Each lane keeps one column in a register;
// the pivot lane publishes the error before the remaining columns update.
__global__ void block_update_kernel(
    float* work, const float* hinv, const float* scale, const float* zero,
    const int32_t* column_group,
    const float* code_lo, const float* code_hi, int64_t code_stride,
    float* quantized, float* errors, float* losses, int64_t rows,
    int64_t work_stride, int64_t hinv_stride_row, int64_t hinv_stride_col, int64_t scale_stride,
    int64_t zero_stride, int64_t column_group_stride, int64_t quant_stride, int64_t error_stride,
    int64_t loss_stride, float maxq) {
    const int64_t row = blockIdx.x;
    const int col = threadIdx.x;
    if (row >= rows) return;

    float value = work[row * work_stride + col];
    __shared__ float pivot_error;

    // Keep the eager column order, round-to-nearest-even and separate
    // multiply/subtract operations. The host compiles with --fmad=false.
    for (int pivot = 0; pivot < kWidth; ++pivot) {
        if (col == pivot) {
            const int32_t group = column_group[pivot * column_group_stride];
            const float row_scale = scale[group * scale_stride + row];
            const float row_zero = zero[group * zero_stride + row];
            const float diagonal = hinv[pivot * hinv_stride_row + pivot * hinv_stride_col];
            const float rounded = nearbyintf(__fdiv_rn(value, row_scale));
            const float shifted = __fadd_rn(rounded, row_zero);
            // The host may pass a per-(group, row) inclusive code range whose
            // reconstruction stays inside the destination dtype's finite
            // range. A null pointer keeps the natural ``[0, maxq]`` grid so
            // existing callers stay bitwise identical.
            const float lo = code_lo == nullptr ? 0.0f : code_lo[group * code_stride + row];
            const float hi = code_hi == nullptr ? maxq : code_hi[group * code_stride + row];
            // CUDA fminf/fmaxf discard NaNs; torch.clamp preserves them.
            const float code = isnan(shifted) ? shifted : fminf(fmaxf(shifted, lo), hi);
            const float q = __fmul_rn(row_scale, __fsub_rn(code, row_zero));
            const float residual = __fsub_rn(value, q);
            const float squared = __fmul_rn(residual, residual);
            const float denominator = __fmul_rn(diagonal, diagonal);
            losses[row * loss_stride + pivot] = __fdiv_rn(squared, denominator);
            pivot_error = __fdiv_rn(residual, diagonal);
            errors[row * error_stride + pivot] = pivot_error;
            quantized[row * quant_stride + pivot] = q;
        }
        __syncthreads();
        if (col >= pivot) {
            const float h = hinv[pivot * hinv_stride_row + col * hinv_stride_col];
            value = __fsub_rn(value, __fmul_rn(pivot_error, h));
        }
        __syncthreads();
    }
    work[row * work_stride + col] = value;
}

void block_update_cuda(
    at::Tensor work, const at::Tensor& hinv, const at::Tensor& scale,
    const at::Tensor& zero, const at::Tensor& column_group,
    const c10::optional<at::Tensor>& code_lo,
    const c10::optional<at::Tensor>& code_hi,
    at::Tensor quantized, at::Tensor errors,
    at::Tensor losses, int64_t maxq) {
    TORCH_CHECK(maxq >= 3 && maxq <= 255 && ((maxq + 1) & maxq) == 0,
                "maxq must be 2^bits - 1 for bits 2 through 8");
    const auto device = work.device();
    TORCH_CHECK(device.is_cuda(), "work must be CUDA");
    for (const auto& tensor : {hinv, scale, zero, quantized, errors, losses}) {
        TORCH_CHECK(tensor.device() == device, "all block tensors must use the same CUDA device");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, "block tensors must be float32");
    }
    TORCH_CHECK(work.scalar_type() == at::kFloat, "work must be float32");
    TORCH_CHECK(column_group.device() == device && column_group.scalar_type() == at::kInt,
                "column_group must be CUDA int32 on the work device");
    TORCH_CHECK(work.dim() == 2 && work.size(1) == kWidth && work.stride(1) == 1,
                "work must have 128 contiguous columns");
    TORCH_CHECK(hinv.sizes() == at::IntArrayRef({kWidth, kWidth}),
                "hinv must be 128 by 128");
    const auto rows = work.size(0);
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) > 0 && scale.size(1) == rows && scale.stride(1) == 1,
                "scale must have shape [groups, rows] with contiguous rows");
    TORCH_CHECK(zero.sizes() == scale.sizes(), "zero must match scale");
    TORCH_CHECK(zero.stride(1) == 1, "zero must have contiguous rows");
    TORCH_CHECK(column_group.dim() == 1 && column_group.size(0) == kWidth,
                "column_group must have 128 entries");
    for (const auto& tensor : {quantized, errors, losses}) {
        TORCH_CHECK(tensor.sizes() == work.sizes() && tensor.stride(1) == 1,
                    "outputs must match work and have contiguous columns");
    }

    // Optional per-(group, row) inclusive code range. Both entries must name
    // the same `[groups, rows]` layout as `scale`/`zero`; a single missing
    // entry is a caller bug because the kernel treats a null bound as the
    // natural grid for that endpoint only.
    TORCH_CHECK(code_lo.has_value() == code_hi.has_value(),
                "code_lo and code_hi must be provided together");
    const float* code_lo_ptr = nullptr;
    const float* code_hi_ptr = nullptr;
    int64_t code_stride = 0;
    if (code_lo.has_value()) {
        for (const auto* bound : {&code_lo.value(), &code_hi.value()}) {
            TORCH_CHECK(bound->device() == device, "code bounds must use the work device");
            TORCH_CHECK(bound->scalar_type() == at::kFloat, "code bounds must be float32");
            TORCH_CHECK(bound->sizes() == scale.sizes(), "code bounds must match [groups, rows]");
            TORCH_CHECK(bound->stride(1) == 1, "code bounds must have contiguous rows");
        }
        code_lo_ptr = code_lo->const_data_ptr<float>();
        code_hi_ptr = code_hi->const_data_ptr<float>();
        code_stride = code_lo->stride(0);
    }

    c10::cuda::CUDAGuard guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    block_update_kernel<<<rows, kWidth, 0, stream>>>(
        work.data_ptr<float>(), hinv.const_data_ptr<float>(),
        scale.const_data_ptr<float>(), zero.const_data_ptr<float>(),
        column_group.const_data_ptr<int32_t>(),
        code_lo_ptr, code_hi_ptr, code_stride,
        quantized.data_ptr<float>(), errors.data_ptr<float>(),
        losses.data_ptr<float>(), rows, work.stride(0), hinv.stride(0), hinv.stride(1),
        scale.stride(0), zero.stride(0), column_group.stride(0),
        quantized.stride(0), errors.stride(0),
        losses.stride(0), static_cast<float>(maxq));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gptqmodel_gptq_block
