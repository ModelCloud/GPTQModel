// Based on the Hugging Face kernels-community `quantization-gptq` CPU kernel:
// https://github.com/huggingface/kernels-community/tree/c467920211e144ea621d9e8193ff71ec160ce256/quantization-gptq
// Original author: jiqing-feng <jiqing.feng@intel.com> (Intel).
// Modified for GPT-QModel by ModelCloud.ai.

#include "int4_qzero_group_scaled.hpp"

namespace int4_qzero_group_scaled_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    // This translation unit is compiled with -mavx512*, but this fallback is
    // the path for CPUs without AVX512-BF16. Keep the compiler from
    // auto-vectorizing it into instructions the caller cannot execute.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
#endif
    void gemm_int4(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &zeros,
        const torch::Tensor &absmax, const std::optional<torch::Tensor> &bias,
        torch::Tensor &out, int64_t blocksize)
    {
        int64_t M = input.size(0);
        int64_t N = weight.size(0);
        int64_t K = input.size(1);
        // strides
        int64_t x_strideM = input.stride(0);
        int64_t out_strideM = out.stride(0);
        const float *bias_ptr = bias.has_value() ? bias->const_data_ptr<float>() : nullptr;
        // Runtime CPU feature detection and dispatch
        if (CPUFeatures::hasAVX512BF16())
        {
            // Use AVX512 optimized implementation
            int4_qzero_group_scaled_cpu::avx512::gemm_int4_inference<at::BFloat16>(
                M, N, K,
                input.const_data_ptr<at::BFloat16>(),
                weight.const_data_ptr<unsigned char>(),
                zeros.const_data_ptr<uint8_t>(),
                absmax.const_data_ptr<float>(),
                bias_ptr,
                out.mutable_data_ptr<at::BFloat16>(),
                blocksize, x_strideM, out_strideM);
        }
        else
        {
            const int64_t packing_block_n = 32;
            const int64_t N = weight.size(0);
            const int64_t K_half = weight.size(1);
            const at::Tensor nibble_mask = at::full({}, 15.0, at::kByte);
            const at::Tensor shift = at::full({}, 4.0, at::kByte);
            auto qw = at::reshape(weight, {-1, packing_block_n});
            auto low = at::bitwise_and(qw, nibble_mask);
            auto high = at::bitwise_and(
                at::bitwise_right_shift(qw, shift), nibble_mask);
            auto restored = at::cat({low, high}, 1);
            restored = at::reshape(
                restored, {N / packing_block_n, K_half, packing_block_n, 2});
            restored = at::transpose(restored, -3, -2);
            auto unpacked_weight =
                at::to(at::reshape(restored, {N, K_half * 2}), at::kChar);
            auto zeros_t = at::contiguous(at::t(zeros));
            auto zeros_expanded = at::repeat_interleave(zeros_t, blocksize, 1);
            auto absmax_t = at::contiguous(at::t(absmax));
            auto scales_expanded = at::repeat_interleave(absmax_t, blocksize, 1);
            // The stable add shim always passes a floating alpha, which
            // integral tensors reject; subtract int8 directly instead.
            const at::Tensor zeros_i8 = at::contiguous(
                at::to(zeros_expanded, at::kChar));
            const at::Tensor unpacked_c = at::contiguous(unpacked_weight);
            at::Tensor diff =
                at::empty({N, K_half * 2}, at::options(unpacked_c));
            const int8_t* unpacked_ptr =
                unpacked_c.const_data_ptr<int8_t>();
            const int8_t* zeros_ptr = zeros_i8.const_data_ptr<int8_t>();
            int8_t* diff_ptr = diff.mutable_data_ptr<int8_t>();
            const int64_t count = N * K_half * 2;
            for (int64_t i = 0; i < count; ++i) {
                diff_ptr[i] = unpacked_ptr[i] - zeros_ptr[i];
            }
            auto original_weight = at::mul(diff, scales_expanded);
            auto weight_final = at::to(at::t(original_weight), input.scalar_type());
            const at::Tensor result = at::matmul(input, weight_final);
            torch::stable::copy_(out, result);
            if (bias.has_value())
            {
                at::add_(out, *bias);
            }
        }
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
} // namespace int4_qzero_group_scaled_cpu
