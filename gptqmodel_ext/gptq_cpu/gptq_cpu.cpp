#include "gptq_cpu.hpp"


namespace gptq_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void gemm_int4(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &zeros,
        const torch::Tensor &absmax, torch::Tensor &out, int64_t blocksize)
    {
        int64_t M = input.size(0);
        int64_t N = weight.size(0);
        int64_t K = input.size(1);
        // strides
        int64_t x_strideM = input.stride(0);
        int64_t out_strideM = out.stride(0);
        // Runtime CPU feature detection and dispatch
        if (CPUFeatures::hasAVX512BF16())
        {
            // Use AVX512 optimized implementation
            gptq_cpu::avx512::gemm_int4_inference<at::BFloat16>(
                M, N, K,
                input.data_ptr<at::BFloat16>(),
                weight.data_ptr<unsigned char>(),
                zeros.data_ptr<uint8_t>(),
                absmax.data_ptr<at::BFloat16>(),
                out.data_ptr<at::BFloat16>(),
                blocksize, x_strideM, out_strideM);
        }
        else
        {
            const int64_t packing_block_n = 32;
            int64_t N = weight.size(0);
            int64_t K_half = weight.size(1);
            auto qw = weight.reshape({-1, packing_block_n});
            auto low = torch::bitwise_and(qw, 0x0F);
            auto high = torch::bitwise_and(torch::bitwise_right_shift(qw, 4), 0x0F);
            auto restored = torch::cat({low, high}, 1);
            restored = restored.reshape({N / packing_block_n, K_half, packing_block_n, 2});
            restored = restored.transpose(-3, -2);
            auto unpacked_weight = restored.reshape({N, K_half * 2}).to(torch::kInt8);
            auto zeros_expanded = zeros.t().repeat_interleave(blocksize, 1);
            auto scales_expanded = absmax.t().repeat_interleave(blocksize, 1);
            auto original_weight = (unpacked_weight - zeros_expanded) * scales_expanded;
            auto weight_final = original_weight.t().to(input.dtype());
            torch::matmul_out(out, input, weight_final);
        }
    }
} // namespace gptq_cpu
