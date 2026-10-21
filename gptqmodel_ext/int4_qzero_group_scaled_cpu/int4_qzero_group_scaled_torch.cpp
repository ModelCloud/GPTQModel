// Based on the Hugging Face kernels-community `quantization-gptq` CPU kernel:
// https://github.com/huggingface/kernels-community/tree/c467920211e144ea621d9e8193ff71ec160ce256/quantization-gptq
// Original author: jiqing-feng <jiqing.feng@intel.com> (Intel).
// Modified for GPT-QModel by ModelCloud.ai.

#include <torch/all.h>
#include "int4_qzero_group_scaled.hpp"
#include "int4_qzero_group_scaled_pack.hpp"
#include <limits>

// Forward implementation for CPU
torch::Tensor gemm_int4_qzero_group_scaled_cpu_forward(
    const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &zeros,
    const torch::Tensor &absmax, int64_t blocksize)
{
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kByte, "weight must be uint8");
    TORCH_CHECK(zeros.scalar_type() == at::kByte, "zeros must be uint8");
    TORCH_CHECK(absmax.scalar_type() == at::kBFloat16, "absmax must be bfloat16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(zeros.dim() == 2, "zeros must be 2D");
    TORCH_CHECK(absmax.dim() == 2, "absmax must be 2D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(zeros.is_contiguous(), "zeros must be contiguous");
    TORCH_CHECK(absmax.is_contiguous(), "absmax must be contiguous");
    TORCH_CHECK(blocksize > 0, "blocksize must be > 0");
    TORCH_CHECK(weight.size(0) % 32 == 0, "out_features must be divisible by 32");
    TORCH_CHECK(input.size(1) == weight.size(1) * 2, "input/weight feature dimensions differ");
    TORCH_CHECK(input.size(1) % blocksize == 0, "in_features must be divisible by blocksize");
    TORCH_CHECK(
        zeros.sizes() == absmax.sizes(),
        "zeros and absmax must have the same shape");
    TORCH_CHECK(
        zeros.size(0) == input.size(1) / blocksize,
        "zeros/absmax group count does not match input features");
    TORCH_CHECK(
        zeros.size(1) == weight.size(0),
        "zeros/absmax out_features does not match weight");

    auto output = at::empty({input.size(0), weight.size(0)}, input.options());
    if (input.size(0) == 0) {
        return output;
    }

    int4_qzero_group_scaled_cpu::gemm_int4(
        input,
        weight,
        zeros,
        absmax,
        output,
        blocksize
    );

    return output;
}

std::vector<torch::Tensor>
pack_gptq_int4_cpu_forward(
    const torch::Tensor &qweight, const torch::Tensor &qzeros,
    const torch::Tensor &ret_idx, int64_t out_features, int64_t in_features)
{
    return int4_qzero_group_scaled_cpu::pack_gptq_int4(
        qweight,
        qzeros,
        ret_idx,
        out_features,
        in_features);
}

std::vector<torch::Tensor>
pack_awq_int4_cpu_forward(
    const torch::Tensor &qweight, const torch::Tensor &qzeros,
    int64_t out_features, int64_t in_features)
{
    return int4_qzero_group_scaled_cpu::pack_awq_int4(
        qweight,
        qzeros,
        out_features,
        in_features);
}

int64_t parallel_thread_count_cpu(const torch::Tensor &input, int64_t rows)
{
    TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
    TORCH_CHECK(rows >= 0, "rows must be >= 0");
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "rows exceeds the supported kernel dimension");
    return int4_qzero_group_scaled_cpu::avx512::adjust_num_threads(static_cast<int>(rows));
}
