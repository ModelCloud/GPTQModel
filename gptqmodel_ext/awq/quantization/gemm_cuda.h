#include <torch/extension.h>

torch::Tensor gemm_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters);

torch::Tensor grouped_gemm_forward(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    torch::Tensor _topk_weights,
    torch::Tensor _sorted_token_ids_ptr,
    torch::Tensor _expert_ids_ptr,
    torch::Tensor _num_tokens_post_padded,
    bool mul_weights,
    int split_k_iters);

torch::Tensor gemmv2_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int group_size, int split_k_iters);

// Source - https://github.com/compressa-ai/AutoAWQ/blob/6673333456b8871522b11a7fb110de612edfdf95/awq_cuda/quantization/gemm_cuda.h#L9C1-L10C106
torch::Tensor dequantize_weights_cuda(torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters, int thx, int thy, bool dbg);