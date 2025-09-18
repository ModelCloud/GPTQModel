#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "layernorm/layernorm.h"
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"
#include "position_embedding/pos_encoding.h"
#include "vllm/moe_alig_block.h"
#include "vllm/activation.h"
#include "vllm/topk_softmax_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel");
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
    m.def("grouped_gemm_forward", &grouped_gemm_forward, "Quantized grouped GEMM kernel.");
    m.def("gemmv2_forward_cuda", &gemmv2_forward_cuda, "Quantized v2 GEMM kernel.");
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    m.def("rotary_embedding_neox", &rotary_embedding_neox, "Apply GPT-NeoX style rotary embedding to query and key");
    m.def("dequantize_weights_cuda", &dequantize_weights_cuda, "Dequantize weights.");
    m.def("moe_alig_block_size", &moe_alig_block_size, "Aligning the number of tokens to be processed by each expert such that it is divisible by the block size.");
    m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
    m.def("topk_softmax", &topk_softmax, "Computes fused topk and softmax operation.");
}