#include <torch/extension.h>
#include <torch/library.h>

#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"

namespace {

torch::Tensor gemm_forward_dispatch(torch::Tensor in_feats, torch::Tensor kernel,
                                    torch::Tensor scaling_factors, torch::Tensor zeros,
                                    int64_t split_k_iters, bool fp32_accum,
                                    bool fused_splitk_reduce) {
  return gemm_forward_cuda(in_feats, kernel, scaling_factors, zeros, static_cast<int>(split_k_iters),
                           fp32_accum, fused_splitk_reduce);
}

torch::Tensor gemm_forward_fp32_reduce_dispatch(torch::Tensor in_feats, torch::Tensor kernel,
                                                torch::Tensor scaling_factors, torch::Tensor zeros,
                                                int64_t split_k_iters, bool fp32_output,
                                                bool fused_splitk_reduce) {
  return gemm_forward_cuda_fp32_reduce(in_feats, kernel, scaling_factors, zeros,
                                       static_cast<int>(split_k_iters), fp32_output,
                                       fused_splitk_reduce);
}

torch::Tensor gemmv2_forward_dispatch(torch::Tensor in_feats, torch::Tensor kernel,
                                      torch::Tensor scaling_factors, torch::Tensor zeros,
                                      int64_t group_size, int64_t split_k_iters,
                                      bool fp32_output) {
  return gemmv2_forward_cuda(in_feats, kernel, scaling_factors, zeros, static_cast<int>(group_size),
                             static_cast<int>(split_k_iters), fp32_output);
}

torch::Tensor gemv_forward_dispatch(torch::Tensor in_feats, torch::Tensor kernel,
                                    torch::Tensor scaling_factors, torch::Tensor zeros,
                                    int64_t group_size, bool fp32_output) {
  return gemv_forward_cuda(in_feats, kernel, scaling_factors, zeros, static_cast<int>(group_size),
                           fp32_output);
}

torch::Tensor gemm_fast_forward_prefill_dispatch(torch::Tensor in_feats, torch::Tensor kernel,
                                                 torch::Tensor scaling_factors,
                                                 torch::Tensor zeros, bool fp32_output) {
  return gemm_forward_cuda_prefill(in_feats, kernel, scaling_factors, zeros, fp32_output);
}

torch::Tensor gemv_fast_forward_decode_dispatch(torch::Tensor in_feats, torch::Tensor kernel,
                                                torch::Tensor scaling_factors, torch::Tensor zeros,
                                                int64_t m, int64_t n, int64_t k,
                                                int64_t group_size, bool fp32_output) {
  return gemv_forward_cuda_decode(in_feats, kernel, scaling_factors, zeros, static_cast<int>(m),
                                  static_cast<int>(n), static_cast<int>(k),
                                  static_cast<int>(group_size), fp32_output);
}

torch::Tensor dequantize_weights_dispatch(torch::Tensor kernel, torch::Tensor scaling_factors,
                                          torch::Tensor zeros, int64_t split_k_iters, int64_t thx,
                                          int64_t thy, bool dbg) {
  return dequantize_weights_cuda(kernel, scaling_factors, zeros, static_cast<int>(split_k_iters),
                                 static_cast<int>(thx), static_cast<int>(thy), dbg);
}

} // namespace

TORCH_LIBRARY(gptqmodel_awq, m) {
  m.def("gemm_forward(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, int split_k_iters, bool fp32_accum=False, bool fused_splitk_reduce=True) -> Tensor");
  m.def("gemm_forward_fp32_reduce(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, int split_k_iters, bool fp32_output=False, bool fused_splitk_reduce=True) -> Tensor");
  m.def("gemmv2_forward(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, int group_size, int split_k_iters, bool fp32_output=False) -> Tensor");
  m.def("gemv_forward(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, int group_size, bool fp32_output=False) -> Tensor");
  m.def("gemm_fast_forward_prefill(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, bool fp32_output=False) -> Tensor");
  m.def("gemv_fast_forward_decode(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, int m, int n, int k, int group_size, bool fp32_output=False) -> Tensor");
  m.def("dequantize_weights(Tensor kernel, Tensor scaling_factors, Tensor zeros, int split_k_iters, int thx, int thy, bool dbg) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_awq, CUDA, m) {
  m.impl("gemm_forward", &gemm_forward_dispatch);
  m.impl("gemm_forward_fp32_reduce", &gemm_forward_fp32_reduce_dispatch);
  m.impl("gemmv2_forward", &gemmv2_forward_dispatch);
  m.impl("gemv_forward", &gemv_forward_dispatch);
  m.impl("gemm_fast_forward_prefill", &gemm_fast_forward_prefill_dispatch);
  m.impl("gemv_fast_forward_decode", &gemv_fast_forward_decode_dispatch);
  m.impl("dequantize_weights", &dequantize_weights_dispatch);
}
