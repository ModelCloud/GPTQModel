// Adapted from https://github.com/HandH1998/QQQ

#include <torch/library.h>

#include "qqq_gemm.h"

namespace {

void qqq_gemm_dispatch(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
                       torch::Tensor& D, const torch::Tensor& s1, const torch::Tensor& s2,
                       const torch::Tensor& s3, torch::Tensor& workspace, int64_t thread_k,
                       int64_t thread_n, int64_t sms, int64_t max_par) {
  qqq_gemm(A, B, C, D, s1, s2, s3, workspace, static_cast<int>(thread_k),
           static_cast<int>(thread_n), static_cast<int>(sms), static_cast<int>(max_par));
}

}  // namespace

TORCH_LIBRARY(gptqmodel_qqq, m) {
  m.def(
      "qqq_gemm(Tensor A, Tensor B, Tensor(a!) C, Tensor(a!) D, Tensor s1, Tensor s2, Tensor s3, "
      "Tensor(a!) workspace, int thread_k=-1, int thread_n=-1, int sms=-1, int max_par=8) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_qqq, CUDA, m) {
  m.impl("qqq_gemm", &qqq_gemm_dispatch);
}
