#include "linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../quant/exl3_gemm.cuh"

void bc_linear_exl3_run
(
    at::Tensor trellis,
    at::Tensor suh,
    at::Tensor svh,
    int64_t K,
    const c10::optional<at::Tensor>& bias,
    bool mcg,
    bool mul1,
    const at::Tensor& x,
    at::Tensor& y
)
{
    TORCH_CHECK(K == trellis.size(-1) / 16, "K does not match packed trellis width");
    // The small-batch tensor-core GEMM reads the activation and writes the
    // result as densely packed rows.
    TORCH_CHECK(x.is_contiguous(), "bc_linear_exl3_run requires a contiguous input");
    TORCH_CHECK(y.is_contiguous(), "bc_linear_exl3_run requires a contiguous output");

    // The Hadamard-rotated input is written by the GEMM for the duration of
    // this call only.  It used to come from a process-wide per-shape buffer
    // shared by every module with the same in_features (the bsz == 1 decode
    // path), so two modules on different streams overwrote each other's
    // rotated input (#4619).  Allocate it per call for every batch size; the
    // caching allocator reuses the block for ordered work on the same stream.
    at::Tensor xh = at::empty_like(x);
    exl3_gemm(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0);

    if (bias)
        y.add_(bias.value());
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    bc_linear_exl3_run(trellis, suh, svh, K, bias, mcg, mul1, x, y);
}
