#include "../../torch_stable_compat.h"
#include "linear.h"
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
    at::Tensor& xh,
    const at::Tensor& x,
    at::Tensor& y
)
{
    TORCH_CHECK(K == trellis.size(-1) / 16, "K does not match packed trellis width");

    if (x.numel() == x.size(-1))
    {
        exl3_gemm(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0);
    }
    else
    {
        at::Tensor xh_ = at::empty_like(x);
        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);
    }

    if (bias)
        at::add_(y, bias.value());
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    bc_linear_exl3_run(trellis, suh, svh, K, bias, mcg, mul1, xh, x, y);
}
