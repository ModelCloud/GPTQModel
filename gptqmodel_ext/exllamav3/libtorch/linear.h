#pragma once
#include "../../torch_stable_compat.h"

struct BC_LinearEXL3
{
    at::Tensor trellis;
    at::Tensor suh;
    at::Tensor svh;
    int K;
    c10::optional<at::Tensor> bias;
    bool mcg;
    bool mul1;

    BC_LinearEXL3
    (
        at::Tensor _trellis,
        at::Tensor _suh,
        at::Tensor _svh,
        int _K,
        c10::optional<at::Tensor> _bias,
        bool _mcg,
        bool _mul1
    ) :
        trellis(std::move(_trellis)),
        suh(std::move(_suh)),
        svh(std::move(_svh)),
        K(_K),
        bias(std::move(_bias)),
        mcg(_mcg),
        mul1(_mul1)
    {}

    void run(const at::Tensor& x, at::Tensor& y);
};

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
);

int64_t bc_linear_exl3_run_configured(
    at::Tensor trellis, at::Tensor suh, at::Tensor svh, double K,
    const c10::optional<at::Tensor>& bias, bool mcg, bool mul1,
    const at::Tensor& x, at::Tensor& y, bool fp16_gemv, int64_t int8_gemv, bool fused_gemm, bool q4nx);
