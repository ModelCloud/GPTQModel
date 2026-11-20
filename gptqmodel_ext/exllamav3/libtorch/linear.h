#pragma once

#include <ATen/Tensor.h>
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

// ROCm-only fast path: run the packed small-batch GEMV (native HIP kernels) and
// return 1, or 0 when the shape/device/codebook is not eligible so the caller
// falls back to reconstruct + hgemm. The op exists on every build so the
// extension's required-op check is platform independent; CUDA returns 0 because
// bc_linear_exl3_run_configured already reaches the tensor-core GEMV there.
int64_t bc_linear_exl3_run_hip_gemv(
    at::Tensor trellis, at::Tensor suh, at::Tensor svh, double K,
    const c10::optional<at::Tensor>& bias, bool mcg, bool mul1,
    const at::Tensor& x, at::Tensor& y, bool q4nx);
