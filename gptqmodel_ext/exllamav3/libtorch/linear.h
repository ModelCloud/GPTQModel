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
    const at::Tensor& x, at::Tensor& y, bool fp16_gemv, int64_t int8_gemv, bool fused_gemm, bool q4nx,
    bool r4nx = false);

// Folded FP16/BF16 input-range guard (#5942). Runs the native GEMM with the
// row bound, the power-of-two input scale and the output re-scale fused into
// its prologue and epilogue, and writes `y` in the activation's dtype. Returns
// the dispatch code, or -1 when the shape, device or dtype is not covered so
// the caller keeps the eager pre-scale path. `bound` is the FP16-safe per-row
// amax (65504 / sqrt(128)); `guard_mode` is 1 for FP16, 2 for BF16 with an
// FP32 bound, and 3 for BF16 whose checkpoint scales can exceed FP32.
int64_t bc_linear_exl3_run_guarded(
    at::Tensor trellis, at::Tensor suh, at::Tensor svh, double K,
    const c10::optional<at::Tensor>& bias, bool mcg, bool mul1,
    const at::Tensor& x, at::Tensor& y, bool fused_gemm, bool q4nx,
    int64_t guard_mode, double bound);

// ROCm-only fast path: run the packed small-batch GEMV (native HIP kernels) and
// return 2 when the int8-activation kernel ran, 1 when the fp16 kernel ran, or
// 0 when the shape/device/codebook is not eligible so the caller falls back to
// reconstruct + hgemm. int8_gemv mirrors the CUDA policy: 0 off, 1 residual
// (m == 1), 2 plain int8 (m <= 2). The op exists on every build so the
// extension's required-op check is platform independent; CUDA returns 0 because
// bc_linear_exl3_run_configured already reaches the tensor-core GEMV there.
int64_t bc_linear_exl3_run_hip_gemv(
    at::Tensor trellis, at::Tensor suh, at::Tensor svh, double K,
    const c10::optional<at::Tensor>& bias, bool mcg, bool mul1,
    const at::Tensor& x, at::Tensor& y, bool q4nx, int64_t int8_gemv);
