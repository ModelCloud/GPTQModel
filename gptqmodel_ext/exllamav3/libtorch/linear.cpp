#include "linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../quant/bits_k.cuh"
#include "../quant/exl3_gemm.cuh"
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include "../quant/exl3_gemv.cuh"
#include "../quant/q4nx.cuh"
#else
#include "../quant/exl3_devctx.cuh"
#endif

int64_t bc_linear_exl3_run_configured
(
    at::Tensor trellis,
    at::Tensor suh,
    at::Tensor svh,
    double K,
    const c10::optional<at::Tensor>& bias,
    bool mcg,
    bool mul1,
    const at::Tensor& x,
    at::Tensor& y,
    bool fp16_gemv,
    int64_t int8_gemv,
    bool fused_gemm,
    bool q4nx,
    bool r4nx
)
{
    const auto bits = bits_from_K(K);
    TORCH_CHECK(trellis.dim() == 3 && trellis.size(-1) == 16 * bits.bits + 8 * bits.half,
                "K does not match packed trellis width");
    TORCH_CHECK(!bits.half || mul1 || q4nx || r4nx, "EXL3 half-integer bitrates require mul1, Q4NX or R4NX");
    // The small-batch tensor-core GEMM reads the activation and writes the
    // result as densely packed rows.
    TORCH_CHECK(x.is_contiguous(), "bc_linear_exl3_run requires a contiguous input");
    TORCH_CHECK(y.is_contiguous(), "bc_linear_exl3_run requires a contiguous output");
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DTYPE(suh, kHalf);
    TORCH_CHECK_DTYPE(svh, kHalf);
    TORCH_CHECK(x.dim() >= 1 && y.dim() >= 1
                && suh.numel() == x.size(-1) && svh.numel() == y.size(-1),
                "EXL3 input and output scales must cover their respective widths");
    for (const auto& tensor : {x, y, trellis, suh, svh})
    {
        TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                    "EXL3 GEMM buffers must be contiguous and on the input device");
        TORCH_CHECK(!tensor.numel() || reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                    "EXL3 GEMM requires 16-byte aligned buffers");
    }

    // The Hadamard-rotated input is written by the GEMM for the duration of
    // this call only.  It used to come from a process-wide per-shape buffer
    // shared by every module with the same in_features (the bsz == 1 decode
    // path), so two modules on different streams overwrote each other's
    // rotated input (#4619).  Allocate it per call for every batch size; the
    // caching allocator reuses the block for ordered work on the same stream.
    at::Tensor xh = at::empty_like(x);
    // Packed GEMM uses its destination for cross-block partial sums, and
    // Q4NX normalizes those sums in the output Hadamard. Keep the destination
    // wide through split-K, the Hadamard and the output scale.
    const bool half_output = y.scalar_type() == at::kHalf;
    at::Tensor accum = half_output ? at::empty_like(y, y.options().dtype(at::kFloat)) : y;
    const auto dispatch = exl3_gemm(x, trellis, accum, suh, xh, svh, mcg, mul1,
                                   fp16_gemv, int8_gemv, fused_gemm, q4nx, r4nx);

    if (bias)
        at::add_out(y, accum, bias.value());
    else if (half_output)
        y.copy_(accum);
    return dispatch;
}

// Folded FP16/BF16 input-range guard (#5942). The native GEMM reduces each
// row's bound in its input-Hadamard pass, applies the row's power-of-two scale
// while narrowing the activation, and folds the re-scale and the bias into its
// epilogue, so a guarded forward costs no launch beyond the GEMM itself.
// Returns -1 on any shape, device or dtype the fused path does not cover; the
// caller then keeps the eager pre-scale reference.
int64_t bc_linear_exl3_run_guarded
(
    at::Tensor trellis,
    at::Tensor suh,
    at::Tensor svh,
    double K,
    const c10::optional<at::Tensor>& bias,
    bool mcg,
    bool mul1,
    const at::Tensor& x,
    at::Tensor& y,
    bool fused_gemm,
    bool q4nx,
    int64_t guard_mode,
    double bound
)
{
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
    // The folded guard lives in the tensor-core kernels, which have no ROCm
    // implementation yet.
    return -1;
#else
    if (guard_mode < 1 || guard_mode > 3) return -1;
    if (!x.is_cuda() || !y.is_cuda() || x.device() != y.device()) return -1;
    if (!x.is_contiguous() || !y.is_contiguous()) return -1;
    if (x.numel() == 0) return -1;
    if (x.dim() < 1 || y.dim() < 1) return -1;
    if (x.scalar_type() != (guard_mode == 1 ? at::kHalf : at::kBFloat16)) return -1;
    if (y.scalar_type() != x.scalar_type()) return -1;
    if (suh.numel() != x.size(-1) || svh.numel() != y.size(-1)) return -1;

    const auto bits = bits_from_K(K);
    if (trellis.dim() != 3 || trellis.size(-1) != 16 * bits.bits + 8 * bits.half) return -1;
    if (bits.half && !mul1 && !q4nx) return -1;
    if (int(mcg) + int(mul1) + int(q4nx) > 1) return -1;

    const int64_t rows = x.numel() / x.size(-1);
    if (rows <= 0 || rows != y.numel() / y.size(-1)) return -1;

    // Keep the Ampere Q4NX prefill reconstruction the unguarded dispatch
    // selects; the folded guard would otherwise replace it with the fused
    // kernel. Declining leaves the caller on the eager guard.
    if (q4nx && !fused_gemm)
    {
        const int64_t weight_elements = (int64_t) x.size(-1) * y.size(-1);
        if (g_get_cc((int) x.get_device()) == CC_AMPERE
            && weight_elements >= 4 * 1024 * 1024 && weight_elements <= 24 * 1024 * 1024
            && rows >= (weight_elements <= 12 * 1024 * 1024 ? 64 : 128))
            return -1;
    }

    for (const auto& tensor : {x, y, trellis, suh, svh})
    {
        TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                    "EXL3 GEMM buffers must be contiguous and on the input device");
        TORCH_CHECK(!tensor.numel() || reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                    "EXL3 GEMM requires 16-byte aligned buffers");
    }

    // The rotated input is written for the duration of this call only, like the
    // unguarded path (#4619). The accumulation buffer stays FP32 so the split-K
    // partial sums keep their precision (#5331); `y` receives the epilogue's
    // single rounded result.
    at::Tensor xh = at::empty_like(x);
    at::Tensor accum = at::empty_like(y, y.options().dtype(at::kFloat));

    Exl3Guard guard;
    guard.row_amax = at::empty({rows}, x.options().dtype(at::kDouble));
    guard.out = y;
    guard.mode = int(guard_mode);
    guard.bound = bound;
    if (bias)
    {
        const at::Tensor& b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.device() == x.device() && b.is_contiguous(),
                    "EXL3 guard bias must be a contiguous tensor on the input device");
        TORCH_CHECK(b.dim() == 1 && b.numel() >= y.size(-1),
                    "EXL3 guard bias must cover the output width");
        guard.bias = b.const_data_ptr();
        switch (b.scalar_type())
        {
            case at::kHalf: guard.bias_mode = 1; break;
            case at::kBFloat16: guard.bias_mode = 2; break;
            case at::kFloat: guard.bias_mode = 3; break;
            default: return -1;
        }
    }

    return exl3_gemm(x, trellis, accum, suh, xh, svh, mcg, mul1,
                     /*fp16_gemv=*/false, /*int8_gemv=*/0, fused_gemm, q4nx,
                     /*r4nx=*/false, &guard);
#endif
}

// ROCm decode-sized fast path. The packed GEMV keeps the trellis packed (no
// fp16 weight reconstruction) and runs the two Hadamard transforms and the
// output scale in the same three launches the CUDA kernel fuses with a grid
// barrier. Returning 0 leaves the caller on reconstruct + hgemm, which is also
// the reference the tests compare against.
int64_t bc_linear_exl3_run_hip_gemv
(
    at::Tensor trellis,
    at::Tensor suh,
    at::Tensor svh,
    double K,
    const c10::optional<at::Tensor>& bias,
    bool mcg,
    bool mul1,
    const at::Tensor& x,
    at::Tensor& y,
    bool q4nx,
    int64_t int8_gemv
)
{
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
    const auto bits = bits_from_K(K);
    TORCH_CHECK(trellis.dim() == 3 && trellis.size(-1) == 16 * bits.bits + 8 * bits.half,
                "K does not match packed trellis width");
    TORCH_CHECK(int(mcg) + int(mul1) + int(q4nx) <= 1, "conflicting EXL3 codebooks");
    TORCH_CHECK(!bits.half || mul1 || q4nx, "EXL3 half-integer bitrates require mul1 or Q4NX");
    TORCH_CHECK(int8_gemv >= 0 && int8_gemv <= 2, "EXL3 INT8 mode must be 0, 1 or 2");
    TORCH_CHECK(x.is_contiguous(), "bc_linear_exl3_run_hip_gemv requires a contiguous input");
    TORCH_CHECK(y.is_contiguous(), "bc_linear_exl3_run_hip_gemv requires a contiguous output");
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DTYPE(suh, kHalf);
    TORCH_CHECK_DTYPE(svh, kHalf);
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK(y.scalar_type() == at::kHalf || y.scalar_type() == at::kFloat,
                "EXL3 HIP GEMV output must be half or float");
    TORCH_CHECK(x.dim() >= 1 && y.dim() >= 1
                && suh.numel() == x.size(-1) && svh.numel() == y.size(-1),
                "EXL3 input and output scales must cover their respective widths");
    for (const auto& tensor : {x, y, trellis, suh, svh})
    {
        TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                    "EXL3 GEMM buffers must be contiguous and on the input device");
        TORCH_CHECK(!tensor.numel() || reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                    "EXL3 GEMM requires 16-byte aligned buffers");
    }

    const int size_m = int(x.numel() / x.size(-1));
    const int size_k = int(x.size(-1));
    const int size_n = int(y.size(-1));
    TORCH_CHECK(int64_t(size_m) * size_n == y.numel(),
                "inconsistent EXL3 input and output rows");

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int cb = q4nx ? EXL3_Q4NX_CODEBOOK : mcg ? 1 : mul1 ? 2 : 0;

    // The packed kernel writes FP32 partial sums and applies svh and the output
    // Hadamard itself; keep the result wide and cast once, like the CUDA path.
    at::Tensor accum = at::empty({size_m, size_n}, x.options().dtype(at::kFloat));
    at::Tensor xh = at::empty_like(x);

    const half* A_ptr = reinterpret_cast<const half*>(x.const_data_ptr());
    const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(trellis.const_data_ptr());
    void* C_ptr = accum.data_ptr();
    const half* su = reinterpret_cast<const half*>(suh.const_data_ptr());
    half* xh_ptr = reinterpret_cast<half*>(xh.data_ptr());
    const half* sv = reinterpret_cast<const half*>(svh.const_data_ptr());
    int* locks = nullptr;
    // The folded input-range guard (#5942) is CUDA-only; keep the shared
    // EXL3_GEMM_ARGS layout so the kernel signature stays one definition.
    double* row_amax = nullptr;
    void* guard_out = nullptr;
    const void* guard_bias = nullptr;
    const int guard_mode = 0;
    const int guard_bias_mode = 0;
    const double guard_bound = 0.0;
    void* args[] = {
        (void*)&A_ptr, (void*)&B_ptr, (void*)&C_ptr, (void*)&size_m, (void*)&size_k,
        (void*)&size_n, (void*)&locks, (void*)&su, (void*)&xh_ptr, (void*)&sv,
        (void*)&row_amax, (void*)&guard_out, (void*)&guard_bias,
        (void*)&guard_mode, (void*)&guard_bias_mode, (void*)&guard_bound
    };

    // The int8-activation GEMV shares the CUDA path's opt-in gate and row limits. It declines
    // (before touching accum) on any shape, rate or device it does not cover, including every
    // codebook but mul1.
    if (int8_gemv && mul1 && !mcg && !q4nx
        && size_m <= (int8_gemv == 1 ? 1 : 2)
        && exl3_gemv_int8(x, trellis, accum, suh, xh, svh, int(int8_gemv), stream))
    {
        if (bias)
            accum.add_(bias.value());
        y.view({size_m, size_n}).copy_(accum);
        return 2;
    }

    if (!exl3_gemv_try_launch(args, size_m, size_k, size_n, bits.bits, bits.half, cb,
                              /*output_fp32=*/true, x.get_device(), stream))
        return 0;

    if (bias)
        accum.add_(bias.value());
    y.view({size_m, size_n}).copy_(accum);
    return 1;
#else
    return 0;
#endif
}

void bc_linear_exl3_run(
    at::Tensor trellis, at::Tensor suh, at::Tensor svh, int64_t K,
    const c10::optional<at::Tensor>& bias, bool mcg, bool mul1, const at::Tensor& x, at::Tensor& y)
{
    bc_linear_exl3_run_configured(trellis, suh, svh, K, bias, mcg, mul1, x, y, false, 0, false, false, false);
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    bc_linear_exl3_run(trellis, suh, svh, K, bias, mcg, mul1, x, y);
}
