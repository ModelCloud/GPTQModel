#include "../../torch_stable_compat.h"
#include "linear.h"
#include "../util.h"
#include "../quant/bits_k.cuh"
#include "../quant/exl3_gemm.cuh"
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include "../quant/exl3_gemv.cuh"
#include "../quant/q4nx.cuh"
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
    at::Tensor accum = half_output ? at::empty_like(y, at::options(y).dtype(at::kFloat)) : y;
    const auto dispatch = exl3_gemm(x, trellis, accum, suh, xh, svh, mcg, mul1,
                                   fp16_gemv, int8_gemv, fused_gemm, q4nx, r4nx);

    if (bias)
        at::add_out(y, accum, bias.value());
    else if (half_output)
        torch::stable::copy_(y, accum);
    return dispatch;
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
    bool q4nx
)
{
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
    const auto bits = bits_from_K(K);
    TORCH_CHECK(trellis.dim() == 3 && trellis.size(-1) == 16 * bits.bits + 8 * bits.half,
                "K does not match packed trellis width");
    TORCH_CHECK(int(mcg) + int(mul1) + int(q4nx) <= 1, "conflicting EXL3 codebooks");
    TORCH_CHECK(!bits.half || mul1 || q4nx, "EXL3 half-integer bitrates require mul1 or Q4NX");
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
    at::Tensor accum = at::empty({size_m, size_n}, at::options(x).dtype(at::kFloat));
    at::Tensor xh = at::empty_like(x);

    const half* A_ptr = reinterpret_cast<const half*>(x.const_data_ptr());
    const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(trellis.const_data_ptr());
    void* C_ptr = accum.data_ptr();
    const half* su = reinterpret_cast<const half*>(suh.const_data_ptr());
    half* xh_ptr = reinterpret_cast<half*>(xh.data_ptr());
    const half* sv = reinterpret_cast<const half*>(svh.const_data_ptr());
    int* locks = nullptr;
    void* args[] = {
        (void*)&A_ptr, (void*)&B_ptr, (void*)&C_ptr, (void*)&size_m, (void*)&size_k,
        (void*)&size_n, (void*)&locks, (void*)&su, (void*)&xh_ptr, (void*)&sv
    };
    if (!exl3_gemv_try_launch(args, size_m, size_k, size_n, bits.bits, bits.half, cb,
                              /*output_fp32=*/true, x.get_device(), stream))
        return 0;

    if (bias)
        at::add_(accum, bias.value());
    torch::stable::copy_(at::view(y, {size_m, size_n}), accum);
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
