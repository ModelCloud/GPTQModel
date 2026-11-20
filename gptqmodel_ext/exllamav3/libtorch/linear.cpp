#include "../../torch_stable_compat.h"
#include "linear.h"
#include "../util.h"
#include "../quant/bits_k.cuh"
#include "../quant/exl3_gemm.cuh"

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
    bool q4nx
)
{
    const auto bits = bits_from_K(K);
    TORCH_CHECK(trellis.dim() == 3 && trellis.size(-1) == 16 * bits.bits + 8 * bits.half,
                "K does not match packed trellis width");
    TORCH_CHECK(!bits.half || mul1 || q4nx, "EXL3 half-integer bitrates require mul1 or Q4NX");
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
                                   fp16_gemv, int8_gemv, fused_gemm, q4nx);

    if (bias)
        at::add_out(y, accum, bias.value());
    else if (half_output)
        torch::stable::copy_(y, accum);
    return dispatch;
}

void bc_linear_exl3_run(
    at::Tensor trellis, at::Tensor suh, at::Tensor svh, int64_t K,
    const c10::optional<at::Tensor>& bias, bool mcg, bool mul1, const at::Tensor& x, at::Tensor& y)
{
    bc_linear_exl3_run_configured(trellis, suh, svh, K, bias, mcg, mul1, x, y, false, 0, false, false);
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    bc_linear_exl3_run(trellis, suh, svh, K, bias, mcg, mul1, x, y);
}
