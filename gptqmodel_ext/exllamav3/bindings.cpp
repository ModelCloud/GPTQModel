#include <cuda_fp16.h>

#include <optional>
#include <torch/extension.h>
#include <torch/library.h>

#include "hadamard.h"
#include "hgemm.cuh"
#include "bf16_scale.h"

#include "quant/quantize.cuh"
#include "quant/frac.cuh"
#include "quant/pack.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"

#include "libtorch/linear.h"

namespace
{

void had_paley_wrapper(torch::Tensor& h)
{
    had_paley(h);
}

void had_paley2_wrapper(torch::Tensor& h)
{
    had_paley2(h);
}

void quantize_tiles_wrapper(
    torch::Tensor input_tiles,
    torch::Tensor& output_tiles,
    torch::Tensor& output_indices,
    torch::Tensor& temp_costs,
    torch::Tensor& temp_edges,
    int64_t K,
    bool mcg,
    bool mul1
)
{
    quantize_tiles(input_tiles, output_tiles, output_indices, temp_costs, temp_edges, static_cast<int>(K), mcg, mul1);
}

void pack_trellis_wrapper(torch::Tensor& packed, torch::Tensor unpacked, int64_t K)
{
    pack_trellis(packed, unpacked, static_cast<int>(K));
}

void unpack_trellis_wrapper(torch::Tensor& unpacked, torch::Tensor packed, int64_t K)
{
    unpack_trellis(unpacked, packed, static_cast<int>(K));
}

void pack_signs_wrapper(torch::Tensor& packed, torch::Tensor unpacked)
{
    pack_signs(packed, unpacked);
}

void reconstruct_wrapper(torch::Tensor& unpacked, torch::Tensor packed, int64_t K, bool mcg, bool mul1)
{
    reconstruct(unpacked, packed, static_cast<int>(K), mcg, mul1);
}

int checked_half_rate(int64_t K)
{
    TORCH_CHECK(K >= 1 && K <= 3, "EXL3 half-bit integral part must be 1..3");
    return static_cast<int>(K);
}

void quantize_tiles_frac_wrapper(
    torch::Tensor input_tiles, torch::Tensor& output_tiles, torch::Tensor& output_indices,
    torch::Tensor& temp_costs, torch::Tensor& temp_edges, int64_t K, int64_t mask)
{
    quantize_tiles_frac(input_tiles, output_tiles, output_indices, temp_costs, temp_edges, checked_half_rate(K), mask);
}

void pack_trellis_frac_wrapper(torch::Tensor& packed, torch::Tensor unpacked, int64_t K, int64_t mask)
{
    pack_trellis_frac(packed, unpacked, checked_half_rate(K), mask);
}

void unpack_trellis_frac_wrapper(torch::Tensor& unpacked, torch::Tensor packed, int64_t K, int64_t mask)
{
    unpack_trellis_frac(unpacked, packed, checked_half_rate(K), mask);
}

void reconstruct_frac_wrapper(torch::Tensor& unpacked, torch::Tensor packed, int64_t K)
{
    reconstruct_frac(unpacked, packed, checked_half_rate(K));
}

#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
void reconstruct_q4nx_wrapper(torch::Tensor& unpacked, torch::Tensor packed, int64_t K, bool half_k)
{
    TORCH_CHECK(K >= 2 && K <= 3, "Q4NX reconstruction requires K in 2..3");
    reconstruct_q4nx(unpacked, packed, static_cast<int>(K), half_k);
}
#endif

void reconstruct_r4nx_wrapper(torch::Tensor& unpacked, torch::Tensor packed, double rate)
{
    reconstruct_r4nx(unpacked, packed, rate);
}

void pack_trellis_r4nx_wrapper(torch::Tensor& packed, torch::Tensor states, int64_t kv)
{
    pack_trellis_r4nx(packed, states, static_cast<int>(kv));
}

void unpack_trellis_r4nx_wrapper(torch::Tensor& states, torch::Tensor packed, int64_t kv)
{
    unpack_trellis_r4nx(states, packed, static_cast<int>(kv));
}

void had_r_128_wrapper(
    torch::Tensor input,
    torch::Tensor& output,
    std::optional<torch::Tensor> pre_scale_opt,
    std::optional<torch::Tensor> post_scale_opt,
    double scale
)
{
    const c10::optional<at::Tensor> pre_scale = pre_scale_opt.has_value() ? c10::optional<at::Tensor>(pre_scale_opt.value()) : c10::nullopt;
    const c10::optional<at::Tensor> post_scale = post_scale_opt.has_value() ? c10::optional<at::Tensor>(post_scale_opt.value()) : c10::nullopt;
    had_r_128(input, output, pre_scale, post_scale, static_cast<float>(scale));
}

void hgemm_wrapper(torch::Tensor a, torch::Tensor b, torch::Tensor& c)
{
    hgemm(a, b, c);
}

void bc_linear_exl3_run_wrapper(
    torch::Tensor trellis,
    torch::Tensor suh,
    torch::Tensor svh,
    int64_t K,
    std::optional<torch::Tensor> bias_opt,
    bool mcg,
    bool mul1,
    torch::Tensor x,
    torch::Tensor& y
)
{
    const c10::optional<at::Tensor> bias = bias_opt.has_value() ? c10::optional<at::Tensor>(bias_opt.value()) : c10::nullopt;
    bc_linear_exl3_run(trellis, suh, svh, K, bias, mcg, mul1, x, y);
}

int64_t bc_linear_exl3_run_configured_wrapper(
    torch::Tensor trellis, torch::Tensor suh, torch::Tensor svh, double K,
    std::optional<torch::Tensor> bias, bool mcg, bool mul1, torch::Tensor x, torch::Tensor& y,
    bool fp16_gemv, int64_t int8_gemv, bool fused_gemm, bool q4nx, bool r4nx)
{
    return bc_linear_exl3_run_configured(trellis, suh, svh, K, bias, mcg, mul1, x, y, fp16_gemv, int8_gemv, fused_gemm, q4nx, r4nx);
}

int64_t bc_linear_exl3_run_hip_gemv_wrapper(
    torch::Tensor trellis, torch::Tensor suh, torch::Tensor svh, double K,
    std::optional<torch::Tensor> bias, bool mcg, bool mul1, torch::Tensor x, torch::Tensor& y,
    bool q4nx, int64_t int8_gemv)
{
    const c10::optional<at::Tensor> bias_tensor = bias.has_value() ? c10::optional<at::Tensor>(bias.value()) : c10::nullopt;
    return bc_linear_exl3_run_hip_gemv(trellis, suh, svh, K, bias_tensor, mcg, mul1, x, y, q4nx, int8_gemv);
}

std::tuple<torch::Tensor, torch::Tensor> bf16_prescale_wrapper(torch::Tensor x, double bound, std::optional<torch::Tensor> suh)
{
    return bf16_prescale(x, bound, suh);
}

template <bool result_has_bias>
torch::Tensor bf16_rescale_wrapper(
    torch::Tensor result,
    torch::Tensor scale,
    std::optional<torch::Tensor> bias_opt,
    int64_t n_cols,
    std::optional<at::ScalarType> output_dtype)
{
    const std::optional<at::Tensor> bias = bias_opt.has_value() ? std::optional<at::Tensor>(bias_opt.value()) : std::nullopt;
    return bf16_rescale(result, scale, bias, n_cols, result_has_bias, output_dtype.value_or(at::kBFloat16));
}

}  // namespace

TORCH_LIBRARY(gptqmodel_exllamav3, m)
{
    m.def("had_paley(Tensor(a!) h) -> ()");
    m.def("had_paley2(Tensor(a!) h) -> ()");
    m.def("quantize_tiles(Tensor input_tiles, Tensor(a!) output_tiles, Tensor(b!) output_indices, Tensor(c!) temp_costs, Tensor(d!) temp_edges, int K, bool mcg, bool mul1) -> ()");
    m.def("pack_trellis(Tensor(a!) packed, Tensor unpacked, int K) -> ()");
    m.def("unpack_trellis(Tensor(a!) unpacked, Tensor packed, int K) -> ()");
    m.def("pack_signs(Tensor(a!) packed, Tensor unpacked) -> ()");
    m.def("reconstruct(Tensor(a!) unpacked, Tensor packed, int K, bool mcg, bool mul1) -> ()");
    m.def("reconstruct_frac(Tensor(a!) unpacked, Tensor packed, int K) -> ()");
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
    m.def("reconstruct_q4nx(Tensor(a!) unpacked, Tensor packed, int K, bool half_k) -> ()");
#endif
    m.def("reconstruct_r4nx(Tensor(a!) unpacked, Tensor packed, float K) -> ()");
    m.def("pack_trellis_r4nx(Tensor(a!) packed, Tensor states, int kv) -> ()");
    m.def("unpack_trellis_r4nx(Tensor(a!) states, Tensor packed, int kv) -> ()");
    m.def("quantize_tiles_frac(Tensor input_tiles, Tensor(a!) output_tiles, Tensor(b!) output_indices, Tensor(c!) temp_costs, Tensor(d!) temp_edges, int K, int mask) -> ()");
    m.def("pack_trellis_frac(Tensor(a!) packed, Tensor unpacked, int K, int mask) -> ()");
    m.def("unpack_trellis_frac(Tensor(a!) unpacked, Tensor packed, int K, int mask) -> ()");
    m.def("had_r_128(Tensor input, Tensor(a!) output, Tensor? pre_scale, Tensor? post_scale, float scale) -> ()");
    m.def("hgemm(Tensor a, Tensor b, Tensor(a!) c) -> ()");
    m.def("bc_linear_exl3_run(Tensor trellis, Tensor suh, Tensor svh, int K, Tensor? bias, bool mcg, bool mul1, Tensor x, Tensor(b!) y) -> ()");
    m.def("bc_linear_exl3_run_configured(Tensor trellis, Tensor suh, Tensor svh, float K, Tensor? bias, bool mcg, bool mul1, Tensor x, Tensor(a!) y, bool fp16_gemv, int int8_gemv, bool fused_gemm=False, bool q4nx=False, bool r4nx=False) -> int");
    m.def("bc_linear_exl3_run_hip_gemv(Tensor trellis, Tensor suh, Tensor svh, float K, Tensor? bias, bool mcg, bool mul1, Tensor x, Tensor(b!) y, bool q4nx=False, int int8_gemv=0) -> int");
    m.def("bf16_prescale(Tensor x, float bound, Tensor? suh=None) -> (Tensor, Tensor)");
    m.def("bf16_rescale(Tensor result, Tensor scale, Tensor? bias, int n_cols, ScalarType? output_dtype=None) -> Tensor");
    m.def("bf16_rescale.raw(Tensor result, Tensor scale, Tensor? bias, int n_cols, ScalarType? output_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_exllamav3, CPU, m)
{
    m.impl("had_paley", &had_paley_wrapper);
    m.impl("had_paley2", &had_paley2_wrapper);
}

TORCH_LIBRARY_IMPL(gptqmodel_exllamav3, CUDA, m)
{
    m.impl("quantize_tiles", &quantize_tiles_wrapper);
    m.impl("pack_trellis", &pack_trellis_wrapper);
    m.impl("unpack_trellis", &unpack_trellis_wrapper);
    m.impl("pack_signs", &pack_signs_wrapper);
    m.impl("reconstruct", &reconstruct_wrapper);
    m.impl("reconstruct_frac", &reconstruct_frac_wrapper);
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
    m.impl("reconstruct_q4nx", &reconstruct_q4nx_wrapper);
#endif
    m.impl("reconstruct_r4nx", &reconstruct_r4nx_wrapper);
    m.impl("pack_trellis_r4nx", &pack_trellis_r4nx_wrapper);
    m.impl("unpack_trellis_r4nx", &unpack_trellis_r4nx_wrapper);
    m.impl("quantize_tiles_frac", &quantize_tiles_frac_wrapper);
    m.impl("pack_trellis_frac", &pack_trellis_frac_wrapper);
    m.impl("unpack_trellis_frac", &unpack_trellis_frac_wrapper);
    m.impl("had_r_128", &had_r_128_wrapper);
    m.impl("hgemm", &hgemm_wrapper);
    m.impl("bc_linear_exl3_run", &bc_linear_exl3_run_wrapper);
    m.impl("bc_linear_exl3_run_configured", &bc_linear_exl3_run_configured_wrapper);
    m.impl("bc_linear_exl3_run_hip_gemv", &bc_linear_exl3_run_hip_gemv_wrapper);
    m.impl("bf16_prescale", &bf16_prescale_wrapper);
    m.impl("bf16_rescale", &bf16_rescale_wrapper<true>);
    m.impl("bf16_rescale.raw", &bf16_rescale_wrapper<false>);
}
