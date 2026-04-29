#include <cuda_fp16.h>

#include <optional>
#include <torch/extension.h>
#include <torch/library.h>

#include "hadamard.h"
#include "hgemm.cuh"

#include "quant/quantize.cuh"
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
    torch::Tensor& xh,
    torch::Tensor x,
    torch::Tensor& y
)
{
    const c10::optional<at::Tensor> bias = bias_opt.has_value() ? c10::optional<at::Tensor>(bias_opt.value()) : c10::nullopt;
    bc_linear_exl3_run(trellis, suh, svh, K, bias, mcg, mul1, xh, x, y);
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
    m.def("had_r_128(Tensor input, Tensor(a!) output, Tensor? pre_scale, Tensor? post_scale, float scale) -> ()");
    m.def("hgemm(Tensor a, Tensor b, Tensor(a!) c) -> ()");
    m.def("bc_linear_exl3_run(Tensor trellis, Tensor suh, Tensor svh, int K, Tensor? bias, bool mcg, bool mul1, Tensor(a!) xh, Tensor x, Tensor(b!) y) -> ()");
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
    m.impl("had_r_128", &had_r_128_wrapper);
    m.impl("hgemm", &hgemm_wrapper);
    m.impl("bc_linear_exl3_run", &bc_linear_exl3_run_wrapper);
}
