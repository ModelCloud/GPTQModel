// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "../torch_stable_compat.h"

#include "quant/exl3_moe_coop.cuh"

namespace {

void run_exl3_moe_coop(
    const at::Tensor& x,
    const at::Tensor& selected_experts,
    const at::Tensor& routing_weights,
    int64_t min_expert,
    int64_t max_expert,
    int64_t input_features,
    const at::Tensor& gate_trellis, const at::Tensor& gate_su, const at::Tensor& gate_sv,
    const at::Tensor& up_trellis, const at::Tensor& up_su, const at::Tensor& up_sv,
    const at::Tensor& down_trellis, const at::Tensor& down_su, const at::Tensor& down_sv,
    const std::optional<at::Tensor>& gate_bias,
    const std::optional<at::Tensor>& up_bias,
    const std::optional<at::Tensor>& down_bias,
    double gate_bits, double up_bits, double down_bits,
    bool mcg, bool mul1,
    int64_t activation, double activation_limit, bool gated,
    at::Tensor& had_gate, at::Tensor& had_up,
    at::Tensor& gate_out, at::Tensor& up_out,
    at::Tensor& activated, at::Tensor& down_out,
    at::Tensor& counters, at::Tensor& out,
    const std::optional<at::Tensor>& shared_out,
    const std::optional<at::Tensor>& shared_gate,
    bool q4nx,
    const std::optional<at::Tensor>& flash_input_scale
) {
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    TORCH_CHECK(min_expert >= -1 && min_expert <= int_max && max_expert >= -1 && max_expert <= int_max,
                "exl3_moe_coop: invalid local expert range");
    TORCH_CHECK(input_features > 0 && input_features <= int_max && activation >= 0 && activation <= 3,
                "exl3_moe_coop: invalid projection width or activation");
    exl3_moe_coop(
        x, selected_experts, routing_weights,
        static_cast<int>(min_expert), static_cast<int>(max_expert), static_cast<int>(input_features),
        gate_trellis, gate_su, gate_sv, up_trellis, up_su, up_sv, down_trellis, down_su, down_sv,
        gate_bias, up_bias, down_bias,
        static_cast<float>(gate_bits), static_cast<float>(up_bits), static_cast<float>(down_bits), mcg, mul1,
        static_cast<int>(activation), static_cast<float>(activation_limit), gated,
        had_gate, had_up, gate_out, up_out, activated, down_out, counters, out, shared_out, shared_gate, q4nx,
        flash_input_scale
    );
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(gptqmodel_exllamav3, m) {
    m.def(
        "exl3_moe_coop(Tensor x, Tensor selected_experts, Tensor routing_weights, "
        "int min_expert, int max_expert, int input_features, "
        "Tensor gate_trellis, Tensor gate_su, Tensor gate_sv, "
        "Tensor up_trellis, Tensor up_su, Tensor up_sv, Tensor down_trellis, Tensor down_su, Tensor down_sv, "
        "Tensor? gate_bias, Tensor? up_bias, Tensor? down_bias, "
        "float gate_bits, float up_bits, float down_bits, bool mcg, bool mul1, "
        "int activation, float activation_limit, bool gated, "
        "Tensor(a!) had_gate, Tensor(b!) had_up, Tensor(c!) gate_out, Tensor(d!) up_out, "
        "Tensor(e!) activated, Tensor(f!) down_out, Tensor(g!) counters, Tensor(h!) out, "
        "Tensor? shared_out=None, Tensor? shared_gate=None, bool q4nx=False, Tensor? flash_input_scale=None) -> ()"
    );
}

STABLE_TORCH_LIBRARY_IMPL(gptqmodel_exllamav3, CUDA, m) {
    m.impl("exl3_moe_coop", TORCH_BOX(&run_exl3_moe_coop));
}
