# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Qwen2MoeQModel(BaseQModel):
    """Qwen2 MoE definition aligned to the upstream HF forward execution order."""

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # The module tree is the forward-order ground truth for subset replay and
    # early-stop. Qwen2 executes the shared expert path before routed experts
    # in upstream HF modeling code, and Defuser should preserve that same
    # execution order when it unfuses the experts.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]
