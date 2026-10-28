# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class FlintQModel(BaseQModel):
    """RecursiveSI Flint hybrid-attention MoE model."""

    require_trust_remote_code = False
    dynamic_expert_index = "num_experts"
    pre_lm_head_norm_module = "model.norm"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "k_norm:!",
                "softmax_offset:!",
                "q_proj:0:in=x",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
                "shared_expert": (
                    "gate_proj:0:in=x",
                    "up_proj:0:in=x",
                    "down_proj:1",
                ),
            },
        },
    ]


__all__ = ["FlintQModel"]
