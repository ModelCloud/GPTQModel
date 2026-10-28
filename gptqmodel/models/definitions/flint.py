# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class FlintQModel(BaseQModel):
    """RecursiveSI Flint hybrid-attention MoE model."""

    require_trust_remote_code = False
    # The first two layers have no routed experts; skip those dynamic paths
    # while quantizing their dense MLP projections instead.
    layer_modules_strict = False
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
                # Flint's first two decoder layers use a dense SwiGLU MLP.
                "": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
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
