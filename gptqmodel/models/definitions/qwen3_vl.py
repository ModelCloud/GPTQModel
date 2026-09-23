# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from .base_qwen3_vl import BaseQwen3VLGPTQ


class Qwen3_VLQModel(BaseQwen3VLGPTQ):
    pass


class Qwen3_VL_MoeQModel(BaseQwen3VLGPTQ):
    """Qwen3-VL MoE decoder with defused per-expert linear projections."""

    layer_modules_strict = False
    dynamic_expert_index = "num_experts"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    # The upstream Qwen3-VL-MoE checkpoint stores each expert as fused
    # gate_up_proj/down_proj tensors. Defuser exposes those as numbered
    # gate_proj/up_proj/down_proj modules before GPT-QModel walks the tree.
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "q_norm:!",
                "k_norm:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
            },
        },
    ]


__all__ = ["Qwen3_VLQModel", "Qwen3_VL_MoeQModel"]
