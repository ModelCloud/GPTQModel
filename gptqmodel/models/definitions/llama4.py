# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForImageTextToText

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Llama4QModel(BaseQModel):
    # some bug in the attention_mask of transformers.modeling_llama4,
    # so batch quantization for Llama4 is temporarily not supported.
    support_batch_quantize = False
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = "language_model.model.norm"

    dynamic_expert_index = "num_local_experts"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "feed_forward:moe": {
                "router": ("router:!",), # Llama4Router.forward() returns two values, skipping its quantization.
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        }
    ]
