# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class GLM4MoEGPTQ(BaseQModel):
    # GLM-4.5-Air MoE Model Structure:
    # Layer 0: Standard MLP (no MoE experts) - handled by ["mlp.down_proj"], ["mlp.gate_proj"], ["mlp.up_proj"]
    # Layers 1-46: MoE with shared_experts and individual experts (128 experts total) - handled by MoE components
    # Layer 46: Additional special structure with expanded parameters (embed_tokens, shared_head, eh_proj, etc.)
    #   This is handled dynamically through layer_modules_strict = False
    #
    # allow dynamic expert index for layer_modules so we don't need to write out 128 layers here
    # config.n_routed_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # Set to False since GLM-4.5-Air may have dynamic module structures
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "q_norm:0:!","k_proj:0", "k_norm:0:!", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "shared_experts": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
                "gate": ("gate:!",), # Glm4MoeTopKRouter, ~1.6MB float32 per layer.  We really do not quant to quantize this.
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                # Standard MLP components for layer 0
                "gate_proj": ("gate_proj",),
                "down_proj": ("down_proj",),
                "up_proj": ("up_proj",),
            },
        }
    ]
