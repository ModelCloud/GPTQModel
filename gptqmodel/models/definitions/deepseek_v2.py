# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


# Both DeepSeek-v2 and DeepSeek-v2-lite are supported in this model def
class DeepSeekV2QModel(BaseQModel):
    # deepseek_v2 requires custom model code
    require_trust_remote_code = True

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # DeepSeek V2-Lite uses dynamic modules based on lora(rank):
    # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L712
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_a_proj:0", "q_b_proj:0", "q_proj:0", "kv_a_proj_with_mqa:0", "kv_b_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        }
    ]
