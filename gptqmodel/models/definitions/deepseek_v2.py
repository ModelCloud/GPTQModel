# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


# Both DeepSeek-v2 and DeepSeek-v2-lite are supported in this model def
class DeepSeekV2QModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.deepseek_v2.configuration_deepseek_v2.DeepseekV2Config",
        model_class="gptqmodel.models.ports.deepseek_v2.modeling_deepseek_v2.DeepseekV2ForCausalLM",
    )

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # DeepSeek V2-Lite uses dynamic modules based on lora(rank):
    # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L712
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Split the MLA projections the same way as DeepSeek-V3: q_a/kv_a
            # produce the low-rank states, q_b/kv_b consume them, and o_proj
            # consumes the attention output built from kv_b.
            "self_attn": (
                "q_a_proj:0:q", "q_a_layernorm:!", "q_b_proj:1:q:in=q_a", "q_proj:0:q",
                "kv_a_proj_with_mqa:0:k:v", "kv_a_layernorm:!", "kv_b_proj:1:k:v:in=kv_a", "o_proj:2:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "experts": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
                "shared_experts": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
            },
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # MLA normalizes/splits low-rank states before expanding K/V; MoE has
        # an unquantized router and shared consumers. Their complete folding
        # and replay objective is unqualified: use AWQ's plain-quantization path.
        return []
