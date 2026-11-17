# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class DeepSeekV3QModel(BaseQModel):
    shared_input_verified_model_types = frozenset({"deepseek_v3"})

    # Standalone Kimi-K2 uses the same architecture but has no native AutoConfig alias.
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.deepseek_v3.DeepseekV3Config",
        model_class="gptqmodel.models.ports.deepseek_v3.DeepseekV3ForCausalLM",
    )

    require_trust_remote_code = False

    require_fast_init = False

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    # The published DeepSeek-V3 checkpoint includes its non-decoder MTP layer.
    out_of_model_tensors = {"prefixes": ["model.layers.61"]}

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # DeepSeek V3 uses dynamic modules based on lora(rank):
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # DeepSeek-V3:
            #   self_attn uses split Q projections:
            #     - q_a_proj
            #     - q_b_proj
            #
            # Moonlight:
            #   self_attn uses a single Q projection:
            #     - q_proj
            "self_attn": (
                "q_a_layernorm:!", "kv_a_layernorm:!",
                "q_proj:0:q", "q_a_proj:0:q:in=h", "kv_a_proj_with_mqa:0:k:v:in=h",
                "q_b_proj:1:q:in=q_a", "kv_b_proj:1:k:v:in=kv_a", "o_proj:2:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "experts": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
            },
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # MLA latent norms and routed/shared consumers are not qualified for scale folding.
        return []
