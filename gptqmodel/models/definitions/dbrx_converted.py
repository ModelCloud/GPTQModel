# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class DbrxConvertedQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.dbrx_converted.configuration_dbrx.DbrxConfig",
        model_class="gptqmodel.models.ports.dbrx_converted.modeling_dbrx.DbrxForCausalLM",
    )

    # DBRX stores its expert cardinality under the nested ``ffn_config``; the
    # base ``_expert_count_config`` resolver follows that nesting.
    dynamic_expert_index = "moe_num_experts"

    pre_lm_head_norm_module = "transformer.norm_f"

    @classmethod
    def awq_input_feature_aggregation(cls, module_name):
        # Plain quantization has no scale group consuming the shared pre-router input.
        if module_name == "ffn":
            return {"mode": "token_rows"}
        return super().awq_input_feature_aggregation(module_name)

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Clipped QKV and routed expert fan-out are not qualified for scale folding.
        return []

    module_tree= [
        "transformer",
        "blocks",
        "#",
        {
            "norm_attn_norm": {
                "norm_1": ("norm_1:!",),
                "attn": ("q_proj:0", "k_proj:0", "v_proj:0", "out_proj:1"),
                "norm_2": ("norm_2:!",),
            },
            "ffn:moe": {
                "router": {
                    "layer": ("layer:!",),
                },
                "experts": {
                    "mlp": {
                        "#": ("w1:0", "v1:0", "w2:1"),
                    },
                },
            },
        }
    ]
