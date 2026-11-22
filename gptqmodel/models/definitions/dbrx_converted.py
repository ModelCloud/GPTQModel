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
                        # `w2(act(w1(x)) * v1(x))`: `v1` is the linear (up)
                        # branch. Without the role, `w1` was inferred as the
                        # only gate/up member and folded across the activation.
                        "#": ("w1:0:gate", "v1:0:up", "w2:1:down"),
                    },
                },
            },
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        attention = getattr(getattr(module, "norm_attn_norm", None), "attn", None)
        if getattr(attention, "clip_qkv", None) is None:
            return nodes
        # `clip_qkv` clamps the q/k/v outputs, so dividing `v_proj`'s rows by a
        # scale is not undone by `out_proj`. The norm -> q/k/v fold keeps the
        # projection outputs, and therefore the clamp, exact.
        v_proj = getattr(attention, "v_proj", None)
        return [node for node in nodes if node.get("prev_op") is not v_proj]
