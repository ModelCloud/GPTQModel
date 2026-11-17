# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class KlearQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.klear.configuration_klear.KlearConfig",
        model_class="gptqmodel.models.ports.klear.modeling_klear.KlearMoeForCausalLM",
    )
    layer_modules_strict = False
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "q_norm:!", "k_norm:!", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "coefficient": ("coefficient:!",),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate_proj:0": ("gate_proj:0",),
                "up_proj:0": ("up_proj:0",),
                "down_proj:1": ("down_proj:1",),
            }
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Routing bias, Q/K norms and the learned shared/routed mixer are not
        # qualified for scale-search replay. Keep all declared linears in AWQ.
        return []
