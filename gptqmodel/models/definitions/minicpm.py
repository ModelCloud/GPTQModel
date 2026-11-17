# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import W1W3W2MoELifecycleHooks


class MiniCPMGPTQ(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.minicpm.configuration_minicpm.MiniCPMConfig",
        model_class="gptqmodel.models.ports.minicpm.modeling_minicpm.MiniCPMForCausalLM",
    )
    dynamic_expert_index = "num_experts"
    moe_lifecycle_hooks = W1W3W2MoELifecycleHooks()
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        if getattr(getattr(model, "config", None), "num_experts", 0):
            tree = copy.deepcopy(tree)
            del tree[-1]["mlp"]
            tree[-1]["mlp:moe"] = {
                "gate": ("gate:!",),
                "experts": {"#": ("w1:0:gate", "w3:0:up", "w2:1:down")},
            }
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Original scaled residuals and sparse expert replay have no qualified fold/search.
        return []
