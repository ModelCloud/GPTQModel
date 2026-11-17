# SPDX-License-Identifier: Apache-2.0
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks


class LLaDA2MoeQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.llada2.configuration_llada2_moe.LLaDA2MoeConfig",
        model_class="gptqmodel.models.ports.llada2.modeling_llada2_moe.LLaDA2MoeModelLM",
    )

    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"

    layer_modules_strict = False

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "attention": (
                "query_key_value:0",
                "query_layernorm:!",
                "key_layernorm:!",
                "dense:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!", "expert_bias:!"),
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Packed QKV, Q/K norms and biased routing have no qualified folds."""
        return []
