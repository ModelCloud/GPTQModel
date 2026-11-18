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
        """Generic AWQ groups without any fold into ``attention.dense``.

        ``attention.dense`` reads the attention output; the packed
        ``query_key_value`` output is wider and the optional Q/K norms are
        per-head, so no producer in the layer can absorb its scale. Without
        ``use_qk_norm`` the declared norms are absent and the generic planner
        falls back to the stale ``input_layernorm`` for ``dense``. The
        biased sigmoid router reads ``post_attention_layernorm`` and is
        compensated as a scale consumer, so routing is unchanged.
        """

        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        dense = getattr(getattr(module, "attention", None), "dense", None)
        kept = []
        for node in nodes:
            layers = [layer for layer in node.get("layers") or [] if layer is not dense]
            if not layers:
                continue
            if len(layers) != len(node.get("layers") or []):
                node = dict(node, layers=layers)
                search = node.get("_awq_search_layers")
                if search is not None:
                    node["_awq_search_layers"] = [layer for layer in search if layer is not dense]
            kept.append(node)
        return kept
