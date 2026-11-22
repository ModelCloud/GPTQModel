# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel, generate_node_for_awq_scaling
from ..local import LocalModelLoader


class QwenQModel(BaseQModel):
    loader = LocalModelLoader(
        "gptqmodel.models.ports.qwen.configuration_qwen.QWenConfig",
        "gptqmodel.models.ports.qwen.modeling_qwen.QWenLMHeadModel",
        tokenizer_class="gptqmodel.models.ports.qwen.tokenization_qwen.QWenTokenizer",
    )
    fusable = None

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # The fused c_attn emits Q|K|V, so it never folds into c_proj; drop it.
        nodes = [
            node for node in super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
            if node["prev_op"] is not module.attn.c_attn
        ]
        # The MLP is c_proj(w1(x) * silu(w2(x))). The planner takes the block's
        # last entry (w2, behind SiLU) as c_proj's producer; the linear w1
        # branch is the real one, so add that group explicitly.
        inp = input_feat.get("mlp.c_proj")
        if inp is not None and inp.numel() and all(node["prev_op"] is not module.mlp.w1 for node in nodes):
            kwargs = module_kwargs
            if isinstance(module_kwargs, dict):
                kwargs = {key: value for key, value in module_kwargs.items() if key != "_awq_feature_kwargs"}
            node, _ = generate_node_for_awq_scaling(
                inp=inp, prev_op=module.mlp.w1, module_kwargs=kwargs, nodes_size=len(nodes),
                subset=[module.mlp.c_proj], module2inspect=None,
            )
            node["_input_feature_name"] = "mlp.c_proj"
            nodes.append(node)
        return nodes

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!",),
            "attn": ("c_attn:0", "c_proj:1"),
            "ln_2": ("ln_2:!",),
            "mlp": ("w1:0", "w2:0:nosrc", "c_proj:1"),
        },
    ]
