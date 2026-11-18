# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from torch import nn

from ..base import generate_node_for_awq_scaling
from ..local import LocalModelLoader
from . import LlamaQModel


class DeciLMQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.decilm.configuration_decilm.DeciLMConfig",
        model_class="gptqmodel.models.ports.decilm.modeling_decilm.DeciLMForCausalLM",
    )
    layer_modules_strict = False  # Some model variants skip modules.

    # NAS can replace either subblock with a single bias-free projection.
    module_tree = ["model", "layers", "#", {
        "input_layernorm": ("input_layernorm:!:norm",),
        "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1:o",
                      "linear_attn:0"),
        "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
        "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down", "linear_mlp:0"),
    }]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Each NAS block declares its single-linear replacement last. On a full
        # block that entry is absent, so the planner keeps `input_layernorm` as
        # the producer: it folds the norm into `o_proj` and never emits the
        # `v_proj -> o_proj` / `up_proj -> down_proj` groups. Drop the stale
        # target and add the two linear producer groups explicitly.
        o_proj = getattr(getattr(module, "self_attn", None), "o_proj", None)
        nodes = []
        for node in super().awq_get_modules_for_scaling(module, input_feat, module_kwargs):
            if isinstance(node["prev_op"], nn.Linear) and o_proj is not None and node["layers"] == [o_proj]:
                continue
            node["layers"] = [layer for layer in node["layers"] if layer is not o_proj]
            if node["layers"]:
                nodes.append(node)
        kwargs = module_kwargs
        if isinstance(module_kwargs, dict):
            kwargs = {key: value for key, value in module_kwargs.items() if key != "_awq_feature_kwargs"}
        for parent, source, target in (("self_attn", "v_proj", "o_proj"), ("mlp", "up_proj", "down_proj")):
            owner = getattr(module, parent, None)
            prev_op, layer = getattr(owner, source, None), getattr(owner, target, None)
            inp = input_feat.get(f"{parent}.{target}")
            # GQA `v_proj` is narrower than `o_proj`'s input and cannot fold.
            if prev_op is None or layer is None or inp is None or not inp.numel():
                continue
            if prev_op.out_features != layer.in_features:
                continue
            node, _ = generate_node_for_awq_scaling(
                inp=inp, prev_op=prev_op, module_kwargs=kwargs, nodes_size=len(nodes),
                subset=[layer], module2inspect=None,
            )
            node["_input_feature_name"] = f"{parent}.{target}"
            nodes.append(node)
        return nodes
