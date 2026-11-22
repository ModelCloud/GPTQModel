# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import List

import torch

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Xing4_0QModel(BaseQModel):
    """Quantization layout for XingChen-AGI's Xing4.0 MLA/HC MoE model."""

    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.xing4_0.configuration_xing4_0.Xing4_0Config",
        model_class="gptqmodel.models.ports.xing4_0.modeling_xing4_0.Xing4_0ForCausalLM",
        tokenizer_class="gptqmodel.models.ports.xing4_0.tokenization_xing4_0.Xing4_0Tokenizer",
    )

    # The first two decoder layers use the dense MLP; later layers use routed
    # experts plus a shared expert.  Keep the tree non-strict for that mix.
    layer_modules_strict = False
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"
    out_of_model_tensors = {"prefixes": ["model.layers.40"]}

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "attn_hc:!": (
                "input_norm:!",
                "hc_fn:!",
                "hc_base:!",
                "hc_scale:!",
            ),
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                # MLA has a low-rank Q/KV path.  q_proj is the alternate
                # dense-Q layout used when q_lora_rank is disabled.
                "q_proj:0:in=h",
                "q_a_proj:0:in=h",
                "kv_a_proj_with_mqa:0:in=h",
                "q_a_layernorm:!",
                "kv_a_layernorm:!",
                "q_b_proj:1:in=q_a",
                "kv_b_proj:1:in=kv_a",
                "o_proj:2",
            ),
            "ffn_hc:!": (
                "input_norm:!",
                "hc_fn:!",
                "hc_base:!",
                "hc_scale:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Dense fallback for the first two layers.
                "": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                # The router and its correction buffer stay in native dtype.
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "experts": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
                "shared_experts": (
                    "gate_proj:0:in=x",
                    "up_proj:0:in=x",
                    "down_proj:1",
                ),
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Drop MLA latent-norm folds whose consumers do not read that norm.

        `q_b_proj` and `kv_b_proj` share one tree block, so the generic planner
        takes the block's last declared module (`kv_a_layernorm`) as the source
        for both when the two LoRA ranks are equal. `q_b_proj` reads
        `q_a_layernorm`, so that fold changes the attention output. A latent
        group survives only when every consumer reads its source norm.
        """

        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        attention = getattr(module, "self_attn", None)
        readers = {}
        for norm_name, consumer_name in (("q_a_layernorm", "q_b_proj"), ("kv_a_layernorm", "kv_b_proj")):
            norm = getattr(attention, norm_name, None)
            if norm is not None:
                readers[id(norm)] = getattr(attention, consumer_name, None)
        return [
            node
            for node in nodes
            if id(node.get("prev_op")) not in readers
            or all(target is readers[id(node["prev_op"])] for target in node.get("layers") or [])
        ]

    def lm_head_pre_quantize_generate_hook(
        self, inputs: List[List[torch.Tensor]]
    ) -> List[List[torch.Tensor]]:
        """Collapse Xing4.0's hyperconnection streams before the final norm."""

        for element in inputs:
            for index, stream in enumerate(element):
                if torch.is_tensor(stream) and stream.ndim == 4:
                    element[index] = stream.mean(dim=2)

        return super().lm_head_pre_quantize_generate_hook(inputs)


__all__ = ["Xing4_0QModel"]
