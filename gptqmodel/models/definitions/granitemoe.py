# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class GraniteMoeQModel(BaseQModel):
    """Quantization definition for the Transformers Granite MoE decoder.

    Defuser expands Granite's packed expert projections into numbered expert
    containers. Each container exposes one ``linear`` module for the input
    and output projection, which are the leaves targeted by this tree.
    """

    layer_modules_strict = False
    dynamic_expert_index = "num_local_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "block_sparse_moe:moe:?": {
                "input_linear": {
                    "#": ("linear:0",),
                },
                "output_linear": {
                    "#": ("linear:1",),
                },
            },
        },
    ]


__all__ = ["GraniteMoeQModel"]
