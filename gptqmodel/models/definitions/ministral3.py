# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..fusable import LlamaFusableConfig


class Ministral3GPTQ(BaseQModel):
    fusable = LlamaFusableConfig().derive("ministral3", parent_types={
        "qkv": (
            "transformers.models.ministral3.modeling_ministral3.Ministral3Attention",
        ),
        "gate_up": (
            "transformers.models.ministral3.modeling_ministral3.Ministral3MLP",
        ),
    })

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
        },
    ]


__all__ = ["Ministral3GPTQ"]
