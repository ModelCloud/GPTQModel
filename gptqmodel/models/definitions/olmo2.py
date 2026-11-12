# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""OLMo 2 decoder layout.

OLMo 2 inherits the Llama block layout but adds per-head Q/K RMSNorm inside
attention and a second RMSNorm after the MLP. Both are checkpoint-owned and
non-quantized, so they must appear as excluded leaves in the module tree; the
generic Llama tree cannot resolve them through ``ModulePlan.find()``.
"""

from ..fusable import FusableConfig, LlamaFusableConfig
from . import LlamaQModel


class Olmo2QModel(LlamaQModel):
    fusable = FusableConfig("olmo2").merge(LlamaFusableConfig())

    shared_input_verified_model_types = frozenset({"olmo2"})

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            # OLMo 2 normalizes only after attention and after the MLP; there is
            # no pre-attention `input_layernorm` on the decoder layer.
            "self_attn": (
                "q_proj:0:q:in=x",
                "q_norm:0:!",
                "k_proj:0:k:in=x",
                "k_norm:0:!",
                "v_proj:0:v:in=x",
                "o_proj:1:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]


__all__ = ["Olmo2QModel"]
