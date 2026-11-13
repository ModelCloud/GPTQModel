# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""OLMo Hybrid decoder layout.

OLMo Hybrid alternates three gated-DeltaNet ``linear_attn`` layers with one
full ``self_attn`` layer. Both block shapes carry checkpoint-owned,
non-quantized norms and the linear branch also owns a depthwise ``conv1d`` and
a gated output norm, so the generic Llama tree cannot resolve them through
``ModulePlan.find()``.
"""

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class OlmoHybridQModel(BaseQModel):
    fusable = FusableConfig("olmo_hybrid").merge(LlamaFusableConfig())

    pre_lm_head_norm_module = "model.norm"

    # Hybrid layers do not share a single block shape: the linear-attention
    # layers omit ``post_feedforward_layernorm`` and the full-attention layers
    # omit ``input_layernorm``. Missing declared subpaths are skipped per layer.
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0:q",
                "q_norm:0:!",
                "k_proj:0:k",
                "k_norm:0:!",
                "v_proj:0:v",
                "o_proj:1:o",
            ),
            # ``A_log`` and ``dt_bias`` are parameters of the DeltaNet module
            # itself, so the container must be an explicit dense owner too.
            "linear_attn:!": (
                "q_proj:0:q",
                "k_proj:0:k",
                "v_proj:0:v",
                "a_proj:0:!",
                "b_proj:0:!",
                "g_proj:0",
                "o_proj:1:o",
                "conv1d:0:!",
                "o_norm:0:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]


__all__ = ["OlmoHybridQModel"]
