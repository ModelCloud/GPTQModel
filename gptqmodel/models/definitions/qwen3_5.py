# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.qwen3_5 import Qwen3_5TextConfig

from . import LlamaQModel


class Qwen3_5QModel(LlamaQModel):
    """
    Qwen3_5 inherits the Llama-style layout but inserts Q/K RMS norm layers
    ahead of the attention projections. We mark those helper modules as
    non-quantized so the layer walker captures the complete structure.
    """

    config_class = Qwen3_5TextConfig

    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "q_proj:0", "k_norm:!", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn": (
                "norm:!",
                "in_proj_qkv:0",
                "in_proj_z:1",
                "in_proj_b:1",
                "in_proj_a:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
