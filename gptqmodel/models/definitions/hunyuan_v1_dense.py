# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from .llama import LlamaQModel


class HunYuanDenseV1QModel(LlamaQModel):
    """
    Hunyuan Dense V1 follows a Llama-style decoder layout with per-head Q/K
    RMSNorm modules inside attention. Those norms are metadata/base modules for
    quantization and should not be replaced by quantized linear kernels.
    """

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "query_layernorm:!",
                "q_proj:0",
                "key_layernorm:!",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
