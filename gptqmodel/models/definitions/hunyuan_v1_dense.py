# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..fusable import LlamaFusableConfig
from .llama import LlamaQModel


class HunYuanDenseV1QModel(LlamaQModel):
    """
    Hunyuan Dense V1 follows a Llama-style decoder layout with per-head Q/K
    RMSNorm modules inside attention. Those norms are metadata/base modules for
    quantization and should not be replaced by quantized linear kernels.
    """
    fusable = LlamaFusableConfig().derive("hunyuan_v1_dense", parent_types={
        "qkv": (
            "transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense.HunYuanDenseV1Attention",
        ),
        "gate_up": (
            "transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense.HunYuanDenseV1MLP",
        ),
    })


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
