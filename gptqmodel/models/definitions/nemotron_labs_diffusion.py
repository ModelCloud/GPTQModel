# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoModel

from ..base import BaseQModel


class NemotronLabsDiffusionQModel(BaseQModel):
    require_trust_remote_code = True
    loader = AutoModel

    lm_head = "diffusion_head"
    pre_lm_head_norm_module = "encoder.norm"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # Nemotron Labs Diffusion uses a custom AutoModel with an internal
    # Ministral-style decoder stack under encoder.layers.
    module_tree = [
        "encoder",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
