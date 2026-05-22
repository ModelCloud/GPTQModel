# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


from ..base import BaseQModel


class Chunk:
    def __init__(self, dim: int = 0):
        self.dim = dim


class HrmTextQModel(BaseQModel):
    # z_L_init is a Parameter on model itself, so child-module traversal misses it.
    modules_with_direct_meta_tensors = ["model"]

    _child_module_tree = {
        "input_layernorm": ("input_layernorm:!",),
        "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "gate_proj:0", "o_proj:1"),
        "post_attention_layernorm": ("post_attention_layernorm:!",),
        "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
    }

    module_tree = [
        [
            "model",
            "L_module",
            "layers",
            "#",
            _child_module_tree,
        ],
        [
            "model",
            "H_module",
            "layers",
            "#",
            _child_module_tree,
        ],
    ]
