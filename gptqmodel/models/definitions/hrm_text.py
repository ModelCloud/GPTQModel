# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from ..looped_capture import LoopedLayerCaptureMixin


class Chunk:
    def __init__(self, dim: int = 0):
        self.dim = dim


class HrmTextQModel(LoopedLayerCaptureMixin, BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    # z_L_init is a Parameter on model itself, so child-module traversal misses it.
    fusable = FusableConfig("hrm_text").merge(LlamaFusableConfig())

    modules_with_direct_meta_tensors = ["model"]

    # The H stack runs last, so its final norm owns the hidden state that
    # `lm_head` consumes.
    pre_lm_head_norm_module = "model.H_module.final_norm"

    _child_module_tree = {
        "input_layernorm": ("input_layernorm:!",),
        # o_proj reads `attn * sigmoid(gate_proj(x))`; the gate makes gate_proj's
        # output non-homogeneous.
        "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "gate_proj:0:nosrc", "o_proj:1"),
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

