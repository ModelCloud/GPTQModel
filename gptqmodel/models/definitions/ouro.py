# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..looped_capture import LoopedLayerCaptureMixin


class OuroQModel(LoopedLayerCaptureMixin, BaseQModel):
    """GPTQ definition for ByteDance's looped Ouro decoder."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.ouro.configuration_ouro.OuroConfig",
        model_class="gptqmodel.models.ports.ouro.modeling_ouro.OuroForCausalLM",
    )
    require_trust_remote_code = False
    fusable = None

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    # Ouro's sandwich RMSNorms are part of the forward path, but only the
    # linear projections are quantized. The recurrent loop reuses these
    # decoder layers, so the tree describes one shared layer topology.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0:in=x",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "input_layernorm_2": ("input_layernorm_2:!",),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": (
                "gate_proj:0:in=x",
                "up_proj:0:in=x",
                "down_proj:1",
            ),
            "post_attention_layernorm_2": ("post_attention_layernorm_2:!",),
        },
    ]


__all__ = ["OuroQModel"]
