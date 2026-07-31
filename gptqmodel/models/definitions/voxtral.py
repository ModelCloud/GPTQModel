# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForPreTraining

from ..base import BaseQModel


class VoxtralGPTQ(BaseQModel):
    loader = AutoModelForPreTraining

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        }
    ]
