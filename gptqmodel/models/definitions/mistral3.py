# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForPreTraining

from ..base import BaseQModel


class Mistral3GPTQ(BaseQModel):
    loader = AutoModelForPreTraining

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
