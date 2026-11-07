# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForPreTraining

from ..base import BaseQModel
from ..fusable import FusableConfig
from .llama import LlamaQModel


class Mistral3GPTQ(BaseQModel):
    fusable = FusableConfig("mistral3").merge(LlamaQModel.fusable, source_model_type="mistral")

    loader = AutoModelForPreTraining

    pre_lm_head_norm_module = ["model.language_model.norm", "model.norm"]

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
