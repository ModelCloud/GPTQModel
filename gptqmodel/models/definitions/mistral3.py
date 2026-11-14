# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForPreTraining

from ..base import BaseQModel
from ..fusable import FusableConfig
from .llama import LlamaQModel


class Mistral3GPTQ(BaseQModel):
    fusable = FusableConfig("mistral3").merge(LlamaQModel.fusable, source_model_type="mistral")

    loader = AutoModelForPreTraining

    # The decoder is the verified Mistral/Llama layout, so q/k/v share the post-norm
    # hidden state and gate/up share the MLP input. The tags stay inert until the
    # `mistral3` model type itself is listed here: that needs a real (tiny) forward
    # of the vision wrapper, which `AutoModelForCausalLM.from_config` cannot build.
    pre_lm_head_norm_module = ["model.language_model.norm", "model.norm"]

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        }
    ]
