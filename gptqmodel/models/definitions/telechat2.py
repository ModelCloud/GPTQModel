import torch

from ..base import BaseGPTQModel


class TeleChat2GPTQ(BaseGPTQModel):
    # telechat2 requires custom model code
    require_trust_remote_code = True
    # telechat2 requires float16
    require_dtype = torch.float16

    layer_type = "TelechatBlock"
    layers_node = "transformer.h"
    base_modules = ["transformer.word_embeddings", "transformer.ln_f"]
    pre_lm_head_norm_module = "transformer.ln_f"

    """
    If other frameworks are used for inference (such as VLLM),
    it is best not to quantify QKV due to the organization of
    key value weights in the Telechat model
    """
    layer_modules = [
        ["self_attention.dense"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

__all__ = ["TeleChat2GPTQ"]
