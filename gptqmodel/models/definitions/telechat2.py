import torch

from ..base import BaseQModel


class TeleChat2QModel(BaseQModel):
    # telechat2 requires custom model code
    require_trust_remote_code = True
    # telechat2 requires float16
    require_dtype = torch.float16

    layers_node = ["transformer.h"]
    base_modules = ["transformer.word_embeddings", "transformer.ln_f"]
    pre_lm_head_norm_module = "transformer.ln_f"

    # TODO: full deprecation by gptqmodel v4.3
    # legacy definition (deprecated): migrate to layers_modules_tree
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

    _layers_modules_tree = [
        "transformer",
        "h",
        "#",
        {
            "self_attention": ("dense"),
            "mlp": ("up_proj:0", "gate_proj:0", "down_proj:1"),
        }
    ]

__all__ = ["TeleChat2QModel"]
