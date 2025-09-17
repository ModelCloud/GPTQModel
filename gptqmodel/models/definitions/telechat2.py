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

    _layers_modules_tree = [
        "transformer",
        "h",
        "#",
        {
            "self_attention": {"dense": ("dense",)},
            "mlp": ("up_proj:0", "gate_proj:0", "down_proj:1"),
        }
    ]

__all__ = ["TeleChat2QModel"]
