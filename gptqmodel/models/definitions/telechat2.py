import torch

from ..base import BaseQModel


class TeleChat2QModel(BaseQModel):
    # telechat2 requires custom model code
    require_trust_remote_code = True
    # telechat2 requires float16
    require_dtype = torch.float16

    pre_lm_head_norm_module = "transformer.ln_f"

    _layers_modules_tree = [
        "transformer",
        "h",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attention": {"dense": ("dense",)},
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

__all__ = ["TeleChat2QModel"]
