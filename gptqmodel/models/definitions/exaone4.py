from transformers import AutoModelForImageTextToText

from ..base import BaseQModel
from .base_qwen3_vl import BaseQwen3VLGPTQ


class Exaone4QModel(BaseQModel):

    pre_lm_head_norm_module = "model.norm"

    # Exaone4 uses GQA (Grouped Query Attention) architecture
    # o_proj must match v_proj shape for AWQ scaling optimizations
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            # Skip q_norm and k_norm (too small, RMSNorm layers)
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "q_norm:!",
                "k_norm:!"
            ),
            # Skip post_attention_layernorm (RMSNorm, too small)
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            # Skip post_feedforward_layernorm (RMSNorm, too small)
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        }
    ]


class Exaone4_5QModel(BaseQwen3VLGPTQ):
    """Quantization layout for the EXAONE 4.5 vision-language model."""

    loader = AutoModelForImageTextToText
    require_load_processor = True
    layer_modules_strict = False

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"
    out_of_model_tensors = {"prefixes": ["mtp"]}

    # GQA makes o_proj shape-incompatible with the Q/K/V AWQ scale group.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # EXAONE 4.5 wraps the EXAONE 4 decoder in a multimodal model and keeps
    # the visual tower outside the language-layer quantization tree.
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "q_norm:!",
                "k_norm:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]


__all__ = ["Exaone4QModel", "Exaone4_5QModel"]
