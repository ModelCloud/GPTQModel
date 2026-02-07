from ..base import BaseQModel


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