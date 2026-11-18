from transformers import AutoModelForImageTextToText

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from .base_qwen3_vl import BaseQwen3VLGPTQ


class Exaone4QModel(BaseQModel):

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.exaone4.Exaone4TextConfig",
        model_class="gptqmodel.models.ports.exaone4.Exaone4ForCausalLM",
    )

    fusable = FusableConfig("exaone4").merge(LlamaFusableConfig())

    # q/k/v share the post-norm hidden state and gate/up share the MLP input.
    # Verified against a real (tiny, CPU) forward in
    # tests/module_tree/test_shared_input_cpu_forward.py.
    shared_input_verified_model_types = frozenset({"exaone4"})

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
                "q_proj:0:in=x",
                "q_norm:0:!",
                "k_proj:0:in=x",
                "k_norm:0:!",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            # Skip post_attention_layernorm (RMSNorm, too small). Exaone4 adds
            # the normalized attention output to the residual, so the norm does
            # not feed the MLP directly and cannot be an AWQ scale source.
            "post_attention_layernorm": ("post_attention_layernorm:!:postnorm",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
            # Skip post_feedforward_layernorm (RMSNorm, too small)
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        }
    ]


class Exaone4_5QModel(BaseQwen3VLGPTQ):
    """Quantization layout for the EXAONE 4.5 vision-language model."""
    fusable = FusableConfig("exaone4_5").merge(Exaone4QModel.fusable)


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
                "q_proj:0:in=x",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
                "q_norm:!",
                "k_norm:!",
            ),
            # Exaone 4.5 adds the normalized attention output to the residual,
            # so this norm does not feed the MLP (same contract as Exaone4).
            "post_attention_layernorm": ("post_attention_layernorm:!:postnorm",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]


__all__ = ["Exaone4QModel", "Exaone4_5QModel"]
