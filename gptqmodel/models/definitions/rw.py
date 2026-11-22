# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class RwgQModel(BaseQModel):
    """Historical RW aliases with explicit config translation to native Falcon."""

    require_trust_remote_code = False

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.rw.RWConfig",
        model_class="gptqmodel.models.ports.rw.RWModelLoader",
    )

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "ln_attn": ("ln_attn:!",),
            "ln_mlp": ("ln_mlp:!",),
            "self_attention": ("query_key_value:0:in=x", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # GELU between the two MLP projections is not positively homogeneous.
            "mlp": ("dense_h_to_4h:0:in=x:nosrc", "dense_4h_to_h:1"),
        },
    ]

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector)
        config = getattr(model, "config", None)
        if config is not None:
            grouped = config.new_decoder_architecture
            shared = config.parallel_attn and (not grouped or config.num_ln_in_parallel_attn == 1)
            if not shared:
                leaves = tree[-1]
                for name in ("self_attention", "mlp"):
                    leaves[name] = tuple(leaf.replace(":in=x", "") for leaf in leaves[name])
                if grouped and config.num_ln_in_parallel_attn in (None, 2) and "input_layernorm" in leaves:
                    # Dual-norm layers: `ln_mlp` feeds only the MLP. Declaring it
                    # after attention keeps AWQ from folding the attention input
                    # scale into it (`ln_attn` is the q/k/v producer).
                    # Falcon's dedicated two-norm tree already has that order.
                    tree[-1] = {
                        name: leaves[name]
                        for name in ("input_layernorm", "ln_attn", "self_attention", "ln_mlp",
                                     "post_attention_layernorm", "mlp")
                    }
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        attention = getattr(module, "self_attention", None)
        if not getattr(getattr(attention, "config", None), "alibi", False):
            return nodes
        # ALiBi (falcon-rw) bakes the bias into the attention mask, which the
        # packed token-row replay of the q/k/v scale search cannot rebuild.
        # Keep the MLP folds and quantize query_key_value without a fold.
        qkv = getattr(attention, "query_key_value", None)
        return [node for node in nodes if not any(layer is qkv for layer in node.get("layers") or [])]


# Falcon's two-norm parallel decoder (`new_decoder_architecture=True`, or
# `num_ln_in_parallel_attn=2`) normalizes each branch on its own: `ln_attn`
# feeds `self_attention`, `ln_mlp` feeds `mlp`. There is no shared
# `input_layernorm` and no sequential `post_attention_layernorm`, so the
# branches must not be folded into each other (#5414).
_FALCON_TWO_NORM_TREE = [
    "transformer",
    "h",
    "#",
    {
        "ln_attn": ("ln_attn:!",),
        "self_attention": ("query_key_value:0", "dense:1"),
        "ln_mlp": ("ln_mlp:!",),
        "mlp": ("dense_h_to_4h:0:nosrc", "dense_4h_to_h:1"),
    },
]


class FalconQModel(RwgQModel):
    """Falcon checkpoints, which name their decoder norms ``input_layernorm``.

    In the parallel-attention variants (the default Falcon configuration) the
    attention and MLP branches read the same ``input_layernorm`` output, so the
    MLP input projection carries the shared ``in=x`` tag. The MLP input
    projection stays a non-source for the GELU-separated output projection.

    The two-norm parallel decoder is a different layout (``ln_attn``/``ln_mlp``);
    ``_model_module_tree`` selects it from the live config.
    """

    # Canonical metadata keeps the exact native config. RoPE keeps the exact
    # native model; ALiBi uses only the narrowly corrected native attention.
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.rw.FalconConfig",
        model_class="gptqmodel.models.ports.rw.RWModelLoader",
    )

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attention": ("query_key_value:0:in=x", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("dense_h_to_4h:0:in=x:nosrc", "dense_4h_to_h:1"),
        },
    ]
    module_tree_two_norm = _FALCON_TWO_NORM_TREE

    @classmethod
    def _model_module_tree(cls, model, quantize_config, detector=None):
        """Mirror transformers' FalconDecoderLayer norm layout selection."""

        config = getattr(model, "config", None)
        parallel = bool(getattr(config, "parallel_attn", False))
        num_ln = getattr(config, "num_ln_in_parallel_attn", None)
        if parallel and num_ln is None and bool(
            getattr(config, "new_decoder_architecture", False)
        ):
            # `FalconDecoderLayer.__init__` upgrades this to 2 in place.
            num_ln = 2
        if parallel and num_ln == 2:
            return cls.module_tree_two_norm
        return super()._model_module_tree(model, quantize_config, detector=detector)



__all__ = ["RwgQModel", "FalconQModel"]
