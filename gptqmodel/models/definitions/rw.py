# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class RwgQModel(BaseQModel):
    """RefinedWeb checkpoints, which name their decoder norms ``ln_1``/``ln_2``."""

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!",),
            "self_attention": ("query_key_value:0", "dense:1"),
            "ln_2": ("ln_2:!",),
            # GELU between the two MLP projections is not positively homogeneous.
            "mlp": ("dense_h_to_4h:0:nosrc", "dense_4h_to_h:1"),
        },
    ]


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
