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


class FalconQModel(RwgQModel):
    """Falcon checkpoints, which name their decoder norms ``input_layernorm``.

    In the parallel-attention variants (the default Falcon configuration) the
    attention and MLP branches read the same ``input_layernorm`` output, so the
    MLP input projection carries the shared ``in=x`` tag. The MLP input
    projection stays a non-source for the GELU-separated output projection.
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


__all__ = ["RwgQModel", "FalconQModel"]
