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
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Native parallel/sequential and dual-norm layouts have distinct replay
        # boundaries. Until those folds are qualified, quantize every linear
        # plainly through AWQ's existing empty-group path (also fallback=None).
        return []


class FalconQModel(RwgQModel):
    """Falcon checkpoints, which name their decoder norms ``input_layernorm``.

    In the parallel-attention variants (the default Falcon configuration) the
    attention and MLP branches read the same ``input_layernorm`` output, so the
    MLP input projection carries the shared ``in=x`` tag. The MLP input
    projection stays a non-source for the GELU-separated output projection.
    """

    # Canonical metadata keeps the exact native config. RoPE keeps the exact
    # native model; ALiBi uses only the narrowly corrected native attention.
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.rw.FalconConfig",
        model_class="gptqmodel.models.ports.rw.RWModelLoader",
    )


__all__ = ["RwgQModel", "FalconQModel"]
