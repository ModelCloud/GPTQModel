# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from ..base import BaseQModel
from ..local import LocalModelLoader


class RicursiveQModel(BaseQModel):
    """Ricursive Ripple: thin-and-deep Gemma-3N-like hybrid-attention model.

    The package owns Ripple's config, modeling and generation preparation.
    Checkpoint Python and trust flags never select the implementation.
    The module layout is Llama-like with four additions:

    * **PeriNorm** adds ``pre_feedforward_layernorm`` and
      ``post_feedforward_layernorm`` around the MLP, on top of the usual
      input/post-attention norms.
    * **Gated attention** adds ``g_proj``, a colwise projection of the same
      hidden states that feeds the sigmoid gate on the attention output.
    * **Per-head Q/K RMSNorm** adds ``q_norm`` / ``k_norm`` inside attention.
    * **Value residual** (``use_valres``) makes ``v_proj`` read a blend of the
      first layer's hidden states and the current ones, so ``v_proj`` is the one
      attention projection that does *not* share the q/k/g input tensor.

    Every fifth layer is ``full_attention`` (NoPE); the rest are
    ``sliding_attention``. That switch is runtime behaviour, so it does not
    change which weights are quantized.
    """

    loader = LocalModelLoader(
        "gptqmodel.models.ports.ricursive.configuration_ricursive.RicursiveConfig",
        "gptqmodel.models.ports.ricursive.modeling_ricursive.RicursiveForCausalLM",
        tokenizer_class="gptqmodel.models.ports.ricursive.tokenization_ricursive.RicursiveTokenizer",
    )
    fusable = None  # The executor is unqualified for value residuals and MLP chunks.

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            # With `use_valres` `v_proj` reads
            # `lerp(value_residual, input_layernorm(h), ...)`: folding a scale
            # into this norm changes only the lerp's normalized half, which no
            # single v_proj weight scale can undo. The norm therefore may not
            # be a fold source; attention falls back to plain quantization.
            "input_layernorm": ("input_layernorm:!:norm:nosrc",),
            # `value_res_lambda` is a direct parameter on the attention
            # container when value residuals are enabled.
            "self_attn:!": (
                "q_norm:!:norm",
                "k_norm:!:norm",
                # `q_proj`, `k_proj` and `g_proj` all read the input norm.
                # `g_proj` is a sigmoid gate applied to the attention output,
                # not `o_proj`'s input, so it may not be a fold source.
                # With `use_valres` `v_proj` reads
                # `lerp(value_residual, normed, sigmoid(value_res_lambda))`,
                # which is not the norm output; keep it in its own stage so it
                # is quantized plainly (the fold into `g_proj` is then dropped).
                # AWQ currently selects no folding groups for either variant.
                "q_proj:0:q:in=x",
                "k_proj:0:k:in=x",
                "g_proj:0:in=x:nosrc",
                "v_proj:1:v",
                "o_proj:2:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!:norm",),
        },
    ]

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        if not getattr(getattr(model, "config", None), "use_gated_attention", True):
            tree = deepcopy(tree)
            attention = tree[-1]["self_attn:!"]
            tree[-1]["self_attn:!"] = tuple(spec for spec in attention if not spec.startswith("g_proj:"))
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Value blending, the sigmoid output gate, and clipped MLP activations
        # cross scale boundaries. No fold/search/replay group is qualified.
        # The normal AWQ path still quantizes every declared decoder linear.
        return []


__all__ = ["RicursiveQModel"]
