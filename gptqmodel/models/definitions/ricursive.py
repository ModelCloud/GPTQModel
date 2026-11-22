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
            # be a fold source (the effective tree lifts this without valres).
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
                # is quantized without an input-norm fold. `v_proj -> o_proj`
                # stays a valid fold: the sigmoid gate is per channel.
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

    # Quantized projections whose output never reaches another quantized
    # projection's input unchanged: q/k feed the per-head norms and RoPE, g_proj
    # feeds the sigmoid gate. The planner may otherwise pick the last of them as
    # the "previous op" of the next stage (e.g. `k_proj -> v_proj` when g_proj
    # is absent), which is not a data-flow edge.
    _AWQ_NON_PRODUCERS = frozenset({"self_attn.q_proj", "self_attn.k_proj", "self_attn.g_proj"})

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        config = getattr(model, "config", None)
        gated = getattr(config, "use_gated_attention", True)
        valres = getattr(config, "use_valres", True)
        if gated and valres:
            return tree
        tree = deepcopy(tree)
        layer = tree[-1]
        attention = layer["self_attn:!"]
        if not gated:
            attention = tuple(spec for spec in attention if not spec.startswith("g_proj:"))
        if not valres:
            # Without the value residual `v_proj` reads the input norm output
            # exactly like q/k/g, so the norm is a valid fold source for all of
            # them and `v_proj` joins their shared-input stage.
            layer["input_layernorm"] = ("input_layernorm:!:norm",)
            attention = tuple(
                "v_proj:0:v:in=x" if spec.startswith("v_proj:") else spec for spec in attention
            )
        layer["self_attn:!"] = attention
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # The generic planner emits input_layernorm -> q/k/v(/g) (only without
        # the value residual, see `_resolve_effective_module_tree`),
        # v_proj -> o_proj (the sigmoid gate multiplies per channel, so it
        # commutes with a channel scale), pre_feedforward_layernorm -> gate/up
        # and up_proj -> down_proj. Drop the groups that are not neutral:
        # * a source among q/k/g is a stage-ordering artefact, not a producer;
        # * `ffn_clamp_threshold` clamps `up_proj`'s output before the product,
        #   so a channel scale on `up_proj` does not commute with the clamp.
        clamp = getattr(getattr(self.model, "config", None), "ffn_clamp_threshold", None)
        kept = []
        for node in super().awq_get_modules_for_scaling(module, input_feat, module_kwargs):
            prev_op = node.get("prev_op")
            source = next((name for name, child in module.named_modules() if child is prev_op), None)
            if source in self._AWQ_NON_PRODUCERS:
                continue
            if clamp and source == "mlp.up_proj":
                continue
            kept.append(node)
        return kept


__all__ = ["RicursiveQModel"]
