# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ...utils.ricursive_patch import apply_ricursive_generate_patch
from ..base import BaseQModel


class RicursiveQModel(BaseQModel):
    """Ricursive Ripple: thin-and-deep Gemma-3N-like hybrid-attention model.

    Ripple ships custom code (``RicursiveForCausalLM``), so the checkpoint must
    be loaded with ``trust_remote_code=True``. The module layout is Llama-like
    with three additions:

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

    require_trust_remote_code = True

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            # `value_res_lambda` is a direct parameter on the attention
            # container when value residuals are enabled.
            "self_attn:!": (
                "q_norm:!:norm",
                "k_norm:!:norm",
                # `q_proj`, `k_proj` and `g_proj` all read the raw hidden states. `v_proj`
                # deliberately carries no `in=` tag: with `use_valres` it reads
                # `lerp(value_residual, hidden_states, sigmoid(value_res_lambda))`, which is
                # a different tensor, so grouping it with q/k would corrupt its Hessian.
                "q_proj:0:q:in=x",
                "k_proj:0:k:in=x",
                "v_proj:0:v",
                "g_proj:0:in=x",
                "o_proj:1:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!:norm",),
        },
    ]

    def after_model_load(self, model, load_quantized_model=False):
        # Ripple's shipped `prepare_inputs_for_generation` injects a spurious axis under
        # transformers >= 5 and breaks `generate()` on the very first prefill.
        apply_ricursive_generate_patch(model)
        return model


__all__ = ["RicursiveQModel"]
