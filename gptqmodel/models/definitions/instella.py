# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from . import LlamaQModel


class InstellaQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.instella.InstellaConfig",
        model_class="gptqmodel.models.ports.instella.InstellaForCausalLM",
    )

    # Instella names its decoder norms `pre_attention_layernorm` and
    # `pre_feedforward_layernorm` (modeling_instella.py:663-664), not the Llama
    # `input_layernorm` / `post_attention_layernorm`. Inheriting the Llama tree
    # left q/k/v without a scaling source and let AWQ fold `mlp.gate/up` into
    # `self_attn.o_proj`, across the residual add (PPL 14616 vs RTN 27.71).
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "pre_attention_layernorm": ("pre_attention_layernorm:!:norm",),
            "self_attn": (
                "q_norm:!:norm", "k_norm:!:norm",
                "q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1:o",
            ),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Full-width Q/K RMSNorm and per-layer rotary/cache replay have not
        # qualified scale search/folding. Keep every linear in plain AWQ.
        return []
