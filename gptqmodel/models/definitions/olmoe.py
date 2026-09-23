# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class OlmoeQModel(BaseQModel):
    """Quantization definition for the Transformers OLMoE decoder.

    Transformers stores OLMoE experts as fused 3-D parameters. Defuser expands
    them into numbered expert modules before GPTQModel builds this tree, so the
    quantizable leaves are the standard gate/up/down projections below.
    """

    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "q_norm:!",
                "k_norm:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]


__all__ = ["OlmoeQModel"]
