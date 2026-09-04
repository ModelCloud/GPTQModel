# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class AXK2QModel(BaseQModel):
    """A.X-K2 (SKT) sparse MoE/MLA model support.

    Architecture highlights:
    - DeepSeek-V3-style MLA attention with a fused q_gate_proj that emits both
      query states and the post-attention sigmoid gate.
    - Per-layer input_layernorm is an AXK2GatedRMSNorm; its small MLP gate is not
      quantized.
    - Dense MLP fallback on layer 0, sparse AXK2MoE on all other layers.
    - Shared expert and routed experts both use gate_proj/up_proj/down_proj.
    """

    require_trust_remote_code = False
    require_fast_init = False

    # allow dynamic expert index for layer_modules so we don't need to write out 256 layers
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # Layer 0 is a dense MLP while the rest are MoE; don't fail if some expected
    # MoE modules are missing on a given layer.
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_proj:0:q",
                "kv_a_proj_with_mqa:0:k:v",
                # The fused Q/gate projection consumes the compressed Q latent pair.
                "q_gate_proj:1:q:input",
                "kv_b_proj:1:k:v:input",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "experts:routed": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
                "shared_experts:shared": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
            },
        },
    ]


__all__ = ["AXK2QModel"]
