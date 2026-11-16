# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class BailingMoeQModel(BaseQModel):
    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"
    layer_modules_strict = False

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # bailing_hybrid mixes delta/linear-attention, full-attention MoE and
            # MLA layers, but every variant keeps its attention projections under
            # `self_attn`. The remote-code `bailing_moe` layout (Ling 2.0) still
            # exposes `attention.query_key_value`/`attention.dense`, so both
            # branches are declared. `layer_modules_strict=False` lets each
            # checkpoint resolve only the branch it owns.
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                # `o_proj` consumes the gated attention output built from
                # `kv_b_proj` (MLA) or q/k/v (linear attention), so it must be
                # a later subset.
                "o_proj:2",
                "g_proj:0",
                "kv_a_proj_with_mqa:0",
                "kv_b_proj:1",
                "forget_gate.f_proj:0",
                "b_proj:0",
                # Linear-attention short convolution (checkpoint-owned, not a
                # quantizable projection).
                "conv1d:!",
            ),
            "attention": (
                "query_key_value:0",
                "dense:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",), # <-- 0.5MB per layer. Not worth quantizing
                # `down_proj` consumes the gate/up product, so it must be
                # calibrated (and AWQ-scaled) after gate/up are quantized.
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
            # delta/linear-attention layers have a dense MLP with no experts.
            # These entries do not resolve on MoE layers, which is fine because
            # layer_modules_strict=False.
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
