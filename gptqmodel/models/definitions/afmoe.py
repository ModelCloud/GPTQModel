# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class AfMoeQModel(BaseQModel):
    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    fusable = FusableConfig("afmoe").merge(LlamaFusableConfig(), profiles=("qkv",))

    dynamic_expert_index = "num_experts"

    require_trust_remote_code = True
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
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "gate_proj:0"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                # The router gate (`mlp.router.gate`) stays in the checkpoint
                # dtype; it is not a quantizable projection.
                "router": ("gate:!",),
                "shared_experts:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
            # Layer 0 has a dense MLP (gate_proj/up_proj/down_proj); the
            # remaining layers route through mlp.experts. Missing on MoE layers,
            # which is fine because layer_modules_strict=False.
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
