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
            # `o_proj` reads `attn * sigmoid(gate_proj(x))`; the gate makes
            # gate_proj's output non-homogeneous, so it is not o_proj's source.
            "self_attn": (
                "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "gate_proj:0:nosrc",
                "q_norm:!", "k_norm:!",
            ),
            # Both post-attention and post-MLP norms consume a branch output
            # before the residual add, so neither feeds the next projection.
            "post_attention_layernorm": ("post_attention_layernorm:!:postnorm",),
            "pre_mlp_layernorm": ("pre_mlp_layernorm:!",),
            "mlp:moe:?": {
                # The router gate (`mlp.router.gate`) stays in the checkpoint
                # dtype; it is not a quantizable projection.
                "router": ("gate:!",),
                "shared_experts:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
            # The first num_dense_layers use gate_proj/up_proj/down_proj;
            # later layers route through mlp.experts. Missing on MoE layers,
            # which is fine because layer_modules_strict=False.
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "post_mlp_layernorm": ("post_mlp_layernorm:!:postnorm",),
        }
    ]
