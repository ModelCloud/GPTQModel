# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class LongCatFlashQModel(BaseQModel):
    fusable = FusableConfig("longcat_flash").merge(LlamaFusableConfig(), profiles=("gate_up",))

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("0:!:nosrc", "1:!:nosrc"),
            "self_attn": {
                "0": ("q_a_layernorm:!:nosrc", "kv_a_layernorm:!:nosrc", "q_a_proj:0:nosrc", "q_b_proj:0:in=q_a:nosrc", "kv_a_proj_with_mqa:0:nosrc", "kv_b_proj:0:in=kv_a:nosrc", "o_proj:1:nosrc"),
                "1": ("q_a_layernorm:!:nosrc", "kv_a_layernorm:!:nosrc", "q_a_proj:0:nosrc", "q_b_proj:0:in=q_a:nosrc", "kv_a_proj_with_mqa:0:nosrc", "kv_b_proj:0:in=kv_a:nosrc", "o_proj:1:nosrc")
            },
            "post_attention_layernorm": ("0:!:nosrc", "1:!:nosrc"),
            "mlps": {
                "0": ("gate_proj:0", "up_proj:0", "down_proj:1:nosrc"),
                "1": ("gate_proj:0", "up_proj:0", "down_proj:1:nosrc")
            },
            "mlp:moe": {
                "router:!": ("classifier:!:nosrc",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1")
                }
            }
        }
    ]
