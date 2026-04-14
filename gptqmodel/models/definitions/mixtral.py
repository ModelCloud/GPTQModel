# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class MixtralQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_local_experts"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # The first alias in each token is the runtime shell name. Later aliases are
    # checkpoint-side names that LazyTurtle may resolve directly from module_tree.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp|block_sparse_moe:moe:?": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj|w1:0", "up_proj|w3:0", "down_proj|w2:1"),
                }
            }
        }
    ]
