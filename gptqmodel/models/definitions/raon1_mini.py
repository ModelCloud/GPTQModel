# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Raon1MiniQModel(BaseQModel):
    """KRAFTON Raon-1 mini hybrid RDA/MLA model with routed MoE layers."""

    require_trust_remote_code = True
    dynamic_expert_index = "num_experts"
    pre_lm_head_norm_module = "model.norm"
    layer_modules_strict = False
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # RDA and MLA layers have different projection names. Raon-mHC control
    # modules are intentionally left in full precision.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": (
                "q_proj:0", "k_proj:0", "v_proj:0", "f_a_proj:0", "f_b_proj:1",
                "b_proj:0", "rda_w_proj:0", "g_proj:0", "q_a_proj:0",
                "q_b_proj:1", "kv_a_proj_with_mqa:0", "kv_b_proj:1", "o_proj:2",
            ),
            "mlp": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
            "moe:moe": {
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        # Raon decoder layers take `cache`, not the Transformers `use_cache` flag.
        additional_inputs.pop("use_cache", None)
        additional_inputs.setdefault("cache", None)
        return additional_inputs


__all__ = ["Raon1MiniQModel"]
