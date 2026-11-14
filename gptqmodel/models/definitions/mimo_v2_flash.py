# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Native Transformers ``MiMoV2FlashForCausalLM`` (MiMo-V2-Flash)."""

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class MimoV2FlashQModel(BaseQModel):
    """Quantize the native text-only MiMo-V2-Flash decoder.

    `mimo_v2` covers the repository-defined MiMo checkpoints. Hub
    MiMo-V2-Flash configs report `model_type == "mimo_v2_flash"`, which had no
    definition and fell back to `BaseQModel` ("no explicit module_tree").
    """

    # Transformers 5.22 fuses the checkpoint's per-expert projections into 3-D
    # parameters; Defuser has no registry entry for this model type, so the
    # loader runs its generic experts pass explicitly.
    defuser_generic_experts = True

    dynamic_expert_index = "n_routed_experts"
    layer_modules_strict = False  # dense layers use a plain MLP

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]


__all__ = ["MimoV2FlashQModel"]
