# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForImageTextToText

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


def _register_defuser_model_type():
    try:
        from defuser.model_registry import MODEL_CONFIG
        from defuser.utils.common import MIN_SUPPORTED_TRANSFORMERS_VERSION
    except Exception:
        return

    MODEL_CONFIG.setdefault(
        "minimax_m3_vl",
        {"min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION},
    )


_register_defuser_model_type()


class MiniMaxM3VLGPTQ(BaseQModel):
    loader = AutoModelForImageTextToText
    require_load_processor = False
    support_batch_quantize = False

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    # MiniMax-M3 starts with dense MLP layers, then switches to sparse MoE.
    layer_modules_strict = False
    dynamic_expert_index = "num_local_experts"

    # Defuser splits MiniMax-M3 packed expert tensors into gate/up/down modules.
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "q_norm:0:!",
                "k_proj:0",
                "k_norm:0:!",
                "v_proj:0",
                "indexer.q_proj:0",
                "indexer.q_norm:0:!",
                "indexer.k_proj:0",
                "indexer.k_norm:0:!",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Dense fallback used by early decoder blocks.
                "": ("gate_up_proj:0", "down_proj:1"),
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "shared_experts": ("gate_up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]


__all__ = ["MiniMaxM3VLGPTQ"]
