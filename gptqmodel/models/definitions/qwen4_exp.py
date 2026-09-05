# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForImageTextToText

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Qwen4ExpQModel(BaseQModel):
    """Qwen3.8-Flash-Next / Qwen4 experimental multimodal MoE."""

    loader = AutoModelForImageTextToText
    require_load_processor = True
    layer_modules_strict = False

    dynamic_expert_index = "num_experts"

    # The final mixer replaces the usual decoder RMSNorm.
    pre_lm_head_norm_module = "model.language_model.hyper_connection_mixer"
    rotary_embedding = "model.language_model.rotary_emb"

    # Transformers intentionally ignores the auxiliary MTP decoder on load.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    # The outer config reuses the text model's PLE checkpoint mapping.
    hf_conversion_model_type_alias = "qwen4_exp_text"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # GQA makes o_proj shape-incompatible with the Q/K/V AWQ scale group.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # Only unmarked entries are quantized; PLE and hyper weights are omitted.
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "self_attn": (
                "indexer.index_qk_proj:!",
                "indexer.q_layernorm:!",
                "indexer.k_layernorm:!",
                "q_proj:0",
                "q_norm:!",
                "k_proj:0",
                "k_norm:!",
                "v_proj:0",
                "o_proj:1",
            ),
            "linear_attn": (
                "conv1d:!",
                "in_proj_qkv:0",
                "in_proj_z:1",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "norm:!",
                "out_proj:2",
            ),
            "mlp:moe": {
                # Keep shared experts separate so placeholder expansion does not duplicate them.
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_expert_gate": ("shared_expert_gate:!",),
            },
        },
    ]


__all__ = ["Qwen4ExpQModel"]
