# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class MiniMaxM2GPTQ(BaseQModel):
    """
    GPTQ config for MiniMax-M2 (HF: MiniMaxM2*), featuring:
      - Per-layer RMSNorm before/after attention
      - Standard attention with q/k normalization parameters
      - Sparse MoE block (MiniMaxM2SparseMoeBlock) with up to 256 experts
    """
    fusable = FusableConfig("minimax_m2").merge(LlamaFusableConfig(), profiles=("qkv",))


    pre_lm_head_norm_module = "model.norm"

    layer_modules_strict = False

    dynamic_expert_index = "num_local_experts"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
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
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {  # MoE module
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]


class MiniMaxQModel(BaseQModel):
    """GPTQ config for the original MiniMax-Text-01 family (HF: ``MiniMax*``).

    Unlike MiniMax-M2, this architecture interleaves full-attention layers
    (``q_proj``/``k_proj``/``v_proj``/``o_proj``) with linear-attention
    ("lightning") layers that expose a fused ``qkv_proj`` plus ``out_proj`` and
    ``output_gate``. Both variants keep their routed experts under
    ``mlp.experts`` with per-expert ``gate_proj``/``up_proj``/``down_proj``
    leaves after Defuser expansion.
    """
    fusable = FusableConfig("minimax").merge(LlamaFusableConfig(), profiles=("qkv",))


    pre_lm_head_norm_module = "model.norm"

    layer_modules_strict = False

    dynamic_expert_index = "num_local_experts"

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                # full-attention layers
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                # linear-attention ("lightning") layers
                "qkv_proj:0",
                "out_proj:1",
                "output_gate:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {  # MoE module
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]
