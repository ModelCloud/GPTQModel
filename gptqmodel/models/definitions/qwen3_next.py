# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...quantization import METHOD
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Qwen3NextGPTQ(BaseQModel):
    """
    GPTQ config for Qwen3-Next (HF: Qwen3Next*), supporting:
      - Mixed token mixers per layer: 'full_attention' (self_attn.*) and 'linear_attention' (linear_attn.*)
      - Dense MLP (Qwen3NextMLP) and Sparse MoE (Qwen3NextSparseMoeBlock)
      - Dynamic expert indexing via config.num_experts
    """

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # -----------------------------------------------------------------------------
    # Preferred modern hierarchical spec. The loader will gracefully skip any
    # subpaths that don't exist on a given layer (e.g., dense vs MoE, or mixer type).
    # -----------------------------------------------------------------------------
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Token mixers
            "self_attn": ("q_norm:!", "k_norm:!", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn": ("norm:!", "conv1d:!", "in_proj_qkvz:0", "in_proj_ba:!:0", "out_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # MLP / MoE
            "mlp:moe": {
                # MoE router + shared expert (Qwen3NextSparseMoeBlock)
                "gate": ("gate:!",),  # router gate linear
                "shared_expert_gate": ("shared_expert_gate:!",), # <-- single (1, N) logic projections should not be quantized
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),

                # Experts list with dynamic index
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    module_tree_overrides = {
        METHOD.AWQ: [
            {
                "mlp:moe": {
                    "gate": ("gate",),
                }
            }
        ]
    }
