# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


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
            #"self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "linear_attn": ("in_proj_qkvz", "in_proj_ba:!", "out_proj"),  # conv1d intentionally excluded
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # MLP / MoE
            "mlp": {
                # MoE router + shared expert (Qwen3NextSparseMoeBlock)
                "gate": ("gate",),  # router gate linear
                "shared_expert_gate": ("shared_expert_gate:!",), # <-- single (1, N) logic projections should not be quantized
                "shared_expert": ("gate_proj", "up_proj", "down_proj"),

                # Experts list with dynamic index
                "experts": {
                    "#": ("gate_proj", "up_proj", "down_proj"),
                },
            },
        },
    ]
