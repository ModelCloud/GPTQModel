# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForImageTextToText

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ..base import BaseQModel


class Qwen3_5_MoeQModel(BaseQModel):
    loader = AutoModelForImageTextToText

    require_load_processor = True

    layer_modules_strict = False

    require_monkeypatch = False

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.language_model.norm"

    rotary_embedding = "model.language_model.rotary_emb"

    out_of_model_tensors = {"prefixes": ["mtp"]}

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "q_proj:0", "k_norm:!", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0",
                "in_proj_z:1",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),  # <-- 0.5MB per layer. Not worth quantizing
                "shared_expert_gate": ("shared_expert_gate:!",),
                # Qwen 3.5/3.6 executes the shared expert path before routed experts
                # in modeling_qwen3_5_moe.py, so keep the module tree aligned with
                # real forward order. The previous reversed order hid a tree-vs-
                # execution mismatch until subset early-stop started relying on the
                # final module in the merged block.
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]
