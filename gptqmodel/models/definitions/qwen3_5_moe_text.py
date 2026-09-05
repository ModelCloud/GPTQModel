# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from .qwen3_moe import Qwen3MoeQModel


class Qwen3_5_MoeTextQModel(Qwen3MoeQModel):
    shared_input_verified_model_types = frozenset({"qwen3_5_moe_text"})
    """
    Text-only Qwen 3.5/3.6 MoE shells use the causal LM loader and keep routed
    experts under `model.layers`.
    """

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.norm"

    rotary_embedding = "model.rotary_emb"

    out_of_model_tensors = {"prefixes": ["mtp"]}

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "q_proj:0:q:in=x", "k_norm:!", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1"),
            "linear_attn": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0:k:q:v:in=x",
                "in_proj_z:1:in=x",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert:0:shared": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
                "experts:0:routed:expert_activation=experts.act_fn": {
                    "#": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
                },
            },
        },
    ]
