# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from .qwen3 import Qwen3QModel


class Qwen3_5TextQModel(Qwen3QModel):
    shared_input_verified_model_types = frozenset({"qwen3_5_text"})
    """
    Text-only Qwen 3.5 shells use the standard causal LM loader and keep their
    decoder stack directly under `model.layers`.
    """

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.norm"

    rotary_embedding = "model.rotary_emb"

    # Preserve auxiliary MTP/draft-head tensors when present.
    # Dense text-only Qwen3.5/Qwen3.6 models can ship mtp.* tensors in
    # auxiliary safetensors files.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    qvq_grouped_p32_candidates = {
        "qkv": (
            ("q_proj", "k_proj", "v_proj"),
            ("in_proj_qkv", "in_proj_z"),
        ),
        "gate_up": (("gate_proj", "up_proj"),),
    }

    qvq_transform_axis_overrides = {
        "self_attn.v_proj": (True, False),
        "mlp.gate_proj": (True, False),
        "mlp.up_proj": (True, False),
        "mlp.down_proj": (False, True),
    }

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
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
        },
    ]
