# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers import AutoModelForImageTextToText

from . import LlamaQModel


class Qwen3_5QModel(LlamaQModel):
    """
    Qwen3_5 inherits the Llama-style layout but inserts Q/K RMS norm layers
    ahead of the attention projections. We mark those helper modules as
    non-quantized so the layer walker captures the complete structure.
    """

    loader = AutoModelForImageTextToText

    require_load_processor = True

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.language_model.norm"

    rotary_embedding = "model.language_model.rotary_emb"

    # Preserve auxiliary MTP/draft-head tensors when present.
    # Qwen3_5_MoeQModel already does this; dense Qwen3.5/Qwen3.6 models
    # can also ship mtp.* tensors in auxiliary safetensors files.
    out_of_model_tensors = {"prefixes": ["mtp"]}

    # The full-attention projections and the two large linear-attention input
    # projections consume the same activation.  Declare both groups here so
    # QVQ can share one input rotation and retain child-local split schedules;
    # the generic grouped runtime remains architecture- and role-agnostic.
    qvq_grouped_p32_candidates = {
        "qkv": (
            ("q_proj", "k_proj", "v_proj"),
            ("in_proj_qkv", "in_proj_z"),
        ),
        "gate_up": (("gate_proj", "up_proj"),),
    }

    # Canonical A41 omits V's module-local output transform.  Qwen3.8-27B's
    # 17,408-wide intermediate dimension also has no supported exact
    # composite Hadamard base (17 * 1024).  Keep gate/up in their native output
    # basis and quantize down in that same native input basis.  This is not a
    # transform moved through SwiGLU: each P32 linear independently encodes its
    # dense weight under the declared axes, so the nonlinear boundary and its
    # FP16 rounding semantics remain unchanged.
    qvq_transform_axis_overrides = {
        "self_attn.v_proj": (True, False),
        "mlp.gate_proj": (True, False),
        "mlp.up_proj": (True, False),
        "mlp.down_proj": (False, True),
    }

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "q_proj:0:q", "k_norm:!", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "linear_attn": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0:k:q:v",
                "in_proj_z:1",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        },
    ]
