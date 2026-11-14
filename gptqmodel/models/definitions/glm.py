# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel

# GLM is HF-ied ChatGLM and marked by -HF suffix in THUDM hf repos
from ..fusable import FusableConfig, LlamaFusableConfig


class GlmQModel(BaseQModel):
    fusable = FusableConfig(("glm", "glm4")).merge(LlamaFusableConfig(), profiles=("qkv",))

    # q/k/v receive the same post-norm hidden state and gate/up the same MLP input,
    # so one Hessian capture covers each group. Verified against a real (tiny, CPU)
    # forward of the converted tree in
    # tests/module_tree/test_shared_input_cpu_forward.py.
    shared_input_verified_model_types = frozenset({"glm"})

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        }
    ]
