# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class PhiQModel(BaseQModel):
    shared_input_verified_model_types = frozenset({"phi"})

    fusable = FusableConfig("phi").merge(LlamaFusableConfig(), profiles=("qkv",))

    pre_lm_head_norm_module = "model.final_layernorm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            # Phi runs attention and the MLP in parallel from one norm, and has
            # no post-attention norm. The shared norm cannot be a scale source
            # for either branch alone, and the MLP's GELU separates fc1 from fc2.
            "input_layernorm": ("input_layernorm:!:nosrc",),
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "dense:1:nosrc"),
            "mlp": ("fc1:0:nosrc", "fc2:1"),
        }
    ]
