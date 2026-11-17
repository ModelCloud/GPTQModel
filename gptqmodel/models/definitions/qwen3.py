# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..fusable import FusableConfig, LlamaFusableConfig
from ..ports.qwen3 import Qwen3Loader
from . import LlamaQModel


class Qwen3QModel(LlamaQModel):
    """
    Qwen3 inherits the Llama-style layout but inserts Q/K RMS norm layers
    ahead of the attention projections. We mark those helper modules as
    non-quantized so the layer walker captures the complete structure.
    """

    fusable = FusableConfig("qwen3").merge(LlamaFusableConfig())

    loader = Qwen3Loader()

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Q/K have per-head RMSNorm after their projections. The shared
        # projection tree is an ownership contract, not fold/search/replay
        # qualification. Keep every projection quantizable without guessing
        # scale sources from this ordering. Derived definitions keep their
        # own native AWQ policy.
        if type(self) is Qwen3QModel:
            return []
        return super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)

    shared_input_verified_model_types = frozenset({"qwen3"})

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            "self_attn": (
                "q_norm:!:norm",
                "k_norm:!:norm",
                "q_proj:0:in=x",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        },
    ]
