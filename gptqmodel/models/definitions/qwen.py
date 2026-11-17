# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class QwenQModel(BaseQModel):
    loader = LocalModelLoader(
        "gptqmodel.models.ports.qwen.configuration_qwen.QWenConfig",
        "gptqmodel.models.ports.qwen.modeling_qwen.QWenLMHeadModel",
        tokenizer_class="gptqmodel.models.ports.qwen.tokenization_qwen.QWenTokenizer",
    )
    fusable = None

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Original fused-QKV/rotary/logn and gated-MLP replay is not scale-
        # qualified. Every declared linear remains in ordinary quantization.
        return []

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!",),
            "attn": ("c_attn:0", "c_proj:1"),
            "ln_2": ("ln_2:!",),
            "mlp": ("w1:0", "w2:0:nosrc", "c_proj:1"),
        },
    ]
