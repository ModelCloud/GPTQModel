# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class TeleChat2QModel(BaseQModel):
    loader = LocalModelLoader(
        "gptqmodel.models.ports.telechat2.configuration_telechat2.Telechat2Config",
        "gptqmodel.models.ports.telechat2.modeling_telechat2.Telechat2ForCausalLM",
        tokenizer_class="gptqmodel.models.ports.telechat2.tokenization_telechat2.TelechatTokenizerFactory",
    )
    fusable = None

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # `query` and `key_value` are the attention's input projections
            # (modeling_telechat2.py:358-359). The old tree declared only
            # `dense`, so both stayed fp16 while the coverage report still
            # claimed every spec matched.
            "self_attention": ("query:0", "key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Residual modes and interleaved KV/RoPE are not scale-search-qualified.
        return []

__all__ = ["TeleChat2QModel"]
