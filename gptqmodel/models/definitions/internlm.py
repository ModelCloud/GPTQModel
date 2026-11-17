# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from . import LlamaQModel


class InternLMQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    loader = LocalModelLoader(
        "gptqmodel.models.ports.internlm.configuration_internlm.InternLMConfig",
        "gptqmodel.models.ports.internlm.modeling_internlm.InternLMForCausalLM",
    )

    # Original InternLM checkpoints serialize per-layer rotary inv_freq. It is
    # a checkpoint-owned non-quantized buffer, not an undeclared runtime cache.
    module_tree = [
        "model", "layers", "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            "self_attn": ("rotary_emb:!", "q_proj:0:q:in=x", "k_proj:0:k:in=x",
                          "v_proj:0:v:in=x", "o_proj:1:o"),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # The port is not AWQ-scale-qualified. Keep every declared linear in
        # ordinary quantization, without inheriting guessed Llama folds.
        return []
