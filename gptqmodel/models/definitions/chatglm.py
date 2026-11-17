# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from typing import List

import torch

from ..base import BaseQModel
from ..local import LocalModelLoader


class ChatGLMQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.chatglm.configuration_chatglm.ChatGLMConfig",
        model_class="gptqmodel.models.ports.chatglm.modeling_chatglm.ChatGLMForConditionalGeneration",
        tokenizer_class="gptqmodel.models.ports.chatglm.tokenization_chatglm.ChatGLMTokenizer",
    )

    pre_lm_head_norm_module = "transformer.encoder.final_layernorm"

    module_tree = [
        "transformer",
        "encoder",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attention": ("query_key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("dense_h_to_4h:0", "dense_4h_to_h:1"),
        },
    ]

    lm_head = "transformer.output_layer"

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Plain AWQ only. Packed MQA, sequence-first GLM2/3 replay, and the
        # post-layernorm residual mode require independent fold qualification.
        # An empty group list preserves the shared processor's quantize path.
        return []

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        if not self.config.post_layer_norm:
            return inputs
        return super().lm_head_pre_quantize_generate_hook(inputs)
