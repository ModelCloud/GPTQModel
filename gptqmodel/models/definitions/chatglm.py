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
        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        norms = (getattr(module, "input_layernorm", None), getattr(module, "post_attention_layernorm", None))
        if getattr(module, "apply_residual_connection_post_layernorm", False):
            # In post-layernorm residual mode both norm outputs are also the
            # residual stream, so a scale folded into either gain leaks past
            # its consumers into the block output. Keep only linear sources.
            return [node for node in nodes if all(node.get("prev_op") is not norm for norm in norms)]
        attention = getattr(module, "self_attention", None)
        for node in nodes:
            if node.get("prev_op") is norms[0] and node.get("module2inspect") is attention:
                # The packed attention block cannot be replayed on token-row
                # packed activations (GLM2/3 run sequence-first and every core
                # needs a sequence-aligned mask and rotary table), so score the
                # input-norm scale on the packed QKV projection it feeds.
                node.pop("module2inspect")
        return nodes

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        if not self.config.post_layer_norm:
            return inputs
        return super().lm_head_pre_quantize_generate_hook(inputs)
