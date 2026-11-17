# SPDX-FileCopyrightText: 2024 OpenBMB
# SPDX-FileCopyrightText: 2024 HuggingFace Inc. team
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Original MiniCPM equations on current native Llama attention/cache primitives.

MiniCPM is not ordinary Llama: embeddings, both residual branches, and the
pre-logit representation have independent muP gains. All three are preserved.
"""

import math

from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel


class MiniCPMDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.gain = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def forward(self, hidden_states, **kwargs):
        attn, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = hidden_states + attn * self.gain
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) * self.gain


class MiniCPMModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([MiniCPMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb
        return super().forward(input_ids=None, inputs_embeds=inputs_embeds, **kwargs)


class MiniCPMForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                             use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, indices] / (self.config.hidden_size / self.config.dim_model_base)).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states, attentions=outputs.attentions)
