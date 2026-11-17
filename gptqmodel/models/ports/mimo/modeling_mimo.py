# SPDX-FileCopyrightText: 2025 Xiaomi Corporation
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""MiMo's actual Qwen2 extension, adapted to Transformers 5 attention APIs.

The pinned upstream causal-LM forward uses the ordinary decoder only. MTP
weights are nevertheless real checkpoint owners and must survive every save.
The auxiliary module can be called explicitly with a prepared causal mask and
rotary pair; speculative decoding and MTP training are not implemented here.
"""

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
)

from .configuration_mimo import MiMoConfig


class MiMoMTPLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.final_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position=None,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        # Upstream spells the pair singular. TF5 Qwen2 requires the plural
        # argument and returns (attention_output, weights). No checkpoint
        # dispatch or process-global monkeypatch is involved.
        if position_embeddings is None:
            position_embeddings = position_embedding
        if position_embeddings is None:
            raise ValueError("MiMo MTP requires prepared position_embeddings (cos, sin)")
        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return self.final_layernorm(hidden_states)


class MiMoModel(Qwen2Model):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.mtp_layers = nn.ModuleList([MiMoMTPLayers(config) for _ in range(config.num_nextn_predict_layers)])
        self.post_init()


class MiMoForCausalLM(Qwen2ForCausalLM):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = MiMoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
