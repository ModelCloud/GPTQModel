# SPDX-FileCopyrightText: 2025 Kwai-Klear Team, Kuaishou
# SPDX-FileCopyrightText: 2024-2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Transformers 5 Klear port. Source revisions are in NOTICE."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel, slice_logits_to_keep
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from .configuration_klear import KlearConfig


# Klear's attention, norms, RoPE and dense MLP are exactly Qwen3-MoE's (same
# equations and parameter names); only routing and the shared-expert mix differ.
class KlearRMSNorm(Qwen3MoeRMSNorm):
    pass


class KlearRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    pass


class KlearAttention(Qwen3MoeAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.q_norm = KlearRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = KlearRMSNorm(self.head_dim, eps=config.rms_norm_eps)


class KlearMLP(Qwen3MoeMLP):
    pass


class KlearSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.top_k, self.norm_topk_prob = (
            config.num_experts, config.num_experts_per_tok, config.norm_topk_prob
        )
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([KlearMLP(config, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.shared_experts = KlearMLP(config, config.moe_intermediate_size * config.n_shared_experts)
        self.coefficient = nn.Linear(config.hidden_size, 2)
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        residual = hidden_states
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(flat.float(), self.gate.weight.float())
        probabilities = torch.sigmoid(router_logits)
        selected = torch.topk(probabilities + self.expert_bias.unsqueeze(0), self.top_k, dim=-1).indices
        weights = probabilities.gather(-1, selected)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(flat.dtype)
        result = torch.zeros_like(flat)
        mask = F.one_hot(selected, self.num_experts).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum((-1, -2)), 0).nonzero().flatten():
            slot, token = torch.where(mask[expert_idx])
            expert_output = self.experts[expert_idx](flat[token]) * weights[token, slot, None]
            result.index_add_(0, token, expert_output.to(flat.dtype))
        coefficient = self.coefficient(residual).softmax(-1)
        result = result.reshape_as(residual) * coefficient[..., :1]
        result = result + self.shared_experts(residual) * coefficient[..., 1:]
        return result, router_logits


class KlearDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KlearAttention(config, layer_idx)
        sparse = config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = KlearSparseMoeBlock(config) if sparse and layer_idx not in config.mlp_only_layers else KlearMLP(config)
        self.input_layernorm = KlearRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KlearRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        residual = hidden_states
        output, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
        )
        hidden_states = residual + output
        residual = hidden_states
        output = self.mlp(self.post_attention_layernorm(hidden_states))
        if isinstance(output, tuple):
            output = output[0]
        return residual + output


class KlearPreTrainedModel(PreTrainedModel):
    config_class = KlearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KlearDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules_strict = ["expert_bias"]
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "router_logits": OutputRecorder(KlearSparseMoeBlock, index=1),
        "hidden_states": KlearDecoderLayer,
        "attentions": KlearAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)  # also covers KlearRMSNorm and KlearRotaryEmbedding
        if isinstance(module, KlearSparseMoeBlock):
            init.zeros_(module.expert_bias)


class KlearModel(KlearPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([KlearDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = KlearRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = KlearRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Klear past_key_values must be a Transformers Cache")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen).unsqueeze(0)
        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask[self.config.layer_types[0]]
        else:
            attention_mask = mask_function(
                config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                past_key_values=past_key_values, position_ids=position_ids,
            )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings,
                position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None,
        )


class KlearMoeForCausalLM(KlearPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = KlearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep))
        loss = self.loss_function(logits, labels, self.vocab_size, **kwargs) if labels is not None else None
        return MoeCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_logits,
        )


__all__ = ["KlearMoeForCausalLM", "KlearModel", "KlearPreTrainedModel"]
