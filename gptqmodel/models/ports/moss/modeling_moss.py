# Copyright (c) OpenMOSS, The EleutherAI and HuggingFace teams.
# SPDX-License-Identifier: Apache-2.0
"""MOSS Moon-003: four-way Q/V/K packing, FP32 rotary keys, parallel residual.

Adapted from OpenMOSS/MOSS; pinned sources are in NOTICE.
No checkpoint Python, global HF registration, optional kernels or dtype mutation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_moss import MossConfig


class MossDynamicLayer(DynamicLayer):
    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        # The source caches FP32 post-rotary K but compute-dtype V. Native
        # DynamicLayer initializes both empty tensors using K's dtype.
        self.values = value_states.new_empty(0)


class MossCache(DynamicCache):
    def __init__(self, config):
        Cache.__init__(self, layers=[MossDynamicLayer() for _ in range(config.n_layer)])


def _rotary(x, positions, dim):
    # Call-local FP32 tables also work after meta construction and .to(dtype).
    # The original sin/cos are FP32: do not narrow rotated keys to model dtype.
    inverse = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = positions.float().unsqueeze(-1) * inverse
    sine = angles.sin().repeat_interleave(2, -1).unsqueeze(2)
    cosine = angles.cos().repeat_interleave(2, -1).unsqueeze(2)
    rotated, rest = x[..., :dim], x[..., dim:]
    pairs = rotated.reshape(*rotated.shape[:-1], -1, 2)
    swapped = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
    return torch.cat((rotated * cosine + swapped * sine, rest), -1)


class MossAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads, self.head_dim = config.n_head, config.head_dim
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # This is a real checkpoint-owned buffer in all three released indices.
        self.register_buffer("causal_mask", torch.ones(
            (1, 1, config.n_positions, config.n_positions), dtype=torch.bool).tril())
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, layer_past=None, attention_mask=None,
                position_ids=None, head_mask=None, use_cache=False, output_attentions=False):
        batch, length, width = hidden_states.shape
        # Each of four partitions owns Q, V, K for its own contiguous heads.
        packed = self.qkv_proj(hidden_states).reshape(batch, length, 4, 3, width // 4)
        query, value, key = [packed[:, :, :, part].reshape(
            batch, length, self.num_attention_heads, self.head_dim) for part in range(3)]
        query = _rotary(query, position_ids, self.config.rotary_dim).transpose(1, 2)
        key = _rotary(key, position_ids, self.config.rotary_dim).transpose(1, 2)
        value = value.transpose(1, 2)
        if layer_past is not None:
            if use_cache:
                key, value = layer_past.update(key, value, self.layer_idx)
            elif layer_past.get_seq_length(self.layer_idx):
                old = layer_past.layers[self.layer_idx]
                key, value = torch.cat((old.keys, key), -2), torch.cat((old.values, value), -2)
        key_length = key.shape[-2]
        causal = self.causal_mask[:, :, key_length - length:key_length, :key_length]
        scores = query.float() @ key.float().transpose(-1, -2) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~causal, torch.finfo(torch.float32).min)
        if attention_mask is not None:
            # Keep the raw padding mask at the decoder boundary so calibration
            # can distinguish all-live rows from all-zero additive bias. Apply
            # the source's compute-dtype bias only at its attention consumer.
            if attention_mask.ndim == 2:
                attention_mask = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
            scores = scores + attention_mask
        probabilities = scores.softmax(-1).to(value.dtype)
        probabilities = self.attn_dropout(probabilities)
        if head_mask is not None:
            probabilities = probabilities * head_mask
        attended = (probabilities @ value).transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.out_proj(attended)), probabilities if output_attentions else None


class MossMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner = config.n_inner or 4 * config.n_embd
        self.fc_in = nn.Linear(config.n_embd, inner)
        self.fc_out = nn.Linear(inner, config.n_embd)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        return self.dropout(self.fc_out(self.act(self.fc_in(hidden_states))))


class MossBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MossAttention(config, layer_idx)
        self.mlp = MossMLP(config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None,
                position_ids=None, head_mask=None, use_cache=False, output_attentions=False):
        normalized = self.ln_1(hidden_states)
        attended, weights = self.attn(normalized, layer_past=layer_past, attention_mask=attention_mask,
                                      position_ids=position_ids, head_mask=head_mask,
                                      use_cache=use_cache, output_attentions=output_attentions)
        return attended + self.mlp(normalized) + hidden_states, weights


class MossPreTrainedModel(PreTrainedModel):
    config_class = MossConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["MossBlock"]
    _supports_sdpa = False

    def _init_weights(self, module):
        super()._init_weights(module)  # TF5 default: normal(initializer_range) Linear/Embedding, unit LayerNorm
        if isinstance(module, MossAttention):
            # HF's no-init/meta loader may need to initialize a missing buffer.
            init.copy_(module.causal_mask, torch.ones_like(module.causal_mask).tril())


class MossModel(MossPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([MossBlock(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, embeddings):
        self.wte = embeddings

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids or inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        hidden = self.wte(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = hidden.shape
        if past_key_values is None:
            past_key_values = MossCache(self.config) if use_cache else None
        elif type(past_key_values) is DynamicCache and past_key_values.get_seq_length() == 0:
            # GenerationMixin creates a standard empty cache before first forward.
            if past_key_values.offloading:
                raise ValueError("MOSS offloaded caches have not been qualified")
            past_key_values = MossCache(self.config)
        elif not isinstance(past_key_values, MossCache):
            raise ValueError("MOSS requires MossCache to preserve FP32 K and compute-dtype V")
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        key_length = past_length + length
        if key_length > self.config.n_positions:
            raise ValueError("MOSS sequence exceeds n_positions")
        if past_key_values is not None and any(
            past_key_values.get_seq_length(i) != past_length for i in range(self.config.n_layer)
        ):
            raise ValueError("MOSS cache layers have inconsistent lengths")
        if position_ids is None:
            position_ids = torch.arange(past_length, key_length, device=hidden.device)[None]
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("MOSS position_ids must match batch and current sequence")
        if position_ids.min() < 0 or position_ids.max() >= self.config.n_positions:
            raise ValueError("MOSS position_ids exceed n_positions")
        if attention_mask is not None:
            if attention_mask.shape != (batch, key_length):
                raise ValueError("MOSS attention_mask must cover the full cached sequence")
        if token_type_ids is not None:
            hidden = hidden + self.wte(token_type_ids)
        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = head_mask[None].expand(self.config.n_layer, -1)
            if head_mask.shape != (self.config.n_layer, self.config.n_head):
                raise ValueError("MOSS head_mask must have one value per head or per layer/head")
            head_mask = head_mask[:, None, :, None, None].to(hidden.dtype)
        hidden = self.drop(hidden)
        states, attentions = (() if output_hidden_states else None), (() if output_attentions else None)
        for i, block in enumerate(self.h):
            if output_hidden_states:
                states += (hidden,)
            hidden, weights = block(hidden, layer_past=past_key_values, attention_mask=attention_mask,
                                    position_ids=position_ids, head_mask=None if head_mask is None else head_mask[i],
                                    use_cache=use_cache, output_attentions=output_attentions)
            if output_attentions:
                attentions += (weights,)
        hidden = self.ln_f(hidden)
        if output_hidden_states:
            states += (hidden,)
        result = BaseModelOutputWithPast(last_hidden_state=hidden,
                                        past_key_values=past_key_values if use_cache else None,
                                        hidden_states=states, attentions=attentions)
        return result if return_dict else result.to_tuple()


class MossForCausalLM(MossPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = MossModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.post_init()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, logits_to_keep=0, **kwargs):
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values,
                                   attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                   use_cache=use_cache, output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states, return_dict=True)
        hidden = outputs.last_hidden_state
        selected = hidden[:, -logits_to_keep:] if isinstance(logits_to_keep, int) else hidden[:, logits_to_keep]
        logits = self.lm_head(selected).float()
        loss = None
        if labels is not None:
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("MOSS labels require logits_to_keep=0")
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                                   labels[:, 1:].reshape(-1).to(logits.device)).to(hidden.dtype)
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                       hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        return result if return_dict else result.to_tuple()
