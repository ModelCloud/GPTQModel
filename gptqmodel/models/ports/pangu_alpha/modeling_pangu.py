# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""PyTorch implementation of the official PanGu-α top-query architecture.

The checkpoint ABI follows the released 2.6B conversion. See NOTICE for the
official architecture reference and the distinction between code and weights.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_pangu import GPTPanguConfig


class GPTPanguAttention(nn.Module):
    bias: torch.Tensor
    masked_bias: torch.Tensor

    def __init__(self, config, layer_idx):
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = width // self.num_heads
        self.layer_idx = layer_idx
        self.scale_attn_weights = config.scale_attn_weights
        length = config.max_position_embeddings
        self.register_buffer("bias", torch.ones(length, length, dtype=torch.uint8).tril()[None, None])
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.c_proj = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        custom_query=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        batch, length, width = hidden_states.shape

        def split(x):
            return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query = split(self.q_proj(hidden_states if custom_query is None else custom_query))
        key, value = split(self.k_proj(hidden_states)), split(self.v_proj(hidden_states))
        if layer_past is not None:
            if use_cache:
                key, value = layer_past.update(key, value, self.layer_idx)
            elif self.layer_idx < len(layer_past.layers) and layer_past.layers[self.layer_idx].is_initialized:
                past = layer_past.layers[self.layer_idx]
                key = torch.cat((past.keys, key), dim=-2)
                value = torch.cat((past.values, value), dim=-2)
        scores = query @ key.transpose(-1, -2)
        if self.scale_attn_weights:
            scores = scores / math.sqrt(self.head_dim)
        keys = key.shape[-2]
        causal = self.bias[:, :, keys - length : keys, :keys].bool()
        scores = torch.where(causal, scores, self.masked_bias.to(scores.dtype))
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = (~attention_mask[:, None, None].bool()).to(scores.dtype) * -1e4
            scores = scores + attention_mask
        weights = self.attn_dropout(F.softmax(scores, dim=-1).to(value.dtype))
        if head_mask is not None:
            weights = weights * head_mask
        context = (weights @ value).transpose(1, 2).contiguous().view(batch, length, width)
        output = self.resid_dropout(self.c_proj(context))
        result = (output, layer_past if use_cache else None)
        return result + (weights,) if output_attentions else result


class GPTPanguMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner = config.intermediate_size or 4 * config.hidden_size
        self.c_fc = nn.Linear(config.hidden_size, inner)
        self.c_proj = nn.Linear(inner, config.hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        return self.dropout(self.c_proj(self.act(self.c_fc(hidden_states))))


class GPTPanguBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTPanguAttention(config, layer_idx)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTPanguMLP(config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        custom_query=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        attention = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            custom_query=custom_query,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attention[0]
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return (hidden_states,) + (attention[1:] if use_cache else attention[2:])


class GPTPanguPreTrainedModel(PreTrainedModel):
    config_class = GPTPanguConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GPTPanguBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, (GPTPanguAttention, GPTPanguMLP)):
            std = self.config.initializer_range / math.sqrt(2 * self.config.num_layers)
            init.normal_(module.c_proj.weight, std=std)


class GPTPanguModel(GPTPanguPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.wqe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTPanguBlock(config, i) for i in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise TypeError(
                "GPT-Pangu supports only DynamicCache; legacy tuples/static/offloaded caches are unqualified"
            )
        if past_key_values is not None and past_key_values.offloading:
            raise ValueError("GPT-Pangu offloaded caches are not qualified")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        inputs_embeds = self.wte(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length = inputs_embeds.shape[:2]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        total = past_length + length
        if total > self.config.max_position_embeddings:
            raise ValueError("GPT-Pangu context exceeds max_position_embeddings")
        if position_ids is None:
            position_ids = torch.arange(past_length, total, device=inputs_embeds.device)[None]
        if position_ids.ndim != 2 or position_ids.shape[-1] != length or position_ids.shape[0] not in (1, batch):
            raise ValueError("position_ids must match the current batch/sequence")
        hidden = inputs_embeds + self.wpe(position_ids)
        if token_type_ids is not None:
            hidden = hidden + self.wte(token_type_ids)
        hidden = self.drop(hidden)
        query = self.wqe(position_ids).expand(batch, -1, -1)
        if attention_mask is not None:
            # Keep the binary mask at the block boundary for calibration replay.
            keep_mask = attention_mask.ndim == 2 and attention_mask.shape == (batch, total)
            additive_mask = (
                attention_mask.ndim == 4
                and attention_mask.shape[0] in (1, batch)
                and attention_mask.shape[1] in (1, self.config.num_heads)
                and attention_mask.shape[-2] in (1, length)
                and attention_mask.shape[-1] == total
                and attention_mask.is_floating_point()
            )
            if not (keep_mask or additive_mask):
                raise ValueError("attention_mask must be a full 2D keep mask or broadcastable 4D additive mask")
        if head_mask is not None:
            if head_mask.shape == (self.config.num_heads,):
                head_mask = head_mask[None, None, :, None, None].expand(len(self.h), -1, -1, -1, -1)
            elif head_mask.shape == (len(self.h), self.config.num_heads):
                head_mask = head_mask[:, None, :, None, None]
            else:
                raise ValueError("head_mask must have shape [heads] or [layers, heads]")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        states = []
        attentions = []
        for i, block in enumerate(self.h):
            final = i == len(self.h) - 1
            if final:
                hidden = self.ln_f(hidden)
            if output_hidden_states:
                states.append(hidden)
            outputs = block(
                hidden,
                layer_past=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=None if head_mask is None else head_mask[i],
                custom_query=query if final else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden = outputs[0]
            if output_attentions:
                attentions.append(outputs[2 if use_cache else 1])
        if output_hidden_states:
            states.append(hidden)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(states) if output_hidden_states else None,
            attentions=tuple(attentions) if output_attentions else None,
        )
        return output if return_dict else output.to_tuple()


class GPTPanguForCausalLM(GPTPanguPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTPanguModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        mask = model_kwargs.get("attention_mask")
        return positions if mask is None else positions.masked_fill(mask == 0, 1)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden = outputs.last_hidden_state if labels is not None else outputs.last_hidden_state[:, selection]
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1).to(logits.device),
            )
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        return output if return_dict else output.to_tuple()
