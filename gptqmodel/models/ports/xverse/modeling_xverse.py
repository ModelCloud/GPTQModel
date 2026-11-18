# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2023-2025 XVERSE and 2026 ModelCloud.ai.
# SPDX-License-Identifier: Apache-2.0
"""Source-matched XVERSE dense, 256K and MoE decoder."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)

from .configuration_xverse import XverseConfig


class XverseRMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps
        self.moe = bool(config.num_experts)

    def forward(self, x):
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon)
        if self.moe:
            return self.weight * normalized.to(dtype)
        return (self.weight * normalized).to(dtype)


class XverseRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=config.xverse_rope_persistent)

    def _apply(self, fn, recurse=True):
        original = self.inv_freq
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self.inv_freq.float() if original.is_meta else original.to(device=self.inv_freq.device)
        return self

    def forward(self, x, position_ids, seq_len):
        c = self.config
        positions = position_ids.float()
        frequency = self.inv_freq.float()
        if c.xverse_rope_type == "dynamic_256k" and seq_len > c.max_position_embeddings:
            base = c.rope_theta * (seq_len / (c.max_position_embeddings / 2) - 1) ** (self.dim / (self.dim - 2))
            frequency = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim))
        if c.xverse_rope_scaling is not None:
            factor = c.xverse_rope_scaling["factor"]
            if c.xverse_rope_scaling["type"] == "linear":
                positions = positions / factor
            elif position_ids.max() + 1 > c.max_position_embeddings:
                length = position_ids.max() + 1
                base = c.rope_theta * (factor * length / c.max_position_embeddings - (factor - 1)) ** (
                    self.dim / (self.dim - 2)
                )
                frequency = 1.0 / (
                    base ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim)
                )
                self.inv_freq = frequency
        with torch.autocast(device_type=x.device.type, enabled=False):
            angles = positions[..., None] * frequency[None, None, :]
            angles = torch.cat([angles, angles], dim=-1)
            return angles.cos().to(x.dtype).unsqueeze(1), angles.sin().to(x.dtype).unsqueeze(1)


def _rotate(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat([-right, left], dim=-1)


class XverseMLP(nn.Module):
    def __init__(self, config, width=None):
        super().__init__()
        self.tp = config.pretraining_tp
        width = width or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.tp == 1:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        gates = torch.cat([F.linear(x, w) for w in self.gate_proj.weight.chunk(self.tp, 0)], -1)
        ups = torch.cat([F.linear(x, w) for w in self.up_proj.weight.chunk(self.tp, 0)], -1)
        return sum(
            F.linear(y, w)
            for y, w in zip((self.act_fn(gates) * ups).chunk(self.tp, -1), self.down_proj.weight.chunk(self.tp, 1))
        )


class XverseMoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        self.num_experts = config.num_experts
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=torch.float32)
        self.experts = nn.ModuleList([XverseMLP(config) for _ in range(config.num_experts)])
        if config.num_shared_experts is not None:
            self.shared_experts = XverseMLP(config, config.intermediate_size * config.num_shared_experts)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = self.router(flat.float())
        weights, selected = logits.softmax(-1, dtype=torch.float32).topk(self.top_k, dim=-1)
        weights = (weights / (weights.sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
        result = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            tokens, slots = torch.where(selected == index)
            if tokens.numel():
                result.index_add_(0, tokens, expert(flat[tokens]) * weights[tokens, slots, None])
        result = result.view_as(x)
        if hasattr(self, "shared_experts"):
            result = result + self.shared_experts(x)
        return result, logits


class XverseAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = XverseRotaryEmbedding(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        b, t, _ = hidden_states.shape

        def project(layer, heads):
            if self.config.pretraining_tp == 1:
                out = layer(hidden_states)
            else:
                out = torch.cat(
                    [F.linear(hidden_states, w) for w in layer.weight.chunk(self.config.pretraining_tp, 0)], -1
                )
            return out.view(b, t, heads, self.head_dim).transpose(1, 2)

        q = project(self.q_proj, self.num_heads)
        k = project(self.k_proj, self.num_key_value_heads)
        v = project(self.v_proj, self.num_key_value_heads)
        length = t + (past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0)
        cos, sin = self.rotary_emb(v, position_ids, length)
        q, k = q * cos + _rotate(q) * sin, k * cos + _rotate(k) * sin
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        groups = self.num_heads // self.num_key_value_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=self.config.attention_dropout if self.training else 0.0
            )
            probabilities = None
        else:
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = (scores + attention_mask).clamp_min(torch.finfo(scores.dtype).min)
            probabilities = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            probabilities = F.dropout(probabilities, p=self.config.attention_dropout, training=self.training)
            out = probabilities @ v
        out = out.transpose(1, 2).reshape(b, t, -1)
        if self.config.pretraining_tp == 1:
            out = self.o_proj(out)
        else:
            out = sum(
                F.linear(y, w)
                for y, w in zip(
                    out.chunk(self.config.pretraining_tp, -1), self.o_proj.weight.chunk(self.config.pretraining_tp, 1)
                )
            )
        return out, probabilities if output_attentions else None


class XverseDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = XverseAttention(config, layer_idx)
        self.mlp = XverseMoEMLP(config) if config.num_experts else XverseMLP(config)
        self.input_layernorm = XverseRMSNorm(config)
        self.post_attention_layernorm = XverseRMSNorm(config)

    def forward(self, hidden_states, **kwargs):
        attended, weights = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = hidden_states + attended
        mlp = self.mlp(self.post_attention_layernorm(hidden_states))
        routed = isinstance(mlp, tuple)
        return hidden_states + (mlp[0] if routed else mlp), weights, mlp[1] if routed else None


class XversePreTrainedModel(PreTrainedModel):
    config_class = XverseConfig
    base_model_prefix = "model"
    _no_split_modules = ["XverseDecoderLayer"]
    _supports_sdpa = True
    _keep_in_fp32_modules_strict = ["router", "inv_freq"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, XverseRMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, XverseRotaryEmbedding):
            frequencies = 1.0 / (
                module.config.rope_theta
                ** (torch.arange(0, module.dim, 2, device=module.inv_freq.device, dtype=torch.float32) / module.dim)
            )
            module.inv_freq.copy_(frequencies)


class XverseModel(XversePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([XverseDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = XverseRMSNorm(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        output_router_logits = (
            self.config.output_router_logits if output_router_logits is None else output_router_logits
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        legacy = isinstance(past_key_values, (tuple, list))
        if legacy:
            cache = DynamicCache()
            for i, (k, v) in enumerate(past_key_values):
                cache.update(k, v, i)
        elif past_key_values is None or isinstance(past_key_values, DynamicCache):
            cache = past_key_values
        else:
            raise ValueError("XVERSE supports DynamicCache or legacy tuple caches only")
        if cache is None and use_cache:
            cache = DynamicCache()
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        b, t, _ = x.shape
        past = cache.get_seq_length() if cache is not None else 0
        slots = torch.arange(past, past + t, device=x.device)
        if position_ids is None:
            position_ids = slots.unsqueeze(0)
        if attention_mask is None or attention_mask.ndim == 2:
            total = past + t
            allowed = torch.arange(total, device=x.device)[None, :] <= slots[:, None]
            allowed = allowed[None, None].expand(b, 1, t, total)
            if attention_mask is not None:
                if attention_mask.shape != (b, total):
                    raise ValueError("XVERSE attention_mask must cover the complete cached sequence")
                allowed = allowed & attention_mask[:, None, None, :].bool()
            mask = torch.zeros((b, 1, t, total), device=x.device, dtype=x.dtype).masked_fill(
                ~allowed, torch.finfo(x.dtype).min
            )
        elif attention_mask.ndim == 4 and attention_mask.shape == (b, 1, t, past + t):
            mask = attention_mask.to(x.dtype)
        else:
            raise ValueError("Invalid XVERSE attention_mask shape")
        states, attentions, routers = (), (), ()
        for layer in self.layers:
            if output_hidden_states:
                states += (x,)
            x, weights, router = layer(
                x,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            if output_attentions:
                attentions += (weights,)
            if output_router_logits and router is not None:
                routers += (router,)
        x = self.norm(x)
        if output_hidden_states:
            states += (x,)
        result_cache = cache if use_cache else None
        if legacy and result_cache is not None:
            result_cache = tuple((layer.keys, layer.values) for layer in cache.layers)
        output = {
            "last_hidden_state": x,
            "past_key_values": result_cache,
            "hidden_states": states if output_hidden_states else None,
            "attentions": attentions if output_attentions else None,
        }
        if self.config.num_experts:
            result = MoeModelOutputWithPast(**output, router_logits=routers if output_router_logits else None)
        else:
            result = BaseModelOutputWithPast(**output)
        return result if return_dict else result.to_tuple()


class XverseForCausalLM(XversePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = XverseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def _build_chat_input(self, tokenizer, messages, max_new_tokens=2048):
        max_new_tokens = max_new_tokens or self.generation_config.max_new_tokens or 2048
        budget = min(
            self.config.max_tokenizer_truncation,
            max(self.config.max_position_embeddings // 2, self.config.max_position_embeddings - max_new_tokens),
        )
        moe = bool(self.config.num_experts)
        extended = moe or self.config.xverse_rope_type == "dynamic_256k"
        prompts = {
            "user": "user: " if moe else "Human: ",
            "assistant": "assistant: " if moe else "Assistant: ",
            "exec": "exec: " if moe else "Exec: ",
            "system": "system: " if moe else "Human: ",
        }

        def encode(text):
            return tokenizer.encode(text, return_token_type_ids=False)

        assistant = encode(prompts["assistant"])
        total, current = [], []
        for index, message in enumerate(reversed(messages)):
            role, content = message["role"], message["content"]
            if role == "user" or (extended and role == "exec"):
                tokens = encode(prompts[role]) + encode(content + ("\n" if moe else "\n\n"))
                if index == 0:
                    tokens = tokens[: budget - len(assistant)] + assistant
                current = tokens + current
                if index and len(total) + len(current) > budget:
                    break
                total = current + total
                current = []
                if index == 0 and len(total) >= budget:
                    break
            elif role == "assistant":
                current = assistant + encode(content) + [self.generation_config.eos_token_id] + current
            elif extended and role == "system":
                if index != len(messages) - 1:
                    raise ValueError("XVERSE system message must be first")
                current = encode(prompts[role]) + encode(content + "\n") + current
                if len(total) + len(current) <= budget:
                    total = current + total
                break
            else:
                raise ValueError(f"message role not supported yet: {role}")
        return torch.tensor([total], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def chat(self, tokenizer, messages, stream=False, generation_config=None):
        if stream:
            raise NotImplementedError("XVERSE threaded chat streaming has not been qualified")
        policy = generation_config if generation_config is not None else self.generation_config
        ids = self._build_chat_input(tokenizer, messages, policy.max_new_tokens)
        result = self.generate(ids, generation_config=policy)
        return tokenizer.decode(result[0, ids.shape[-1] :], skip_special_tokens=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,
            cache_position=cache_position,
        )
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden = output.last_hidden_state[:, selection]
        if self.config.pretraining_tp == 1:
            logits = self.lm_head(hidden).float()
        else:
            logits = torch.cat(
                [F.linear(hidden, w) for w in self.lm_head.weight.chunk(self.config.pretraining_tp, 0)], -1
            ).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1).to(logits.device)
            )
        values = {
            "loss": loss,
            "logits": logits,
            "past_key_values": output.past_key_values,
            "hidden_states": output.hidden_states,
            "attentions": output.attentions,
        }
        if self.config.num_experts:
            aux = None
            if output.router_logits is not None:
                probabilities = torch.cat(output.router_logits, 0).softmax(-1)
                chosen = F.one_hot(
                    probabilities.topk(self.config.moe_top_k, -1).indices, self.config.num_experts
                ).float()
                if attention_mask is None:
                    tokens = chosen.mean(0)
                    means = probabilities.mean(0)
                else:
                    valid = attention_mask.reshape(-1).repeat(len(output.router_logits)).float()
                    tokens = (chosen * valid[:, None, None]).sum(0) / valid.sum()
                    means = (probabilities * valid[:, None]).sum(0) / valid.sum()
                aux = (tokens * means.unsqueeze(0)).sum() * self.config.num_experts
                if loss is not None:
                    values["loss"] = loss + self.config.router_aux_loss_coef * aux
            result = MoeCausalLMOutputWithPast(**values, aux_loss=aux, router_logits=output.router_logits)
        else:
            result = CausalLMOutputWithPast(**values)
        return result if (self.config.return_dict if return_dict is None else return_dict) else result.to_tuple()
