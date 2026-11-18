# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package-owned GRIN-MoE decoder (PhiMoE-style SparseMixer routing with LayerNorm)."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_grinmoe import GRINMoEConfig


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class GRINMoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: GRINMoEConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class mp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        multiplier: torch.Tensor,
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
        ctx,
        grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expaned,
            None,
            None,
            None,
            None,
        )

def sparsemixer(scores, top_k, jitter_eps, training):
    assert top_k == 2

    ################ first expert ################

    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask
    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    if training:
        selected_experts = (
            masked_gates - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
    else:
        selected_experts = max_ind

    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    if training:
        # compute midpoint mask
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            selected_experts == max_ind,
            torch.rand_like(max_scores) > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        )
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

        multiplier = mp.apply(
            scores,
            multiplier_o,
            selected_experts,
            masked_gates,
            mask_for_one,
        )
    else:
        multiplier = multiplier_o

    # masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float('-inf'),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
    if training:
        selected_experts_top2 = (
            masked_gates_top2 - torch.empty_like(masked_gates_top2, memory_format=torch.legacy_contiguous_format).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
    else:
        selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    if training:
        # compute midpoint mask
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        )
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

        multiplier_top2 = mp.apply(
            scores,
            multiplier_top2_o,
            selected_experts_top2,
            masked_gates_top2,
            mask_for_one_top2,
        )
    else:
        multiplier_top2 = multiplier_top2_o

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )

class GRINMoESparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([GRINMoEBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.input_jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)

        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = sparsemixer(
            router_logits,
            top_k=self.top_k,
            jitter_eps=self.router_jitter_noise,
            training=self.training,
        )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits


class GRINMoERotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    def forward(self, x, position_ids):
        # Do not mutate a module-global cosine cache during concurrent forwards.
        # Rebuild FP32 phases after model.to(dtype) rather than upcasting rounded frequencies.
        inverse = self.inv_freq if self.inv_freq.dtype == torch.float32 else self.frequencies(device=x.device)
        freqs = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * inverse
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype).unsqueeze(1), emb.sin().to(x.dtype).unsqueeze(1)


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class GRINMoEAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GRINMoERotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value=None,
                output_attentions=False, **kwargs):
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value, position_ids)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        if past_key_value is not None:
            key, value = past_key_value.update(key, value, self.layer_idx)
        groups = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        weights = None
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
        else:
            weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            weights = F.softmax(weights + attention_mask, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = F.dropout(weights, p=self.config.attention_dropout, training=self.training)
            output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(output), weights if output_attentions else None


class GRINMoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = GRINMoEAttention(config, layer_idx)
        self.block_sparse_moe = GRINMoESparseMoeBlock(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, output_router_logits=False, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states, attention = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states, router_logits = self.block_sparse_moe(self.post_attention_layernorm(hidden_states))
        outputs = (residual + hidden_states,)
        if output_attentions:
            outputs += (attention,)
        if use_cache:
            outputs += (past_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


class GRINMoEPreTrainedModel(PreTrainedModel):
    config_class = GRINMoEConfig
    base_model_prefix = "model"
    _no_split_modules = ["GRINMoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GRINMoERotaryEmbedding):
            module.inv_freq.copy_(module.frequencies().to(module.inv_freq.device))


class GRINMoEModel(GRINMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([GRINMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                output_router_logits=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("GRIN-MoE supports Transformers DynamicCache, not legacy tuples or static caches")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, length, _ = inputs_embeds.shape
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_length = past_length + length
        if position_ids is None:
            position_ids = torch.arange(past_length, kv_length, device=inputs_embeds.device).unsqueeze(0)
        if position_ids.shape[-1] != length:
            raise ValueError("position_ids must describe the unprocessed tokens")
        if attention_mask is not None and attention_mask.ndim not in (2, 4):
            raise ValueError("attention_mask must be a 2D padding mask or 4D additive mask")
        if attention_mask is not None and attention_mask.ndim == 4:
            if attention_mask.shape != (batch, 1, length, kv_length):
                raise ValueError("4D attention_mask must cover query and complete KV lengths")
            causal_mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        else:
            if attention_mask is not None and attention_mask.shape != (batch, kv_length):
                raise ValueError("2D attention_mask must cover the complete KV length")
            queries = torch.arange(past_length, kv_length, device=inputs_embeds.device)[:, None]
            keys = torch.arange(kv_length, device=inputs_embeds.device)[None, :]
            allowed = keys <= queries
            # Preserve upstream's distinct eager/windowed and SDPA/full-context semantics.
            if self.config.sliding_window is not None and (
                self.config._attn_implementation == "eager" or output_attentions
            ):
                allowed = allowed & (keys >= queries - self.config.sliding_window)
            allowed = allowed[None, None].expand(batch, 1, length, kv_length)
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].to(device=allowed.device, dtype=torch.bool)
            causal_mask = torch.zeros(allowed.shape, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            causal_mask.masked_fill_(~allowed, torch.finfo(inputs_embeds.dtype).min)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        hidden_states = inputs_embeds
        all_hidden = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_routers = () if output_router_logits else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            outputs = layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=past_key_values, output_attentions=output_attentions,
                output_router_logits=output_router_logits, use_cache=use_cache,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions += (outputs[1],)
            if output_router_logits:
                all_routers += (outputs[-1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        result = MoeModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden, attentions=all_attentions, router_logits=all_routers,
        )
        return result if return_dict else result.to_tuple()


class GRINMoEForCausalLM(GRINMoEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = GRINMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, output_router_logits=None, return_dict=None,
                logits_to_keep=0, **kwargs):
        return_dict = self.config.return_dict if return_dict is None else return_dict
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits, return_dict=True, **kwargs,
        )
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices]).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits, self.config.num_local_experts, self.config.num_experts_per_tok, attention_mask,
            )
            if loss is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_loss.to(loss.device)
        result = MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_logits,
        )
        return result if return_dict else result.to_tuple()
