# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""PyTorch implementation of the MiniMax M2 architecture for Hugging Face Transformers."""

from __future__ import annotations

import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, repeat_kv, rotate_half

from .configuration_minimax_m2 import MiniMaxM2Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniMaxM2Config"
_CHECKPOINT_FOR_DOC = "MiniMaxAI/MiniMax-M2"


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gate_logits is None:
        return torch.tensor(0.0)
    if isinstance(gate_logits, torch.Tensor):
        logits = gate_logits
    else:
        logits = torch.cat([layer_gate.to(gate_logits[0].device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
    _, selected = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, seq_len = attention_mask.shape
        num_layers = logits.shape[0] // (batch_size * seq_len)

        expanded_mask = (
            attention_mask[None, :, :, None, None]
            .expand(num_layers, batch_size, seq_len, top_k, num_experts)
            .reshape(-1, top_k, num_experts)
            .to(logits.device)
        )
        tokens_per_expert = torch.sum(expert_mask.float() * expanded_mask, dim=0) / torch.sum(expanded_mask, dim=0)

        router_mask = (
            attention_mask[None, :, :, None]
            .expand(num_layers, batch_size, seq_len, num_experts)
            .reshape(-1, num_experts)
            .to(logits.device)
        )
        router_prob_per_expert = torch.sum(routing_weights * router_mask, dim=0) / torch.sum(router_mask, dim=0)

    loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return loss * num_experts


def apply_rotary_pos_emb_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)[..., :rotary_dim]
    sin = sin.unsqueeze(unsqueeze_dim)[..., :rotary_dim]
    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q = torch.cat((q_rot, q[..., rotary_dim:]), dim=-1)
    k = torch.cat((k_rot, k[..., rotary_dim:]), dim=-1)
    return q, k


class MiniMaxM2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class MiniMaxM2MLP(nn.Module):
    def __init__(self, config: MiniMaxM2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.w1(hidden_states))
        up = self.w3(hidden_states)
        hidden_states = gate * up
        hidden_states = self.w2(hidden_states)
        return hidden_states


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(self, config: MiniMaxM2Config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn = nn.ModuleList([MiniMaxM2MLP(config) for _ in range(config.num_local_experts)])
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.use_routing_bias = config.use_routing_bias
        self.scoring_func = getattr(config, "scoring_func", "softmax")

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        else:
            self.register_parameter("e_score_correction_bias", None)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            noise = torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise,
                1.0 + self.jitter_noise,
            )
            hidden_states = hidden_states * noise

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states.to(torch.float32))
        router_logits = router_logits.to(torch.float32)
        if self.e_score_correction_bias is not None:
            # Bias is applied after scoring (see vLLM/SGLang implementations).
            correction_bias = self.e_score_correction_bias.to(router_logits.device, router_logits.dtype)
        else:
            correction_bias = None

        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
        elif self.scoring_func == "softmax":
            scores = torch.softmax(router_logits, dim=-1)
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        if correction_bias is not None:
            original_scores = scores
            scores = scores + correction_bias
        else:
            original_scores = scores

        topk_scores, selected_experts = torch.topk(scores, self.top_k, dim=-1)
        if correction_bias is not None:
            routing_weights = original_scores.gather(1, selected_experts)
        else:
            routing_weights = topk_scores

        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        routing_weights = routing_weights.to(hidden_states.dtype)
        selected_experts = selected_experts.to(torch.long)

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.nonzero(expert_mask.sum(dim=(-1, -2)) > 0, as_tuple=False).flatten()

        for expert_idx in expert_hit.tolist():
            expert_layer = self.ffn[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            token_states = hidden_states.index_select(0, top_x)
            expert_output = expert_layer(token_states) * routing_weights[top_x, idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, top_x, expert_output.to(final_hidden_states.dtype))

        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits


class MiniMaxM2Attention(nn.Module):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rotary_dim = config.rotary_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = MiniMaxM2RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = MiniMaxM2RMSNorm(self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

        rope_config = copy.deepcopy(config)
        rope_config.hidden_size = config.hidden_size
        rope_config.num_attention_heads = config.num_attention_heads
        rope_config.partial_rotary_factor = float(config.rotary_dim) / float(self.head_dim)
        self.rotary_emb = LlamaRotaryEmbedding(rope_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q_flat = query_states.transpose(1, 2).reshape(bsz * q_len, -1)
            k_flat = key_states.transpose(1, 2).reshape(bsz * q_len, -1)
            q_flat = self.q_norm(q_flat)
            k_flat = self.k_norm(k_flat)
            query_states = q_flat.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = k_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb_partial(
            query_states.transpose(1, 2), key_states.transpose(1, 2), cos, sin, self.rotary_dim
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class MiniMaxM2DecoderLayer(nn.Module):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniMaxM2Attention(config, layer_idx)
        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config)
        self.input_layernorm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + moe_output

        return hidden_states, router_logits, attn_weights


class MiniMaxM2PreTrainedModel(PreTrainedModel):
    config_class = MiniMaxM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM2DecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_attention_backend = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MiniMaxM2Model(MiniMaxM2PreTrainedModel):
    def __init__(self, config: MiniMaxM2Config) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniMaxM2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.rotary_emb = LlamaRotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MoeModelOutputWithPast, Tuple]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.config.sliding_window is not None:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            router_logits = layer_outputs[1]
            attn_weights = layer_outputs[2]

            if output_router_logits:
                all_router_logits = all_router_logits + (router_logits,)
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states, past_key_values)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            if output_router_logits:
                outputs += (all_router_logits,)
            return outputs

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
        )


class MiniMaxM2ForCausalLM(MiniMaxM2PreTrainedModel, GenerationMixin):
    def __init__(self, config: MiniMaxM2Config) -> None:
        super().__init__(config)
        self.model = MiniMaxM2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -past_key_values.get_seq_length() - 1 :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[MoeCausalLMOutputWithPast, Tuple]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,
        )

        hidden_states = model_outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        aux_loss = None
        if output_router_logits and model_outputs.router_logits is not None:
            aux_loss = load_balancing_loss_func(
                model_outputs.router_logits,
                num_experts=self.num_experts,
                top_k=self.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            if loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + (model_outputs.past_key_values,)
            if output_hidden_states:
                output += (model_outputs.hidden_states,)
            if output_attentions:
                output += (model_outputs.attentions,)
            if output_router_logits:
                output += (model_outputs.router_logits,)
            return ((loss,) + output) if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            router_logits=model_outputs.router_logits,
        )


__all__ = [
    "MiniMaxM2RMSNorm",
    "MiniMaxM2SparseMoeBlock",
    "MiniMaxM2Attention",
    "MiniMaxM2DecoderLayer",
    "MiniMaxM2Model",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2PreTrainedModel",
]
