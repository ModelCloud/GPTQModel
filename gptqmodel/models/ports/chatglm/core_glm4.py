# SPDX-FileCopyrightText: 2023 THUDM
# SPDX-FileCopyrightText: 2024 Zhipu AI
# SPDX-FileCopyrightText: 2026 Z.AI Co., Ltd
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Adapted from THUDM ChatGLM sources.
"""GLM4's batch-first checkpoint computation, including its extended RoPE base.

Modules whose math and layout match ChatGLM2/3 are inherited from ``core``.
"""
import torch
import torch.nn.functional as F

from . import core
from .core import split_tensor_along_last_dim


class RotaryEmbedding(core.RotaryEmbedding):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__(dim, original_impl=original_impl, device=device, dtype=dtype)
        self.rope_ratio = rope_ratio

    def forward_impl(self, seq_len, n_elem, dtype, device, base=10000):
        return super().forward_impl(seq_len, n_elem, dtype, device, base=base * self.rope_ratio)


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq = x.size(0), x.size(1), x.size(2)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class CoreAttention(core.CoreAttention):
    def __init__(self, config, layer_number):
        super().__init__(config, layer_number)
        self.config = config
        self.is_causal = True

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # [b, np, sq, sk]
        output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))

        # [b, np, sq, hn] -> [b * np, sq, hn]
        query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
        # [b, np, sk, hn] -> [b * np, sk, hn]
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
            device=query_layer.device
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # query layer shape: [b * np, sq, hn]
        # value layer shape: [b, np, sk, hn]
        # attention shape: [b, np, sq, sk]
        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(1), value_layer.size(3))
        # change view [b * np, sk, hn]
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.transpose(1, 2).contiguous()
        # [b, sq, np, hn] --> [b, sq, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer


class SdpaAttention(CoreAttention):
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             is_causal=True,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             attention_mask,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


CORE_ATTENTION_CLASSES = {"eager": CoreAttention, "sdpa": SdpaAttention}


class SelfAttention(core.SelfAttention):
    def __init__(self, config, layer_number, device=None):
        super().__init__(config, layer_number, device=device)
        self.core_attention = CORE_ATTENTION_CLASSES[config._attn_implementation](config, self.layer_number)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, past_key_values=None, **kwargs):
        # hidden_states: [b, sq, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        if past_key_values is not None:
            key_layer, value_layer = past_key_values.update(key_layer, value_layer, self.layer_idx)

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output


class GLMBlock(core.GLMBlock):
    attention_class = SelfAttention


class GLMTransformer(core.GLMTransformer):
    block_class = GLMBlock
