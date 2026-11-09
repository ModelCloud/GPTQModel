# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Dense, independent attention arithmetic with explicit TP reduction rounding."""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def oracle(model, ids, shards=2, trace=None):
    dtype = model.lm_head.weight.dtype
    positions = torch.arange(ids.shape[1], device=ids.device)

    def norm(x, layer):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + layer.eps)).to(
            dtype
        ) * layer.weight

    def row_linear(x, weight):
        # Each rank stores its GEMM output in activation dtype before all-reduce.
        partials = [F.linear(a, b) for a, b in zip(x.chunk(shards, -1), weight.chunk(shards, -1))]
        return torch.stack(partials).sum(0, dtype=dtype)

    x = F.embedding(ids, model.model.embed_tokens.weight)
    if model.config.embed_scale is not None:
        x = x * model.config.embed_scale
    seed = x
    batch, length = ids.shape
    for index, layer in enumerate(model.model.layers):
        att = layer.self_attn
        h = norm(x, layer.input_layernorm)
        q = norm(
            F.linear(h, att.q_proj.weight).reshape(batch, length, att.num_heads, att.head_dim), att.q_norm
        ).transpose(1, 2)
        k = norm(
            F.linear(h, att.k_proj.weight).reshape(batch, length, att.num_kv_heads, att.head_dim), att.k_norm
        ).transpose(1, 2)
        value_input = h
        if model.config.use_valres:
            blend = att.value_res_lambda.float().sigmoid().to(dtype)
            value_input = (seed.float() + blend.float() * (h.float() - seed.float())).to(dtype)
        v = (
            F.linear(value_input, att.v_proj.weight)
            .reshape(batch, length, att.num_kv_heads, att.head_dim)
            .transpose(1, 2)
        )
        if att.layer_type == "sliding_attention":
            inv = model.config.rope_theta ** (
                -torch.arange(0, att.head_dim, 2, device=ids.device).float() / att.head_dim
            )
            angles = positions[:, None] * inv[None, :]
            cos = torch.cat([angles.cos(), angles.cos()], -1).to(dtype).float()[None, None]
            sin = torch.cat([angles.sin(), angles.sin()], -1).to(dtype).float()[None, None]

            def rotate(z):
                a, b = z.float().chunk(2, -1)
                return (z.float() * cos + torch.cat([-b, a], -1) * sin).to(dtype)

            q, k = rotate(q), rotate(k)
        k = k.repeat_interleave(att.num_heads // att.num_kv_heads, dim=1)
        v = v.repeat_interleave(att.num_heads // att.num_kv_heads, dim=1)
        scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(att.head_dim)
        allowed = positions[:, None] >= positions[None, :]
        if att.layer_type == "sliding_attention":
            allowed &= positions[:, None] - positions[None, :] < model.config.sliding_window
        probability = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        values = (probability @ v.float()).to(dtype).transpose(1, 2).reshape(batch, length, -1)
        if model.config.use_gated_attention:
            values = values * F.linear(h, att.g_proj.weight).float().sigmoid().to(dtype)
        projected = row_linear(values, att.o_proj.weight)
        if trace is not None:
            trace[f"model.layers.{index}.self_attn.o_proj"] = projected
        x = x + norm(projected, layer.post_attention_layernorm)
        h = norm(x, layer.pre_feedforward_layernorm)
        gate, up = F.linear(h, layer.mlp.gate_proj.weight), F.linear(h, layer.mlp.up_proj.weight)
        if model.config.ffn_clamp_threshold:
            clamp = model.config.ffn_clamp_threshold
            gate, up = gate.clamp(max=clamp), up.clamp(min=-clamp, max=clamp)
        projected = row_linear(F.silu(gate) * up, layer.mlp.down_proj.weight)
        if trace is not None:
            trace[f"model.layers.{index}.mlp.down_proj"] = projected
        x = x + norm(projected, layer.post_feedforward_layernorm)
    return F.linear(norm(x, model.model.norm), model.lm_head.weight)
