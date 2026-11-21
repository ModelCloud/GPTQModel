# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Vision backbone of Inkling 2 Mini.

A ViT that consumes one raster block of ``merge x merge`` patches per language
model image token, attends inside each image with axial 2-D rotary positions,
and merges every ``merge x merge`` block of patch features into one language
model embedding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel

from .configuration import Inkling2LoopVisionConfig


def _rotate_pairs(states: torch.Tensor) -> torch.Tensor:
    """Map every interleaved pair ``(a, b)`` to ``(-b, a)``."""

    pairs = states.view(*states.shape[:-1], -1, 2)
    first, second = pairs.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2)


def _apply_axial_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = query.dtype, key.dtype
    query, key = query.float(), key.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    query = query * cos + _rotate_pairs(query) * sin
    key = key * cos + _rotate_pairs(key) * sin
    return query.to(q_dtype), key.to(k_dtype)


class Inkling2LoopVisionRotary(nn.Module):
    """Axial rotary positions: half the head rotates on the row index, half on the column."""

    def __init__(self, config: Inkling2LoopVisionConfig):
        super().__init__()
        spatial_dim = config.head_dim // 2
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, spatial_dim, 2).float() / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (tokens, 2) holding the (row, column) of every token
        angles = position_ids[..., None].float() * self.inv_freq.to(position_ids.device)
        angles = torch.cat([angles[:, 1], angles[:, 0]], dim=-1).repeat_interleave(2, dim=-1)
        return angles.cos(), angles.sin()


class Inkling2LoopVisionAttention(nn.Module):
    def __init__(self, config: Inkling2LoopVisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        shape = (tokens, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(0, 1).unsqueeze(0)
        key = self.k_proj(hidden_states).view(shape).transpose(0, 1).unsqueeze(0)
        value = self.v_proj(hidden_states).view(shape).transpose(0, 1).unsqueeze(0)
        query, key = _apply_axial_rotary(query, key, *position_embeddings)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        pieces = []
        for q, k, v in zip(
            torch.split(query, lengths, dim=2),
            torch.split(key, lengths, dim=2),
            torch.split(value, lengths, dim=2),
        ):
            weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            weights = F.softmax(weights.float(), dim=-1).to(q.dtype)
            weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
            pieces.append(torch.matmul(weights, v))
        attended = torch.cat(pieces, dim=2)
        return self.out_proj(attended.reshape(tokens, -1).contiguous())


class Inkling2LoopVisionMLP(nn.Module):
    def __init__(self, config: Inkling2LoopVisionConfig):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(hidden_states)))


class Inkling2LoopVisionLayer(nn.Module):
    def __init__(self, config: Inkling2LoopVisionConfig):
        super().__init__()
        self.self_attn = Inkling2LoopVisionAttention(config)
        self.mlp = Inkling2LoopVisionMLP(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lambda_1 = nn.Parameter(config.layer_scale_init_value * torch.ones(config.hidden_size))
        self.lambda_2 = nn.Parameter(config.layer_scale_init_value * torch.ones(config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        attended = self.self_attn(
            self.layernorm_before(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + self.lambda_1 * attended
        residual = hidden_states
        return residual + self.lambda_2 * self.mlp(self.layernorm_after(hidden_states))


def _patch_positions(
    grid_h: int, grid_w: int, merge: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, column) of every patch, in the merge-block order the patches arrive in."""

    block_rows, block_cols = grid_h // merge, grid_w // merge
    rows = torch.arange(block_rows).repeat_interleave(block_cols * merge * merge)
    cols = torch.arange(block_cols).repeat_interleave(merge * merge).repeat(block_rows)
    inner_rows = torch.arange(merge).repeat_interleave(merge).repeat(block_rows * block_cols)
    inner_cols = torch.arange(merge).repeat(block_rows * block_cols * merge)
    return rows * merge + inner_rows, cols * merge + inner_cols


def _bilinear_taps(
    grid_h: int, grid_w: int, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather indices/weights that resample a ``side x side`` table to ``(grid_h, grid_w)``."""

    def taps(target: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # align_corners=False mapping from a destination cell to the source grid
        source = (torch.arange(target, dtype=torch.float32) + 0.5) * (side / target) - 0.5
        source = source.clamp(min=0.0, max=side - 1)
        low = source.floor().to(torch.long)
        high = (low + 1).clamp(max=side - 1)
        return low, high, source - low.float()

    row_low, row_high, row_frac = taps(grid_h)
    col_low, col_high, col_frac = taps(grid_w)
    row_low = row_low.repeat_interleave(grid_w)
    row_high = row_high.repeat_interleave(grid_w)
    col_low = col_low.repeat(grid_h)
    col_high = col_high.repeat(grid_h)
    row_frac = row_frac.repeat_interleave(grid_w)
    col_frac = col_frac.repeat(grid_h)
    indices = torch.stack(
        [
            row_low * side + col_low,
            row_low * side + col_high,
            row_high * side + col_low,
            row_high * side + col_high,
        ],
        dim=-1,
    )
    weights = torch.stack(
        [
            (1 - row_frac) * (1 - col_frac),
            (1 - row_frac) * col_frac,
            row_frac * (1 - col_frac),
            row_frac * col_frac,
        ],
        dim=-1,
    )
    return indices, weights


class Inkling2LoopVisionEmbeddings(nn.Module):
    def __init__(self, config: Inkling2LoopVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.merge_size = config.spatial_merge_size
        self.grid_size = config.posemb_grid_size
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size**2, config.hidden_size, bias=False
        )
        self.position_embedding = nn.Embedding(config.posemb_grid_size**2, config.hidden_size)
        self.class_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.cls_position_embedding = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(
        self,
        pixel_values: torch.Tensor,
        grids: list[tuple[int, int]],
        patch_index: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        merge, patch = self.merge_size, self.patch_size
        channels = pixel_values.shape[-1]
        blocks = pixel_values.reshape(-1, merge, patch, merge, patch, channels).transpose(2, 3)
        patch_embeddings = self.patch_embedding(
            blocks.reshape(-1, self.patch_embedding.in_features)
        )

        table = self.position_embedding.weight
        position_embeddings = []
        for grid_h, grid_w in grids:
            indices, weights = _bilinear_taps(grid_h, grid_w, self.grid_size)
            indices = indices.to(pixel_values.device)
            weights = weights.to(pixel_values.device)
            position_embeddings.append((table[indices] * weights[:, :, None]).sum(1))
        patch_embeddings = patch_embeddings + torch.cat(position_embeddings).to(
            patch_embeddings.dtype
        )

        class_token = (self.class_embedding + self.cls_position_embedding).to(
            patch_embeddings.dtype
        )
        hidden_states = class_token.expand(sequence_length, -1).contiguous()
        hidden_states = hidden_states.index_copy(0, patch_index, patch_embeddings)
        return hidden_states


class Inkling2LoopVisionMerger(nn.Module):
    def __init__(self, config: Inkling2LoopVisionConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.spatial_merge_size**2 * config.hidden_size,
            config.projection_hidden_dim,
            bias=config.projector_bias,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear_2 = nn.Linear(
            config.projection_hidden_dim, config.text_hidden_size, bias=config.projector_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_fn(self.linear_1(hidden_states)))


class Inkling2LoopVisionModel(PreTrainedModel):
    """Patch encoder that turns one image token's patch block into one embedding."""

    config_class = Inkling2LoopVisionConfig
    config: Inkling2LoopVisionConfig
    base_model_prefix = "vision_tower"
    main_input_name = "pixel_values"
    _no_split_modules = ["Inkling2LoopVisionLayer"]

    def __init__(self, config: Inkling2LoopVisionConfig):
        super().__init__(config)
        self.embeddings = Inkling2LoopVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary = Inkling2LoopVisionRotary(config)
        self.layers = nn.ModuleList(
            [Inkling2LoopVisionLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.merger = Inkling2LoopVisionMerger(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        merge = self.config.spatial_merge_size
        device = pixel_values.device
        image_grid_thw = image_grid_thw.to(torch.long)
        patch_grids = image_grid_thw[:, 1:] * merge
        grids = [(int(h), int(w)) for h, w in patch_grids.tolist()]
        patches_per_image = torch.tensor(
            [h * w for h, w in grids], dtype=torch.long, device=device
        )
        total_patches = int(patches_per_image.sum())
        total_tokens = total_patches + len(grids)

        if cu_seqlens is None:
            cu_seqlens = F.pad((patches_per_image + 1).cumsum(0), (1, 0)).to(torch.int32)
        # one class token in front of every image, patches follow it
        image_ids = torch.arange(1, len(grids) + 1, device=device)
        patch_index = torch.arange(total_patches, device=device) + torch.repeat_interleave(
            image_ids, patches_per_image
        )

        pixel_values = pixel_values.to(self.embeddings.patch_embedding.weight.dtype)
        hidden_states = self.embeddings(pixel_values, grids, patch_index, total_tokens)
        hidden_states = self.pre_layernorm(hidden_states)

        rows = torch.cat(
            [
                _patch_positions(h, w, merge)[0].to(device)
                for h, w in grids
            ]
        )
        cols = torch.cat(
            [
                _patch_positions(h, w, merge)[1].to(device)
                for h, w in grids
            ]
        )
        # patches are 1-indexed so the class tokens can sit at (0, 0)
        position_ids = torch.zeros(total_tokens, 2, dtype=torch.long, device=device)
        position_ids = position_ids.index_copy(
            0, patch_index, torch.stack([rows, cols], dim=-1) + 1
        )
        position_embeddings = self.rotary(position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
            )

        patch_states = hidden_states[patch_index.to(hidden_states.device)]
        pooled = self.merger(
            patch_states.reshape(-1, merge**2 * hidden_states.shape[-1])
        )
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooled)


__all__ = ["Inkling2LoopVisionModel"]
