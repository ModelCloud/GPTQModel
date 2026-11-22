# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""MoonViT-style vision tower used by LocateAnything (SPEC section 3)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from .configuration_locateanything import MoonViTConfig


_ROPE_THETA = 10000.0
_ROPE_MAX_GRID = 512


class MoonViTPatchEmbed(nn.Module):
    """Per-patch linear projection plus a learned, resizable absolute position table."""

    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.pos_emb = MoonViTPositionTable(config)

    def forward(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.proj.weight.dtype)
        tokens = self.proj(pixel_values).reshape(pixel_values.shape[0], -1)
        return self.pos_emb(tokens, grid_hws)


class MoonViTPositionTable(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.height = config.init_pos_emb_height
        self.width = config.init_pos_emb_width
        self.weight = nn.Parameter(torch.empty(self.height, self.width, config.hidden_size))

    def table_for(self, h: int, w: int) -> torch.Tensor:
        if (h, w) == (self.height, self.width):
            return self.weight.reshape(h * w, -1)
        grid = self.weight.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0).reshape(h * w, -1)

    def forward(self, tokens: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        tables = [self.table_for(int(h), int(w)) for h, w in grid_hws.tolist()]
        return tokens + torch.cat(tables, dim=0).to(tokens.dtype)


def rope_2d_frequencies(grid_hws: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Complex rotations [N_patches, head_dim/2]; even slots use the column angle, odd the row angle."""
    if head_dim % 4 != 0:
        raise ValueError("MoonViT 2D rotary embedding requires head_dim divisible by 4")
    device = grid_hws.device
    quarter = head_dim // 4
    inv_freq = 1.0 / (_ROPE_THETA ** (torch.arange(quarter, device=device, dtype=torch.float32) * 4 / head_dim))
    rotations = []
    for h, w in grid_hws.tolist():
        assert 1 <= h <= _ROPE_MAX_GRID and 1 <= w <= _ROPE_MAX_GRID, f"MoonViT grid {h}x{w} exceeds 512x512"
        rows = torch.arange(h, device=device, dtype=torch.float32).repeat_interleave(w)
        cols = torch.arange(w, device=device, dtype=torch.float32).repeat(h)
        col_angles = torch.outer(cols, inv_freq)
        row_angles = torch.outer(rows, inv_freq)
        angles = torch.stack((col_angles, row_angles), dim=-1).reshape(h * w, 2 * quarter)
        rotations.append(torch.polar(torch.ones_like(angles), angles))
    return torch.cat(rotations, dim=0)


def apply_rope_2d(x: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """x: [N, heads, head_dim]; rotations: [N, head_dim/2] complex."""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(pairs * rotations.unsqueeze(1)).flatten(-2)
    return rotated.to(x.dtype)


def image_visibility(grid_hws: torch.Tensor, device) -> torch.Tensor:
    image_ids = torch.cat(
        [torch.full((int(h) * int(w),), i, device=device) for i, (h, w) in enumerate(grid_hws.tolist())]
    )
    return image_ids[:, None] == image_ids[None, :]


class MoonViTMLP(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.fc0 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc1 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class MoonViTBlock(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.norm0 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp = MoonViTMLP(config)

    def attention(self, x: torch.Tensor, rotations: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        qkv = self.wqkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q = apply_rope_2d(qkv[:, 0], rotations).transpose(0, 1)
        k = apply_rope_2d(qkv[:, 1], rotations).transpose(0, 1)
        v = qkv[:, 2].transpose(0, 1)
        if (getattr(self.config, "_attn_implementation", None) or "sdpa") == "eager":
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~visible, float("-inf"))
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = weights @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=visible)
        return self.wo(out.transpose(0, 1).reshape(n, -1))

    def forward(self, x: torch.Tensor, rotations: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm0(x), rotations, visible)
        return x + self.mlp(self.norm1(x))


class MoonViTEncoder(nn.Module):
    def __init__(self, config: MoonViTConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.blocks = nn.ModuleList([MoonViTBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        rotations = rope_2d_frequencies(grid_hws, self.head_dim)
        visible = image_visibility(grid_hws, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, rotations, visible)
        return self.final_layernorm(hidden_states)


def merge_patches(hidden_states: torch.Tensor, grid_hws: torch.Tensor, kernel) -> list[torch.Tensor]:
    kh, kw = int(kernel[0]), int(kernel[1])
    width = hidden_states.shape[-1]
    merged = []
    offset = 0
    for h, w in grid_hws.tolist():
        if h % kh or w % kw:
            raise ValueError(f"MoonViT grid {h}x{w} is not divisible by merge kernel {kh}x{kw}")
        tokens = hidden_states[offset : offset + h * w]
        offset += h * w
        tokens = tokens.reshape(h // kh, kh, w // kw, kw, width).permute(0, 2, 1, 3, 4)
        merged.append(tokens.reshape((h // kh) * (w // kw), kh * kw * width))
    return merged


class MoonViTPreTrainedModel(PreTrainedModel):
    config_class = MoonViTConfig
    config: MoonViTConfig
    base_model_prefix = "vision_model"
    main_input_name = "pixel_values"
    _supports_sdpa = True
    _supports_flash_attn = False
    _no_split_modules = ["MoonViTBlock"]

    def __init__(self, config: MoonViTConfig):
        if config._attn_implementation == "flash_attention_2":
            config._attn_implementation = "sdpa"
        super().__init__(config)
        self.merge_kernel_size = list(config.merge_kernel_size)
        self.patch_embed = MoonViTPatchEmbed(config)
        self.encoder = MoonViTEncoder(config)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> list[torch.Tensor]:
        grid_hws = torch.as_tensor(grid_hws)
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        return merge_patches(hidden_states, grid_hws, self.merge_kernel_size)


__all__ = ["MoonViTPreTrainedModel", "merge_patches", "rope_2d_frequencies", "apply_rope_2d"]
