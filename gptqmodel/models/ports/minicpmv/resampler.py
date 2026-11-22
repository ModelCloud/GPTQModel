# Copyright (c) Alibaba Cloud.
# Copyright 2024 OpenBMB.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible 2D/3D perceiver resampling, without shared mutable caches.

Derived sine tables are invocation-local. This avoids the upstream 72000 x hidden
temporal allocation and cache mutation; checkpoint-owned parameters are unchanged.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def spatial_positions(dim, height, width, legacy=False):
    dtype = np.float64 if legacy else np.float32
    grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    omega = 1.0 / 10000 ** (np.arange(dim // 4, dtype=dtype) / (dim / 4.0))
    parts = []
    for axis in grid:
        angles = np.einsum("hw,d->hwd", axis, omega)
        parts.extend((np.sin(angles), np.cos(angles)))
    return torch.from_numpy(np.concatenate(parts, axis=-1).reshape(height * width, dim))


def temporal_position(dim, index):
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float32) / (dim / 2.0))
    angle = np.float32(index) * omega
    return torch.from_numpy(np.concatenate((np.sin(angle), np.cos(angle))))


class Resampler(nn.Module):
    def __init__(self, config, vision_dim):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        heads = dim // 128
        if not heads or dim % heads or dim % 4:
            raise ValueError("MiniCPM-V resampler requires hidden_size divisible by its hidden_size//128 heads and 4")
        self.embed_dim = dim
        self.num_queries = config.query_num
        self.query = nn.Parameter(torch.zeros(config.query_num, dim))
        self.proj = nn.Parameter(torch.randn(dim, dim) * dim ** -0.5)
        self.kv_proj = nn.Linear(vision_dim, dim, bias=False) if vision_dim != dim else nn.Identity()
        self.attn = nn.MultiheadAttention(dim, heads)
        self.ln_q = nn.LayerNorm(dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(dim, eps=1e-6)
        if config.version < 2.5:
            side = math.isqrt(config.query_num)
            if side * side != config.query_num:
                raise ValueError("Original MiniCPM-V resampler requires a square query_num")
            self.pos_embed = nn.Parameter(spatial_positions(dim, side, side, legacy=True).to(self.query), requires_grad=False)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x, tgt_sizes, temporal_ids=None):
        if x.shape[0] != len(tgt_sizes):
            raise ValueError("Resampler target count differs from image count")
        x = self.ln_kv(self.kv_proj(x))
        q = self.ln_q(self.query)
        if self.config.version < 2.5:
            if x.shape[0] != 1:
                raise ValueError("Original MiniCPM-V resamples individual image crops")
            h, w = map(int, tgt_sizes[0])
            if self.config.version == 2.0:
                pos = spatial_positions(self.embed_dim, h, w, legacy=True).to(x)
            else:
                side = math.isqrt(x.shape[1])
                src = math.isqrt(self.num_queries)
                pos = self.pos_embed
                if side != src:
                    pos = F.interpolate(pos.float().reshape(1, src, src, -1).permute(0, 3, 1, 2),
                                        size=(side, side), mode="bicubic", align_corners=False)
                    pos = pos.permute(0, 2, 3, 1).flatten(0, 2).to(x.dtype)
            queries = (q + self.pos_embed.to(q))[:, None]
            out = self.attn(queries, (x + pos.to(x)).transpose(0, 1), x.transpose(0, 1), need_weights=False)[0]
        else:
            positions = [spatial_positions(self.embed_dim, int(h), int(w)).to(x) for h, w in tgt_sizes]
            positions = nn.utils.rnn.pad_sequence(positions, batch_first=True)
            lengths = tgt_sizes.prod(-1).to(x.device)
            mask = torch.arange(x.shape[1], device=x.device)[None] >= lengths[:, None]
            k, v = x + positions, x
            groups = [[int(t) for t in group] for group in temporal_ids] if temporal_ids is not None else None
            flat = [t for group in groups for t in group] if groups is not None else None
            if flat is not None and len(flat) != x.shape[0]:
                raise ValueError("temporal_ids must account for every image crop")
            temporal = flat is not None and self.config.version == 4.5
            if flat is not None and any(t >= 0 for t in flat) and self.config.version != 4.5:
                raise ValueError("Temporal frame pooling requires MiniCPM-V 4.5")
            if temporal:
                if any(t < -1 for t in flat):
                    raise ValueError("Temporal IDs must be nonnegative or -1 for still images")
                tpos = [temporal_position(self.embed_dim, t).to(x) if t >= 0 else torch.zeros_like(x[0, 0])
                        for t in flat]
                k = k + torch.stack(tpos)[:, None]
                out_list, start = [], 0
                for group in groups:
                    stop = start + len(group)
                    if not group:
                        raise ValueError("Temporal groups cannot be empty")
                    # Preserve the two published 4.5 modes, including foreach's
                    # L,F flattening order (batch mode flattens F,L).
                    if self.config.batch_3d_resampler:
                        kg, vg = k[start:stop].flatten(0, 1), v[start:stop].flatten(0, 1)
                    else:
                        kg = k[start:stop].transpose(0, 1).reshape(-1, self.embed_dim)
                        vg = v[start:stop].transpose(0, 1).reshape(-1, self.embed_dim)
                    mg = mask[start:stop].flatten()
                    out_list.append(self.attn(q, kg, vg, key_padding_mask=mg, need_weights=False)[0])
                    start = stop
                out = torch.stack(out_list, dim=1)
            else:
                out = self.attn(q[:, None].expand(-1, x.shape[0], -1), k.transpose(0, 1), v.transpose(0, 1),
                                key_padding_mask=mask, need_weights=False)[0]
        return self.ln_post(out.transpose(0, 1)) @ self.proj
