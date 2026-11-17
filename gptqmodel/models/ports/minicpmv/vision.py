# SPDX-FileCopyrightText: 2024 HuggingFace Inc. team
# SPDX-FileCopyrightText: 2024 OpenBMB
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NaViT SigLIP from the pinned MiniCPM-V implementation (see PROVENANCE.json).

Flattened patch strips retain their original 2D bucketed positions via tgt_sizes.
The native 4.6 vision tower uses a different layout and is deliberately not used.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPooling


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_patches_per_side = config.image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(config.num_channels, config.hidden_size,
                                         config.patch_size, stride=config.patch_size)
        self.position_embedding = nn.Embedding(self.num_patches_per_side ** 2, config.hidden_size)

    def forward(self, pixels, mask, tgt_sizes=None):
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        positions = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        # Match upstream FP32 bucket construction, including right=True edges.
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        for i, keep in enumerate(mask):
            h, w = (map(int, tgt_sizes[i]) if tgt_sizes is not None
                    else (int(keep[:, 0].sum()), int(keep[0].sum())))
            if h <= 0 or w <= 0 or h * w != int(keep.sum()):
                raise ValueError("tgt_sizes must match the valid patch count")
            bh = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / h), boundaries, right=True)
            bw = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / w), boundaries, right=True)
            positions[i, keep.flatten()] = (bh[:, None] * self.num_patches_per_side + bw).flatten().to(x.device)
        return x + self.position_embedding(positions)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.dropout = config.attention_dropout
        self.config = config
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, nn.Linear(config.hidden_size, config.hidden_size))

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = [getattr(self, name)(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
                   for name in ("q_proj", "k_proj", "v_proj")]
        if self.config._attn_implementation == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-1, -2)) * self.head_dim ** -0.5
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = F.dropout(scores.float().softmax(-1).to(q.dtype), p=self.dropout, training=self.training)
            out = weights @ v
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def forward(self, x, mask):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config, drop_last=False):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.embeddings = VisionEmbeddings(config)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([VisionLayer(config)
                                            for _ in range(config.num_hidden_layers - int(drop_last))])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values, patch_attention_mask=None, tgt_sizes=None):
        b, _, h, w = pixel_values.shape
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones((b, h // self.patch_size, w // self.patch_size),
                                              dtype=torch.bool, device=pixel_values.device)
        x = self.embeddings(pixel_values, patch_attention_mask, tgt_sizes)
        mask = patch_attention_mask.flatten(1)[:, None, None, :]
        for layer in self.encoder.layers:
            x = layer(x, mask)
        return BaseModelOutputWithPooling(last_hidden_state=self.post_layernorm(x))
