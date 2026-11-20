# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Vision tower for the published DeepSeek V4.1 checkpoint."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import V41RMSNorm


def _rotate_2d(x, cos, sin):
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype)


class _VisionAttention(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.wqkv = nn.Linear(width, 3 * width, dtype=dtype)
        self.wo = nn.Linear(width, width, dtype=dtype)

    def forward(self, x, cos, sin):
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(0, 1) for t in self.wqkv(x).chunk(3, dim=-1))
        q, k = _rotate_2d(q, cos, sin), _rotate_2d(k, cos, sin)
        output = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        return self.wo(output.squeeze(0).transpose(0, 1).flatten(-2))


class _VisionMLP(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class _VisionBlock(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.norm1 = V41RMSNorm(config.hidden_size, eps, dtype)
        self.norm2 = V41RMSNorm(config.hidden_size, eps, dtype)
        self.attn = _VisionAttention(config, dtype)
        self.mlp = _VisionMLP(config, dtype)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class V41VisionTower(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.rope_width = config.hidden_size // config.num_attention_heads // 2
        if self.rope_width % 2:
            raise ValueError("DeepSeek V4.1 vision head width must be divisible by four")
        self.theta = config.rope_theta
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList(_VisionBlock(config, dtype) for _ in range(config.num_hidden_layers))
        self.norm = V41RMSNorm(config.hidden_size, getattr(config, "rms_norm_eps", 1e-6), dtype)

    def forward(self, patches, height, width):
        weight = self.patch_embed.proj.weight
        x = self.patch_embed.proj(patches.flatten(1).to(device=weight.device, dtype=weight.dtype))
        rows, columns = torch.meshgrid(
            torch.arange(height, device=x.device), torch.arange(width, device=x.device), indexing="ij"
        )
        frequencies = self.theta ** (-torch.arange(0, self.rope_width, 2, device=x.device) / self.rope_width)
        angles = (torch.stack((rows, columns), dim=-1).reshape(-1, 2, 1) * frequencies).flatten(-2)
        cos, sin = angles.cos(), angles.sin()
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class V41Aligner(nn.Module):
    def __init__(self, vision_config, output_size, dtype):
        super().__init__()
        self.ratio = vision_config.downsample_ratio
        self.w1 = nn.Linear(vision_config.hidden_size * self.ratio**2, output_size, dtype=dtype)
        self.w2 = nn.Linear(output_size, output_size, dtype=dtype)

    def forward(self, features, height, width):
        ratio = self.ratio
        image = features.to(self.w1.weight.device).reshape(height, width, -1).permute(2, 0, 1)
        image = F.pad(image, (0, -width % ratio, 0, -height % ratio))
        patches = F.unfold(image.unsqueeze(0), ratio, stride=ratio)[0].T
        return self.w2(F.gelu(self.w1(patches))).unflatten(0, (-1, (width + ratio - 1) // ratio))
