# SPDX-FileCopyrightText: 2024 HuggingFace Inc. team
# SPDX-FileCopyrightText: 2024 OpenBMB
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NaViT SigLIP embeddings from MiniCPM-V on native Idefics2/SigLIP encoder layers.

Flattened patch strips retain their original 2D bucketed positions via tgt_sizes.
The native 4.6 vision tower uses a different layout and is deliberately not used.
"""

import torch
from torch import nn
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.idefics2.modeling_idefics2 import Idefics2EncoderLayer


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


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config, drop_last=False):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.embeddings = VisionEmbeddings(config)
        self.encoder = nn.Module()
        # Native Idefics2/SigLIP layer: identical keys and attention-interface dispatch.
        self.encoder.layers = nn.ModuleList([Idefics2EncoderLayer(config)
                                            for _ in range(config.num_hidden_layers - int(drop_last))])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values, patch_attention_mask=None, tgt_sizes=None):
        b, _, h, w = pixel_values.shape
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones((b, h // self.patch_size, w // self.patch_size),
                                              dtype=torch.bool, device=pixel_values.device)
        x = self.embeddings(pixel_values, patch_attention_mask, tgt_sizes)
        mask = create_bidirectional_mask(config=self.config, inputs_embeds=x,
                                         attention_mask=patch_attention_mask.flatten(1))
        for layer in self.encoder.layers:
            x = layer(x, mask)
        return BaseModelOutputWithPooling(last_hidden_state=self.post_layernorm(x))
