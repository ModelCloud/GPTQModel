# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# For licensing see accompanying LICENSE-Apple file.
# Package-local construction and explicit native CLIP configuration: ModelCloud.ai, 2026.

import math
import re
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


class MCi(nn.Module):
    def __init__(self, config):
        super().__init__()
        # timm primitives are optional and imported only for the MobileCLIP tower.
        from .fastvit import FastViT, GlobalPool2D, LayerNormChannel, RepCPE

        values = {
            "layers": [2, 12, 24, 4, 2], "embed_dims": [96, 192, 384, 768, 1536],
            "mlp_ratios": [4, 4, 4, 4, 4], "projection_dim": 768,
        }
        values.update(config)
        dims, layers = values["embed_dims"], values["layers"]
        if len(dims) != 5 or len(layers) != 5 or any(n <= 0 for n in layers):
            raise ValueError("MobileCLIP requires five nonempty FastViTHD stages")
        self.model = FastViT(
            layers, token_mixers=("repmixer", "repmixer", "repmixer", "attention", "attention"),
            embed_dims=dims, mlp_ratios=values["mlp_ratios"], downsamples=[True] * 5,
            pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7)),
                      partial(RepCPE, spatial_shape=(7, 7))],
            norm_layer=LayerNormChannel, stem_scale_branch=False, inference_mode=True,
        )
        self.model.head = GlobalPool2D(dims[-1] * 2, values["projection_dim"])

    def forward(self, images):
        return self.model(images, return_image_embeddings=True)["image_embeddings"]


class MobileCLIPVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        match = re.fullmatch(r"mobileclip_l_(\d+)", config.mm_vision_tower)
        if match is None:
            raise ValueError(f"Unsupported MobileCLIP tower: {config.mm_vision_tower}")
        self.input_image_size = int(match[1])
        if self.input_image_size <= 0 or self.input_image_size % 64:
            raise ValueError("MobileCLIP image size must be a positive multiple of 64")
        self.vision_tower = MCi(config.vision_config or {})
        self.hidden_size = self.vision_tower.model.conv_exp.out_channels
        self.num_patches_per_side = self.input_image_size // 64
        self.num_patches = self.num_patches_per_side ** 2
        self.image_processor = CLIPImageProcessor(
            size={"shortest_edge": self.input_image_size},
            crop_size={"height": self.input_image_size, "width": self.input_image_size},
            image_mean=[0.0] * 3, image_std=[1.0] * 3,
        )
        self.requires_grad_(getattr(config, "unfreeze_mm_vision_tower", False))

    def forward(self, images):
        parameter = next(self.parameters())
        features = self.vision_tower(images.to(device=parameter.device, dtype=parameter.dtype))
        return features.flatten(2).transpose(1, 2)


class CLIPVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.vision_config is None:
            sizes = {
                "openai/clip-vit-large-patch14": (224, 14, 1024, 4096, 24, 16),
                "openai/clip-vit-large-patch14-336": (336, 14, 1024, 4096, 24, 16),
                "openai/clip-vit-base-patch16": (224, 16, 768, 3072, 12, 12),
                "openai/clip-vit-base-patch32": (224, 32, 768, 3072, 12, 12),
            }
            if config.mm_vision_tower not in sizes:
                raise ValueError("Unknown CLIP tower requires an explicit vision_config; no remote config discovery")
            size, patch, hidden, intermediate, layers, heads = sizes[config.mm_vision_tower]
            values = {"image_size": size, "patch_size": patch, "hidden_size": hidden,
                          "intermediate_size": intermediate, "num_hidden_layers": layers, "num_attention_heads": heads}
        else:
            values = config.vision_config
        self.vision_tower = CLIPVisionModel(CLIPVisionConfig(**values))
        self.select_layer = config.mm_vision_select_layer
        self.select_feature = config.mm_vision_select_feature
        if self.select_feature not in ("patch", "cls_patch"):
            raise ValueError("CLIP feature selection must be patch or cls_patch")
        self.input_image_size = self.vision_tower.config.image_size
        self.hidden_size = self.vision_tower.config.hidden_size
        self.num_patches_per_side = self.input_image_size // self.vision_tower.config.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.image_processor = CLIPImageProcessor(
            size={"shortest_edge": self.input_image_size},
            crop_size={"height": self.input_image_size, "width": self.input_image_size},
        )
        self.requires_grad_(getattr(config, "unfreeze_mm_vision_tower", False))

    def forward(self, images):
        parameter = next(self.parameters())
        features = self.vision_tower(images.to(device=parameter.device, dtype=parameter.dtype),
                                     output_hidden_states=True).hidden_states[self.select_layer]
        return features[:, 1:] if self.select_feature == "patch" else features


class CLIPVisionTowerS2(CLIPVisionTower):
    """S2 chessboard multiscale inference, without runtime s2wrapper imports.

    Adapted from Baifeng Shi's MIT-licensed scaling_on_scales; see NOTICE.
    """

    def __init__(self, config):
        super().__init__(config)
        self.scales = sorted(int(value) for value in getattr(config, "s2_scales", "336,672,1008").split(","))
        if not self.scales or self.scales[0] != self.input_image_size:
            raise ValueError("The smallest S2 scale must match the CLIP image size")
        if any(value % self.scales[0] for value in self.scales):
            raise ValueError("S2 scales must be positive whole multiples of the CLIP image size")
        self.hidden_size *= len(self.scales)
        self.image_processor.size["shortest_edge"] = self.scales[-1]
        self.image_processor.crop_size = {"height": self.scales[-1], "width": self.scales[-1]}

    def forward(self, images):
        if images.ndim != 4 or images.shape[-2] != images.shape[-1]:
            raise ValueError("S2 requires square NCHW images")
        outputs = []
        batch, channels = images.shape[:2]
        for size in self.scales:
            splits = math.ceil(size / self.scales[0])
            scaled = F.interpolate(images.float(), size=size, mode="bicubic").to(images.dtype)
            crops = scaled.reshape(batch, channels, splits, size // splits, splits, size // splits)
            crops = crops.permute(2, 4, 0, 1, 3, 5).reshape(splits * splits * batch, channels,
                                                         size // splits, size // splits)
            features = super().forward(crops)
            side = math.isqrt(features.shape[1])
            if side * side != features.shape[1]:
                raise ValueError("S2 requires patch-only square-grid CLIP features")
            features = features.transpose(1, 2).reshape(splits, splits, batch, -1, side, side)
            features = features.permute(2, 3, 0, 4, 1, 5).flatten(2, 3).flatten(3, 4)
            outputs.append(F.interpolate(features.float(), size=self.num_patches_per_side,
                                          mode="area").to(features.dtype))
        return torch.cat(outputs, 1).flatten(2).transpose(1, 2)


def build_vision_tower(config):
    if config.mm_vision_tower.startswith("mobileclip"):
        tower = MobileCLIPVisionTower(config)
    else:
        tower = (CLIPVisionTowerS2 if getattr(config, "s2", False) else CLIPVisionTower)(config)
    if tower.hidden_size != config.mm_hidden_size:
        raise ValueError("mm_hidden_size does not match the actual vision tower output")
    return tower
