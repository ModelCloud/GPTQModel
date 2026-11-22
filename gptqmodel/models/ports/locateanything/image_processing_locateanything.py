# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""Native-resolution patch image processor for LocateAnything."""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


# The vision tower's 2D rotary table covers at most this many patches per side.
_MAX_GRID_SIDE = 512


class LocateAnythingImageProcessor(BaseImageProcessor):
    """Resize each image to whole merge cells and cut it into ``patch_size`` patches."""

    model_type = "locateanything"
    model_input_names = ["pixel_values", "image_grid_hws"]

    def __init__(
        self,
        patch_size: int = 14,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        in_token_limit: int = 4096,
        merge_kernel_size=(2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.in_token_limit = in_token_limit
        self.merge_kernel_size = list(merge_kernel_size)

    def to_dict(self):
        return {
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "in_token_limit": self.in_token_limit,
            "merge_kernel_size": list(self.merge_kernel_size),
            "patch_size": self.patch_size,
            "image_processor_type": type(self).__name__,
        }

    def _resize_to_cells(self, image: Image.Image, patch_size: int, in_token_limit: int, merge) -> Image.Image:
        width, height = image.size
        tokens = (width // patch_size) * (height // patch_size)
        if tokens > in_token_limit:
            scale = math.sqrt(in_token_limit / tokens)
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
            width, height = image.size
        cell_h = merge[0] * patch_size
        cell_w = merge[1] * patch_size
        target = (math.ceil(width / cell_w) * cell_w, math.ceil(height / cell_h) * cell_h)
        if target != (width, height):
            image = image.resize(target, Image.Resampling.BICUBIC)
        width, height = image.size
        if width // patch_size >= _MAX_GRID_SIDE or height // patch_size >= _MAX_GRID_SIDE:
            raise ValueError("Exceed pos emb")
        return image

    def _to_patches(self, image: Image.Image, patch_size: int, mean, std):
        from torchvision.transforms import functional as F

        tensor = F.to_tensor(image.convert("RGB"))
        tensor = F.normalize(tensor, mean, std)
        channels, height, width = tensor.shape
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = (
            tensor.view(channels, grid_h, patch_size, grid_w, patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, channels, patch_size, patch_size)
        )
        return patches, (grid_h, grid_w)

    def preprocess(self, images, return_tensors=None, **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]
        images = list(images)
        if not images or not all(isinstance(image, Image.Image) for image in images):
            raise ValueError(
                "Invalid image type. LocateAnythingImageProcessor expects a PIL image or a list of PIL images."
            )
        patch_size = kwargs.get("patch_size") or self.patch_size
        in_token_limit = kwargs.get("in_token_limit") or self.in_token_limit
        merge = kwargs.get("merge_kernel_size") or self.merge_kernel_size
        mean = list(kwargs.get("image_mean") or self.image_mean)
        std = list(kwargs.get("image_std") or self.image_std)

        pixel_values, grids = [], []
        for image in images:
            image = self._resize_to_cells(image, patch_size, in_token_limit, merge)
            patches, grid = self._to_patches(image, patch_size, mean, std)
            pixel_values.append(patches)
            grids.append(grid)

        data = {
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_hws": np.array(grids, dtype=np.int64),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


__all__ = ["LocateAnythingImageProcessor"]
