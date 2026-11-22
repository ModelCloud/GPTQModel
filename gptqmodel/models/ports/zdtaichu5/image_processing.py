# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, ImageType, get_image_type, make_list_of_images
from transformers.utils import TensorType


class ZDTaichu5_0_ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        selected = {key: kwargs[key] for key in (
            "cache_dir", "revision", "token", "local_files_only", "subfolder", "force_download",
        ) if key in kwargs}
        payload, _ = cls.get_image_processor_dict(pretrained_model_name_or_path, **selected)
        for key in ("image_size", "patch_size", "downsample_ratio", "max_num_tiles", "use_thumbnail"):
            if key in kwargs:
                payload[key] = kwargs[key]
        return cls(**payload)

    def __init__(self, image_size=512, max_num_tiles=12, use_thumbnail=True, norm_mean=None, norm_std=None, do_rescale=True, patch_size=16, downsample_ratio=0.5, merge_size=1, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.norm_mean = norm_mean or [0.485, 0.456, 0.406]
        self.norm_std = norm_std or [0.229, 0.224, 0.225]
        self.do_rescale = do_rescale
        self.merge_size = merge_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        if image_size <= 0 or patch_size <= 0 or max_num_tiles < 1:
            raise ValueError("Image size, patch size and tile budget must be positive")
        if len(self.norm_mean) != 3 or len(self.norm_std) != 3 or any(x <= 0 for x in self.norm_std):
            raise ValueError("Image normalization requires three channels and positive standard deviations")

    def _process_image(
        self,
        image: ImageInput,
        **kwargs,
    ) -> torch.Tensor:
        image_type = get_image_type(image)
        if image_type == ImageType.PIL:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Keep PIL input through tiling so resize order matches vLLM.
        return image

    def preprocess(
        self,
        images: List[torch.Tensor],
        image_size: int = None,
        max_num_tiles: int = None,
        use_thumbnail: bool = None,
        do_rescale: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        image_size = image_size if image_size is not None else self.image_size
        max_num_tiles = max_num_tiles if max_num_tiles is not None else self.max_num_tiles
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale

        images = [self._process_image(image) for image in make_list_of_images(images)]

        all_patches = []
        num_patches = []
        image_grid_thw = []
        for image in images:
            patches, tile_rows, tile_cols = dynamic_preprocess(image, image_size, max_num_tiles, use_thumbnail)
            all_patches.extend(patches)
            num_patches.append(len(patches))
            image_grid_thw.append([1, tile_rows, tile_cols])

        # vLLM converts each already-cropped PIL tile with ToTensor.
        pixel_values = torch.stack([T.ToTensor()(patch) for patch in all_patches], dim=0)
        norm_mean = torch.Tensor(self.norm_mean).view(1, 3, 1, 1)
        norm_std = torch.Tensor(self.norm_std).view(1, 3, 1, 1)
        pixel_values = (pixel_values - norm_mean) / norm_std
        pixel_values = pixel_values.to(torch.bfloat16)
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "num_patches": num_patches,
                "image_grid_thw": image_grid_thw,
            },
            tensor_type=return_tensors,
        )


def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def select_tile_grid(
    *,
    orig_width: int,
    orig_height: int,
    image_size: int,
    min_num_tiles: int,
    max_num_tiles: int,
) -> Tuple[int, int, int, int]:
    """
    Choose (rw, rh) tile grid with small-image and aspect-sanity guards.

    Returns (num_grid_blocks, target_width, target_height, effective_max_tiles).
    num_grid_blocks = rw * rh (does NOT include the optional thumbnail).

    Guards:
      1. Area cap: don't create more tiles than the source has pixels for.
      2. Aspect-sanity: drop candidates whose ratio differs from source by >3x.
    """
    # ── Guard 1: area cap ─────────────────────────────────────────────────
    src_pixels = orig_width * orig_height
    tile_pixels = image_size * image_size
    area_max_tiles = max(1, math.ceil(src_pixels / tile_pixels))
    effective_max = min(max_num_tiles, area_max_tiles)
    effective_max = max(effective_max, min_num_tiles)

    target_ratios = get_internvl_target_ratios(min_num_tiles, effective_max)

    # ── Guard 2: aspect-sanity ────────────────────────────────────────────
    src_ar = orig_width / orig_height
    filtered = [
        (rw, rh) for (rw, rh) in target_ratios
        if (1.0 / 3.0) <= (rw / rh) / src_ar <= 3.0
    ]
    # Fall back to unfiltered set for extreme panoramas / long strips where
    # no candidate is within 3x — better to pick *something* than error.
    if filtered:
        target_ratios = filtered

    # ── Pick best ratio ───────────────────────────────────────────────────
    rw, rh = find_closest_aspect_ratio(
        src_ar, target_ratios,
        width=orig_width, height=orig_height, image_size=image_size,
    )

    target_width = image_size * rw
    target_height = image_size * rh
    num_grid_blocks = rw * rh

    return num_grid_blocks, target_width, target_height, effective_max


def count_tiles(
    *,
    orig_width: int,
    orig_height: int,
    image_size: int,
    min_num_tiles: int,
    max_num_tiles: int,
    use_thumbnail: bool,
) -> int:
    """
    Total number of tiles (grid blocks + optional thumbnail) for this image.
    This MUST match what the actual tiling produces, or prompt expansion and
    embedding count will diverge.
    """
    n_grid, _, _, _ = select_tile_grid(
        orig_width=orig_width, orig_height=orig_height,
        image_size=image_size,
        min_num_tiles=min_num_tiles, max_num_tiles=max_num_tiles,
    )
    if use_thumbnail and n_grid != 1:
        return n_grid + 1
    return n_grid

def calculate_targets(
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
) -> tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    return blocks, target_width, target_height

def dynamic_preprocess(image, image_size=512, max_num_tiles=12, use_thumbnail=True, min_num_tiles=1):
    """Split a PIL image using vLLM's resize/crop order."""
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    orig_width, orig_height = image.size

    n_grid, target_width, target_height, _ = select_tile_grid(
        orig_width=orig_width,
        orig_height=orig_height,
        image_size=image_size,
        min_num_tiles=min_num_tiles,
        max_num_tiles=max_num_tiles,
    )

    # Tile grid dimensions (rows × cols of the InternVL tiling)
    tile_rows = target_height // image_size
    tile_cols = target_width // image_size

    resized_img = image.resize((target_width, target_height), Image.BICUBIC)
    cols = target_width // image_size
    patches = []
    for i in range(n_grid):
        col = i % cols
        row = i // cols
        patches.append(
            resized_img.crop(
                (col * image_size, row * image_size,
                 (col + 1) * image_size, (row + 1) * image_size)
            )
        )
    assert len(patches) == n_grid

    if use_thumbnail and n_grid != 1:
        thumbnail = image.resize((image_size, image_size), Image.BICUBIC)
        patches.append(thumbnail)

    #print(orig_height, orig_width, target_width, target_height, len(patches))

    return patches, tile_rows, tile_cols
