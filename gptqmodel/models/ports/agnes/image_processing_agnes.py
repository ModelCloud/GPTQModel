# Copyright 2026 Agnes AI. All rights reserved.
# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
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
"""Image processor for Agnes 3.0 Flash: dynamic-resolution patching."""

import math
from collections.abc import Iterable

import torch
from torchvision.transforms.v2 import functional as tvF
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import ImageInput, PILImageResampling, SizeDict
from transformers.processing_utils import ImagesKwargs, Unpack
from transformers.utils import TensorType, auto_docstring


class AgnesImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `256 * 256`):
        Lower bound on the pixel count after resizing.
    max_pixels (`int`, *optional*, defaults to `4096 * 4096`):
        Upper bound on the pixel count after resizing.
    patch_size (`int`, *optional*, defaults to 16):
        Spatial patch size of the vision tower.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Temporal patch size of the vision tower (images are duplicated to fill it).
    merge_size (`int`, *optional*, defaults to 2):
        Side of the patch square merged into one language-model token.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int


def fit_to_grid(height: int, width: int, factor: int = 32, min_pixels: int = 256 * 256, max_pixels: int = 4096 * 4096):
    """Pick a (height, width) that is a multiple of `factor` on both sides,
    keeps the pixel count inside [min_pixels, max_pixels] and stays as close
    as possible to the original aspect ratio."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    return h, w


@auto_docstring
class AgnesImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096}
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = AgnesImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[AgnesImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        size = dict(self.size if size is None else size)
        # min_pixels / max_pixels are the older spelling of the two size keys
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)
        kwargs = super()._standardize_kwargs(size=size, **kwargs)
        size = kwargs.get("size", self.size)
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        return kwargs

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[AgnesImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # 1. resize, batched per input shape
        by_shape, order = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized = {}
        for shape, batch in by_shape.items():
            height, width = batch.shape[-2:]
            if do_resize:
                new_h, new_w = fit_to_grid(
                    height, width, factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge, max_pixels=size.longest_edge,
                )
                batch = self.resize(image=batch, size=SizeDict(height=new_h, width=new_w), resample=resample)
            resized[shape] = batch
        images = reorder_images(resized, order)

        # 2. normalise and cut into merge-ordered patches, batched per resized shape
        by_shape, order = group_images_by_shape(images, disable_grouping=disable_grouping)
        flat = {}
        grids = {}
        for shape, batch in by_shape.items():
            new_h, new_w = batch.shape[-2:]
            px = self.rescale_and_normalize(batch, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            n, c = px.shape[:2]
            gh, gw = new_h // patch_size, new_w // patch_size
            px = px.reshape(n, c, gh // merge_size, merge_size, patch_size, gw // merge_size, merge_size, patch_size)
            # -> [n, gh/merge, gw/merge, merge, merge, c, patch, patch]: patches of one
            # merge square end up adjacent in the flattened sequence
            px = px.permute(0, 2, 5, 3, 6, 1, 4, 7)
            px = (
                px.unsqueeze(6)
                .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
                .reshape(n, gh * gw, c * temporal_patch_size * patch_size * patch_size)
            )
            flat[shape] = px
            grids[shape] = [[1, gh, gw]] * n

        pixel_values = torch.cat(reorder_images(flat, order), dim=0)
        image_grid_thw = torch.tensor(reorder_images(grids, order), dtype=torch.long)
        return BatchFeature(data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """Number of vision patches an image of this size produces (used by
        serving engines to lay out placeholders without running the processor)."""
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        new_h, new_w = fit_to_grid(height, width, patch_size * merge_size, min_pixels=min_pixels, max_pixels=max_pixels)
        return (new_h // patch_size) * (new_w // patch_size)


__all__ = ["AgnesImageProcessor"]
