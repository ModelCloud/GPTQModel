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
"""Video processor for Agnes 3.0 Flash: frame sampling and dynamic-resolution patching."""

import math

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType, is_torchvision_available, logging
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


def fit_video_to_grid(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    """Spatial size for a clip: multiples of `factor`, with the frame count
    rounded up to `temporal_factor` and the total voxel count kept inside
    [min_pixels, max_pixels]."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    t = math.ceil(num_frames / temporal_factor) * temporal_factor
    if t * h * w > max_pixels:
        scale = math.sqrt((num_frames * height * width) / max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif t * h * w < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    return h, w


class AgnesVideoProcessorInitKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


class AgnesVideoProcessor(BaseVideoProcessor):
    """Agnes frame sampling and merge-ordered spatiotemporal patching."""
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    fps = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = True
    valid_kwargs = AgnesVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[AgnesVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        return kwargs

    def sample_frames(self, metadata: VideoMetadata, num_frames: int | None = None, fps: int | float | None = None, **kwargs):
        """Frame indices to keep: `fps` frames per second of source video when
        metadata is available, clamped to [min_frames, max_frames]; `num_frames`
        overrides that.  Indices are spread uniformly over the clip."""
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")
        total = metadata.total_num_frames
        fps = fps if fps is not None else self.fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                metadata.fps = 24
                logger.warning_once(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                )
            num_frames = int(total / metadata.fps * fps)
            num_frames = min(max(num_frames, self.min_frames), self.max_frames, total)
        if num_frames is None:
            num_frames = min(max(total, self.min_frames), self.max_frames)
        return np.linspace(0, total - 1, num_frames).round().astype(int)

    def get_number_of_video_patches(self, num_frames, height, width, videos_kwargs=None):
        """Count patches after sampling, for the supplied frame dimensions."""
        options = videos_kwargs or {}
        size = options.get("size") or self.size
        patch = options.get("patch_size", self.patch_size)
        temporal = options.get("temporal_patch_size", self.temporal_patch_size)
        merge = options.get("merge_size", self.merge_size)
        h, w = fit_video_to_grid(
            num_frames, height, width, temporal_factor=temporal, factor=patch * merge,
            min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"],
        )
        return math.ceil(num_frames / temporal) * (h // patch) * (w // patch)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        # 1. resize, batched per input shape
        by_shape, order = group_videos_by_shape(videos)
        resized = {}
        for shape, batch in by_shape.items():
            if do_convert_rgb:
                batch = self.convert_to_rgb(batch)
            n, t, c, h, w = batch.shape
            if do_resize:
                new_h, new_w = fit_video_to_grid(
                    num_frames=t, height=h, width=w, temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size, min_pixels=size.shortest_edge, max_pixels=size.longest_edge,
                )
                batch = self.resize(batch.view(n * t, c, h, w), size=SizeDict(height=new_h, width=new_w), resample=resample)
                batch = batch.view(n, t, c, new_h, new_w)
            resized[shape] = batch
        videos = reorder_videos(resized, order)

        # 2. normalise, pad the frame count to the temporal patch, cut into patches
        by_shape, order = group_videos_by_shape(videos)
        flat = {}
        grids = {}
        for shape, batch in by_shape.items():
            new_h, new_w = get_image_size(batch[0], channel_dim=ChannelDimension.FIRST)
            px = self.rescale_and_normalize(batch, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            t = px.shape[1]
            if pad := -t % temporal_patch_size:
                px = torch.cat((px, px[:, -1:].expand(-1, pad, -1, -1, -1)), dim=1)
            n, gt, c = px.shape[:3]
            gt = gt // temporal_patch_size
            gh, gw = new_h // patch_size, new_w // patch_size
            px = px.view(
                n, gt, temporal_patch_size, c, gh // merge_size, merge_size, patch_size, gw // merge_size, merge_size, patch_size
            )
            px = px.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flat[shape] = px.reshape(n, gt * gh * gw, c * temporal_patch_size * patch_size * patch_size)
            grids[shape] = [[gt, gh, gw]] * n

        pixel_values_videos = torch.cat(reorder_videos(flat, order), dim=0)
        video_grid_thw = torch.tensor(reorder_videos(grids, order))
        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}, tensor_type=return_tensors
        )


__all__ = ["AgnesVideoProcessor"]
