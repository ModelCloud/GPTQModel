# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Intern-S1 video preprocessing and midpoint sampling on Transformers 5."""

import numpy as np
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from transformers.processing_utils import VideosKwargs
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata


class InternS1VideoProcessorInitKwargs(VideosKwargs, total=False):
    initial_shift: bool | float | int


class InternS1VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    initial_shift = True
    do_sample_frames = False
    valid_kwargs = InternS1VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        initial_shift: bool | float | int | None = None,
        **kwargs,
    ) -> np.ndarray:
        num_frames = num_frames if num_frames is not None else self.num_frames
        fps = fps if fps is not None else self.fps
        total_num_frames = metadata.total_num_frames
        if num_frames is not None and fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive")
        if num_frames is None and fps is not None:
            if metadata.fps is None or metadata.fps <= 0:
                raise ValueError("Sampling fps requires video metadata with a positive fps")
            num_frames = int(total_num_frames / metadata.fps * fps)
        if num_frames is None:
            num_frames = total_num_frames
        if num_frames <= 0 or num_frames > total_num_frames:
            raise ValueError(f"Cannot sample {num_frames} frames from {total_num_frames}")
        initial_shift = initial_shift if initial_shift is not None else self.initial_shift
        if initial_shift is True:
            initial_shift = total_num_frames / num_frames / 2
        return np.arange(initial_shift, total_num_frames, total_num_frames / num_frames).astype(np.int64)


__all__ = ["InternS1VideoProcessor"]
