# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""Prompt templating, media fetching and placeholder expansion for LocateAnything."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import tempfile
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from ....utils.image import ImageFetchError, load_bytes_from_url, load_image_from_url
from ....utils.image import fetch_image as _fetch_local_image
from .image_processing_locateanything import LocateAnythingImageProcessor


logger = logging.getLogger(__name__)

# Upper bound for one remote image or video download.
REMOTE_MEDIA_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_VIDEO_TOTAL_PIXELS = 25913856
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_MAX_FRAMES = 64

_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
_DEFAULT_SYSTEM = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
_PROCESSOR_DICT_KEYS = (
    "chat_template",
    "image_token",
    "video_token",
    "merge_kernel_size",
    "image_placeholder",
    "video_placeholder",
    "image_start_token",
    "image_end_token",
)


# ----------------------------------------------------------------------------------------------
# Media fetching
# ----------------------------------------------------------------------------------------------

def _is_remote(source) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def fetch_image(ele: dict) -> Image.Image:
    source = ele["image"] if "image" in ele else ele["image_url"]
    if _is_remote(source):
        image = load_image_from_url(source, max_bytes=REMOTE_MEDIA_MAX_BYTES)
    else:
        image = _fetch_local_image({"image": source})
    return _to_rgb(image)


def get_video_frame_indices(ele: dict, total_frames: int, video_fps: float):
    """Uniformly sample frame indices at ``ele['fps']`` frames per second."""
    target_fps = ele.get("fps", DEFAULT_SAMPLE_FPS)
    count = max(1, round(total_frames / video_fps * target_fps))
    count = min(count, ele.get("max_frames", DEFAULT_MAX_FRAMES), total_frames)
    if count == total_frames:
        indices = list(range(total_frames))
    else:
        indices = torch.linspace(0, total_frames - 1, count).round().long().tolist()
    sample_fps = count / max(total_frames, 1e-6) * video_fps
    return indices, sample_fps


def _read_video_torchvision(ele: dict):
    import torchvision.io

    video_start = ele.get("video_start", 0.0)
    video, _audio, info = torchvision.io.read_video(
        ele["video"],
        start_pts=video_start,
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    video_fps = info["video_fps"]
    indices, sample_fps = get_video_frame_indices(ele, video.size(0), video_fps)
    timestamps = [video_start + index / video_fps for index in indices]
    return video[indices], sample_fps, timestamps


def _read_video_decord(ele: dict):
    import decord

    reader = decord.VideoReader(ele["video"])
    video_fps = reader.get_avg_fps()
    video_start = ele.get("video_start", 0.0)
    indices, sample_fps = get_video_frame_indices(ele, len(reader), video_fps)
    frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    timestamps = [video_start + index / video_fps for index in indices]
    return frames, sample_fps, timestamps


# Looked up at call time so a reader can be replaced per backend name.
VIDEO_READER_BACKENDS = {
    "torchvision": _read_video_torchvision,
    "decord": _read_video_decord,
}


def _default_video_backend() -> str:
    return "decord" if importlib.util.find_spec("decord") is not None else "torchvision"


def fetch_video(ele: dict, return_video_sample_fps: bool = False, video_reader_backend="torchvision"):
    source = ele["video"]

    if isinstance(source, (list, tuple)):
        extra = {key: value for key, value in ele.items() if key != "video"}
        frames = [fetch_image({**extra, "image": frame}) for frame in source]
        if return_video_sample_fps:
            return frames, ele.get("fps", DEFAULT_SAMPLE_FPS), [-1] * len(frames)
        return frames

    if _is_remote(source):
        data = load_bytes_from_url(source, max_bytes=REMOTE_MEDIA_MAX_BYTES)
        suffix = os.path.splitext(urlparse(source).path)[1]
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            with handle:
                handle.write(data)
            return fetch_video(
                {**ele, "video": handle.name},
                return_video_sample_fps=return_video_sample_fps,
                video_reader_backend=video_reader_backend,
            )
        finally:
            os.unlink(handle.name)

    path = source[len("file://"):] if source.startswith("file://") else source
    request = {**ele, "video": path}
    backend = video_reader_backend or _default_video_backend()
    try:
        video, sample_fps, timestamps = VIDEO_READER_BACKENDS[backend](request)
    except Exception as exc:
        logger.warning("LocateAnything video reader %r failed (%s); retrying with torchvision", backend, exc)
        video, sample_fps, timestamps = VIDEO_READER_BACKENDS["torchvision"](request)

    frames, _channels, height, width = video.shape
    max_pixels = ele.get("video_total_pixels", DEFAULT_VIDEO_TOTAL_PIXELS)
    if frames * height * width > max_pixels:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F

        scale = math.sqrt(max_pixels / (frames * height * width))
        video = F.resize(
            video,
            [int(height * scale), int(width * scale)],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    video = video.float()
    if return_video_sample_fps:
        return video, sample_fps, timestamps
    return video


def extract_vision_info(conversations) -> list[dict]:
    if conversations and isinstance(conversations[0], dict):
        conversations = [conversations]
    infos = []
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for ele in content:
                if not isinstance(ele, dict):
                    continue
                if (
                    "image" in ele
                    or "image_url" in ele
                    or "video" in ele
                    or ele.get("type") in ("image", "image_url", "video")
                ):
                    infos.append(ele)
    return infos


def process_vision_info(conversations, return_video_kwargs: bool = False, video_reader_backend="torchvision"):
    images, videos, fps_list, timestamps_list = [], [], [], []
    for ele in extract_vision_info(conversations):
        if "image" in ele or "image_url" in ele:
            images.append(fetch_image(ele))
        elif "video" in ele:
            video, fps, timestamps = fetch_video(
                ele, return_video_sample_fps=True, video_reader_backend=video_reader_backend
            )
            videos.append(video)
            fps_list.append(fps)
            timestamps_list.append(timestamps)
        else:
            raise ValueError("image, image_url or video should be in content.")
    images = images or None
    videos = videos or None
    if return_video_kwargs:
        return images, videos, {"fps": fps_list, "timestamps": timestamps_list}
    return images, videos


def _video_tensor_to_frames(video: torch.Tensor) -> list[Image.Image]:
    frames = []
    for frame in video:
        if frame.is_floating_point():
            frame = frame.to(torch.uint8) if frame.max() > 1 else (frame * 255).to(torch.uint8)
        else:
            frame = frame.to(torch.uint8)
        frames.append(Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()))
    return frames


# ----------------------------------------------------------------------------------------------
# Processor
# ----------------------------------------------------------------------------------------------

class LocateAnythingProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "image_token",
        "video_token",
        "images_kwargs",
        "videos_kwargs",
        "text_kwargs",
    ]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<IMG_CONTEXT>",
        video_token="<IMG_CONTEXT>",
        merge_kernel_size=(2, 2),
        image_placeholder="image",
        video_placeholder="video",
        image_start_token="<img>",
        image_end_token="</img>",
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", None) or image_token
        self.video_token = getattr(tokenizer, "video_token", None) or video_token
        self.image_token_id = getattr(tokenizer, "image_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.image_token
        )
        self.video_token_id = getattr(tokenizer, "video_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.video_token
        )
        self.merge_kernel_size = list(merge_kernel_size)
        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    # -- serialization ------------------------------------------------------------------------

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        files = super().save_pretrained(save_directory, **kwargs)
        self.image_processor.save_pretrained(save_directory)
        return files

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers.utils import cached_file

        from ....utils.hf import load_hf_tokenizer

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = kwargs.pop("config", None)
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            if config is None:
                from .configuration_locateanything import LocateAnythingConfig

                config = LocateAnythingConfig.from_pretrained(pretrained_model_name_or_path)
            tokenizer = load_hf_tokenizer(
                pretrained_model_name_or_path,
                model_config=config,
                trust_remote_code=False,
            )
        image_processor = LocateAnythingImageProcessor.from_pretrained(pretrained_model_name_or_path)

        processor_dict = {}
        processor_file = cached_file(
            pretrained_model_name_or_path,
            "processor_config.json",
            _raise_exceptions_for_missing_entries=False,
        )
        if processor_file is not None:
            with open(processor_file, encoding="utf-8") as handle:
                processor_dict = json.load(handle)
        if processor_dict.get("chat_template") is None:
            template_file = cached_file(
                pretrained_model_name_or_path,
                "chat_template.jinja",
                _raise_exceptions_for_missing_entries=False,
            )
            if template_file is not None:
                with open(template_file, encoding="utf-8") as handle:
                    processor_dict["chat_template"] = handle.read()
        processor_dict = {key: processor_dict[key] for key in _PROCESSOR_DICT_KEYS if key in processor_dict}
        return cls.from_args_and_dict([image_processor, tokenizer], processor_dict, **kwargs)

    @classmethod
    def from_args_and_dict(cls, args, processor_dict, **kwargs):
        processor_dict = dict(processor_dict)
        for key in ("processor_class", "auto_map", *cls.attributes):
            processor_dict.pop(key, None)
        processor_dict.update(kwargs)
        init_names = cls.__init__.__code__.co_varnames[1: cls.__init__.__code__.co_argcount]
        unused, valid = cls.validate_init_kwargs(processor_config=processor_dict, valid_kwargs=init_names)
        processor = cls(*args, **valid)
        for key, value in unused.items():
            if hasattr(processor, key):
                setattr(processor, key, value)
        return processor

    @property
    def model_input_names(self):
        names = list(self.tokenizer.model_input_names) + list(self.image_processor.model_input_names)
        return list(dict.fromkeys(names))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # -- templating ---------------------------------------------------------------------------

    def py_apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize, "LocateAnythingProcessor.py_apply_chat_template only renders text"
        user_text = ""
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                user_text += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        user_text += item
                    elif isinstance(item, dict) and item.get("type") == "text":
                        user_text += item.get("text", "")

        rendered = ""
        image_count = 0
        video_count = 0
        for index, message in enumerate(messages):
            if index == 0 and message.get("role") != "system":
                rendered += _DEFAULT_SYSTEM
            rendered += f"<|im_start|>{message['role']}\n"
            content = message.get("content")
            if isinstance(content, str):
                rendered += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        rendered += item
                        continue
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image" or "image" in item or "image_url" in item:
                        image_count += 1
                        tag = f"<{self.image_placeholder}-{image_count}>"
                        if tag not in user_text:
                            rendered += tag
                    elif item.get("type") == "video" or "video" in item:
                        video_count += 1
                        tag = f"<{self.video_placeholder}-{video_count}>"
                        if tag not in user_text:
                            rendered += tag
                    elif item.get("type") == "text":
                        rendered += item.get("text", "")
            rendered += "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    @staticmethod
    def extract_vision_info(conversations):
        return extract_vision_info(conversations)

    @staticmethod
    def process_vision_info(conversations, return_video_kwargs=False, video_reader_backend="torchvision"):
        return process_vision_info(
            conversations,
            return_video_kwargs=return_video_kwargs,
            video_reader_backend=video_reader_backend,
        )

    # -- placeholder expansion ----------------------------------------------------------------

    def _media_tokens(self, grid) -> int:
        kh, kw = self.image_processor.merge_kernel_size
        return int(grid[0]) * int(grid[1]) // (kh * kw)

    def _media_segment(self, count: int) -> str:
        return self.image_start_token + self.image_token * count + self.image_end_token

    def replace_media_placeholder(self, text, image_list, video_list, timestamps_list, fps_list, **kwargs):
        images_kwargs = kwargs["images_kwargs"]
        videos_kwargs = kwargs["videos_kwargs"]
        image_list = list(image_list or [])
        video_list = list(video_list or [])
        pattern = re.compile(
            rf"<({re.escape(self.image_placeholder)}|{re.escape(self.video_placeholder)})-(\d+)>"
        )
        pixel_values, grids = [], []
        counts = {"image": 0, "video": 0}

        def expand(match):
            kind = "image" if match.group(1) == self.image_placeholder else "video"
            number = int(match.group(2))
            if number < 1:
                raise IndexError(f"{kind} placeholder index must be 1-based, got {number}")
            if kind == "image":
                replacement = self._expand_image(number, image_list[number - 1], images_kwargs, pixel_values, grids)
            else:
                replacement = self._expand_video(
                    number, video_list[number - 1], timestamps_list, fps_list, videos_kwargs, pixel_values, grids
                )
            counts[kind] += 1
            return replacement

        text = pattern.sub(expand, text)
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)
            grids = np.concatenate(grids, axis=0)
        else:
            pixel_values = torch.tensor([])
            grids = np.array([])
        return text, pixel_values, grids, counts["image"], counts["video"]

    def _run_image_processor(self, images, options, pixel_values, grids):
        options = {key: value for key, value in options.items() if key != "return_tensors"}
        outputs = self.image_processor(images, **options)
        pixel_values.append(torch.as_tensor(outputs["pixel_values"]))
        grid = np.asarray(outputs["image_grid_hws"])
        grids.append(grid)
        return grid

    def _expand_image(self, number, image, options, pixel_values, grids):
        grid = self._run_image_processor(image, options, pixel_values, grids)
        return f"<image {number}>" + self._media_segment(self._media_tokens(grid[0]))

    def _expand_video(self, number, video, timestamps_list, fps_list, options, pixel_values, grids):
        frames = _video_tensor_to_frames(video) if torch.is_tensor(video) else list(video)
        timestamps = None
        if timestamps_list is not None and -1 not in timestamps_list:
            timestamps = list(timestamps_list[number - 1])
            if len(timestamps) != len(frames):
                logger.warning(
                    "LocateAnything video %d has %d frames but %d timestamps; truncating to the shorter",
                    number, len(frames), len(timestamps),
                )
                keep = min(len(timestamps), len(frames))
                timestamps = timestamps[:keep]
                frames = frames[:keep]
        fps = None
        if fps_list is not None and number - 1 < len(fps_list):
            fps = fps_list[number - 1]

        if not 1 <= number <= len(_ORDINALS):
            raise ValueError(f"LocateAnything supports at most {len(_ORDINALS)} videos per sample")
        ordinal = _ORDINALS[number - 1]
        prefix = f"The {ordinal} video sampled with {fps:.2f} fps: " if fps is not None else f"The {ordinal} video: "

        grid = self._run_image_processor(frames, options, pixel_values, grids)
        segments = []
        for index, frame_grid in enumerate(grid):
            label = f"Frame-{index + 1}-{timestamps[index]:.2f}s: " if timestamps is not None else f"Frame-{index + 1}: "
            segments.append(label + self._media_segment(self._media_tokens(frame_grid)))
        return prefix + "".join(segments)

    # -- call ---------------------------------------------------------------------------------

    _IMAGE_OPTION_KEYS = ("patch_size", "in_token_limit", "image_mean", "image_std", "merge_kernel_size")

    def _split_kwargs(self, kwargs):
        text_kwargs = {"padding": False, **dict(kwargs.pop("text_kwargs", None) or {})}
        images_kwargs = dict(kwargs.pop("images_kwargs", None) or {})
        videos_kwargs = dict(kwargs.pop("videos_kwargs", None) or {})
        for key, value in kwargs.items():
            if key in self._IMAGE_OPTION_KEYS:
                images_kwargs.setdefault(key, value)
                videos_kwargs.setdefault(key, value)
            else:
                text_kwargs[key] = value
        return text_kwargs, images_kwargs, videos_kwargs

    def __call__(self, images=None, text=None, audio=None, videos=None, **kwargs):
        del audio
        text_kwargs, images_kwargs, videos_kwargs = self._split_kwargs(kwargs)
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, list) or not text or not all(isinstance(item, str) for item in text):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        timestamps = videos_kwargs.pop("timestamps", None)
        fps = videos_kwargs.pop("fps", None)
        if images is None:
            images = []
        elif not isinstance(images, (list, tuple)):
            images = [images]
        videos = [] if videos is None else list(videos)

        expanded, pixel_values, grids = [], [], []
        images_used = 0
        videos_used = 0
        for sample in text:
            sample_text, sample_pixels, sample_grids, num_images, num_videos = self.replace_media_placeholder(
                sample,
                list(images)[images_used:],
                videos[videos_used:],
                None if timestamps is None else timestamps[videos_used:],
                None if fps is None else fps[videos_used:],
                images_kwargs=images_kwargs,
                videos_kwargs=videos_kwargs,
            )
            expanded.append(sample_text)
            if num_images or num_videos:
                pixel_values.append(sample_pixels)
                grids.append(sample_grids)
            images_used += num_images
            videos_used += num_videos

        return_tensors = text_kwargs.get("return_tensors")
        data = dict(self.tokenizer(expanded, **text_kwargs))
        if pixel_values:
            data["pixel_values"] = torch.cat(pixel_values, dim=0)
            data["image_grid_hws"] = np.concatenate(grids, axis=0)
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = [
    "ImageFetchError",
    "LocateAnythingProcessor",
    "REMOTE_MEDIA_MAX_BYTES",
    "VIDEO_READER_BACKENDS",
    "extract_vision_info",
    "fetch_image",
    "fetch_video",
    "get_video_frame_indices",
    "load_bytes_from_url",
    "load_image_from_url",
    "process_vision_info",
]
