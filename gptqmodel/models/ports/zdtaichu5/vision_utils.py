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

# ============================================================================
#  Vision utilities for ZDTaichu-5.0
#
#  Provides ``process_vision_info()`` to extract images and videos from
#  Qwen-style structured messages, following the conventions established
#  by ``qwen_vl_utils``.  This allows the model to accept messages like:
#
#      messages = [
#          {"role": "user", "content": [
#              {"type": "video", "video": "path/to/video.mp4", "fps": 2.0},
#              {"type": "text", "text": "Describe this video."},
#          ]}
#      ]
#
#  Supported input formats:
#    - Images: local path, ``file://`` URI, ``http(s)://`` URL, base64 data
#              URI, ``PIL.Image.Image`` object
#    - Videos: local path, ``file://`` URI, ``http(s)://`` URL (string),
#              or a list of image paths/URLs (treated as pre-extracted frames)
#
#  Video decoding backends (auto-detected, in priority order):
#    1. decord      — fastest, recommended
#    2. torchvision — fallback, always available
#
#  Frame sampling follows the same ``smart_nframes`` logic as qwen_vl_utils:
#    - Default: 2 FPS, clamped to [4, 768] frames, rounded to factor of 2
#    - Override via ``fps``, ``nframes``, ``min_frames``, ``max_frames``
#    - Temporal trimming via ``video_start`` / ``video_end`` (seconds)
# ============================================================================

import base64
import copy
import logging
import math
import os
import sys
import tempfile
import time
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image

from ....utils.image import load_bytes_from_url, load_image_from_url


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (aligned with qwen_vl_utils defaults)
# ─────────────────────────────────────────────────────────────────────────────

FPS = 2.0                   # default sampling rate
FRAME_FACTOR = 2            # frame count must be divisible by this
FPS_MIN_FRAMES = 4          # minimum sampled frames
FPS_MAX_FRAMES = 768        # maximum sampled frames


# ─────────────────────────────────────────────────────────────────────────────
# Rounding helpers
# ─────────────────────────────────────────────────────────────────────────────

def round_by_factor(number: float, factor: int) -> int:
    """Closest integer to *number* divisible by *factor*."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Smallest integer ≥ *number* divisible by *factor*."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Largest integer ≤ *number* divisible by *factor*."""
    return math.floor(number / factor) * factor


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def fetch_image(ele: Dict[str, Any]) -> Image.Image:
    """
    Load a single image from various sources.

    Supported formats for ``ele["image"]``:
      - ``PIL.Image.Image`` instance
      - Local file path (``/path/to/img.jpg``)
      - ``file://`` URI
      - ``http://`` or ``https://`` URL
      - Base64 data URI (``data:image/...;base64,...``)

    Returns:
        PIL.Image.Image in RGB mode.
    """
    image = ele.get("image") or ele.get("image_url")
    if image is None:
        raise ValueError("Element must contain 'image' or 'image_url' key")

    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = load_image_from_url(image)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, b64 = image.split("base64,", 1)
            image_obj = copy.deepcopy(Image.open(BytesIO(base64.b64decode(b64))))
    else:
        # Treat as local file path
        image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(
            f"Unrecognised image input. Supported: local path, file:// URI, "
            f"http(s) URL, base64 data URI, PIL.Image. Got: {image!r:.120}"
        )

    # Convert to RGB
    if image_obj.mode == "RGBA":
        bg = Image.new("RGB", image_obj.size, (255, 255, 255))
        bg.paste(image_obj, mask=image_obj.split()[3])
        return bg
    return image_obj.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ─────────────────────────────────────────────────────────────────────────────

def smart_nframes(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> int:
    """
    Compute the number of frames to sample from a video.

    Follows the same logic as ``qwen_vl_utils.smart_nframes``:
      - If ``ele["nframes"]`` is set, use it directly (rounded to FRAME_FACTOR).
      - Otherwise, sample at ``ele.get("fps", 2.0)`` FPS, clamped to
        ``[min_frames, max_frames]`` and rounded down to FRAME_FACTOR.

    Args:
        ele: Dict with optional keys ``fps``, ``nframes``, ``min_frames``,
             ``max_frames``.
        total_frames: Total frames in the (possibly trimmed) video.
        video_fps: Original video FPS.

    Returns:
        Number of frames to sample.
    """
    assert not ("fps" in ele and "nframes" in ele), (
        "Only accept either `fps` or `nframes`, not both"
    )

    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(
            ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR
        )
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: computed nframes ({nframes:.1f}) > "
                f"total_frames ({total_frames})"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)

    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(
            f"nframes should be in [{FRAME_FACTOR}, {total_frames}], "
            f"got {nframes}."
        )
    return nframes


def calculate_video_frame_range(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> Tuple[int, int, int]:
    """
    Calculate start/end frame indices from optional ``video_start``/``video_end``
    keys (in seconds).

    Returns:
        (start_frame, end_frame, frame_count)  — end_frame is inclusive.
    """
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    video_start = ele.get("video_start")
    video_end = ele.get("video_end")

    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps

    if video_start is not None:
        start_sec = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(start_sec * video_fps)
    else:
        start_frame = 0

    if video_end is not None:
        end_sec = max(0.0, min(video_end, max_duration))
        end_frame = min(math.floor(end_sec * video_fps), total_frames - 1)
    else:
        end_frame = total_frames - 1

    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: start_frame={start_frame} >= end_frame={end_frame}. "
            f"Video: {max_duration:.2f}s ({total_frames} frames @ {video_fps:.1f}fps)"
        )

    return start_frame, end_frame, end_frame - start_frame + 1


# ─────────────────────────────────────────────────────────────────────────────
# Video decoding backends
# ─────────────────────────────────────────────────────────────────────────────

def _read_video_decord(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, dict, float]:
    """Read video with decord. Returns (video_TCHW, metadata, sample_fps)."""
    import decord

    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    video = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)  # TCHW
    logger.info(
        f"decord: {video_path}, {total_frames} frames, "
        f"{video_fps:.1f} fps, sampled {nframes}, "
        f"time={time.time() - st:.3f}s"
    )

    metadata = {
        "fps": video_fps,
        "sample_fps": sample_fps,
        "frames_indices": idx,
        "total_num_frames": total_frames,
        "video_backend": "decord",
    }
    return video, metadata, sample_fps


def _read_video_torchvision(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, dict, float]:
    """Read video with torchvision. Returns (video_TCHW, metadata, sample_fps)."""
    from torchvision import io as tio

    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    st = time.time()
    video, _audio, info = tio.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end"),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]

    logger.info(
        f"torchvision: {video_path}, {total_frames} frames, "
        f"{video_fps:.1f} fps, sampled {nframes}, "
        f"time={time.time() - st:.3f}s"
    )

    metadata = {
        "fps": video_fps,
        "sample_fps": sample_fps,
        "frames_indices": idx.tolist(),
        "total_num_frames": total_frames,
        "video_backend": "torchvision",
    }
    return video, metadata, sample_fps


def _is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None


@lru_cache(maxsize=1)
def _get_video_backend() -> str:
    if _is_decord_available():
        backend = "decord"
    else:
        backend = "torchvision"
    print(
        f"ZDTaichu-5.0 utilities using {backend} to read video.",
        file=sys.stderr,
    )
    return backend


_VIDEO_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}


# ─────────────────────────────────────────────────────────────────────────────
# fetch_video — main entry point for video loading
# ─────────────────────────────────────────────────────────────────────────────

def _localize_remote_video(ele: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Download an http(s) video through the bounded fetcher into a temp file.

    Video decoders would otherwise stream the URL themselves without the byte,
    time, redirect and private-address limits used for images.
    """
    video = ele["video"]
    if not video.startswith(("http://", "https://")):
        return ele, None
    data = load_bytes_from_url(video)
    suffix = os.path.splitext(urlsplit(video).path)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
    return {**ele, "video": handle.name}, handle.name


def fetch_video(
    ele: Dict[str, Any],
) -> Tuple[List[Image.Image], float, dict]:
    """
    Load and sample frames from a video.

    The ``ele["video"]`` value can be:
      - A string path / URI  → decoded with decord or torchvision
      - A list of image paths → loaded as pre-extracted frames

    Returns:
        (frames, sample_fps, metadata)
          - frames: list of PIL.Image.Image in RGB (one per sampled frame)
          - sample_fps: effective sampling rate after frame selection
          - metadata: dict with ``fps``, ``sample_fps``, ``total_num_frames``,
                      ``frames_indices``, ``video_backend``
    """
    if isinstance(ele["video"], str):
        # ── Decode from video file ───────────────────────────────────────
        ele, downloaded = _localize_remote_video(ele)
        try:
            backend = _get_video_backend()
            try:
                video_tensor, metadata, sample_fps = _VIDEO_BACKENDS[backend](ele)
            except Exception as exc:
                if backend != "torchvision":
                    logger.warning(
                        f"{backend} failed ({exc}), falling back to torchvision"
                    )
                    video_tensor, metadata, sample_fps = _read_video_torchvision(ele)
                else:
                    raise
        finally:
            if downloaded is not None:
                os.unlink(downloaded)

        # Convert TCHW tensor → list of PIL images
        frames = []
        for i in range(video_tensor.size(0)):
            frame_np = video_tensor[i].permute(1, 2, 0).numpy().astype(np.uint8)  # HWC
            frames.append(Image.fromarray(frame_np, "RGB"))

    elif isinstance(ele["video"], (list, tuple)):
        # ── Pre-extracted frames (paths or PIL images) ───────────────────
        frame_elements = ele["video"]
        frames = []
        for item in frame_elements:
            frames.append(fetch_image({"image": item}))

        # Pad to FRAME_FACTOR multiple
        nframes = ceil_by_factor(len(frames), FRAME_FACTOR)
        while len(frames) < nframes:
            frames.append(frames[-1].copy())

        sample_fps = ele.get("fps", FPS)
        raw_fps = ele.get("raw_fps", sample_fps)
        metadata = {
            "fps": raw_fps,
            "sample_fps": sample_fps,
            "frames_indices": list(range(len(frames))),
            "total_num_frames": len(frames),
            "video_backend": "frames_list",
        }
    else:
        raise TypeError(
            f"ele['video'] must be a string (path) or list (frames), "
            f"got {type(ele['video'])}"
        )

    return frames, sample_fps, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Message parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """
    Extract all vision elements (image / video dicts) from Qwen-style
    structured messages.

    Args:
        conversations: Either a single conversation (list of message dicts)
            or a batch of conversations.

    Returns:
        Flat list of vision element dicts, in order of appearance.
    """
    # Normalise to batch format
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    vision_infos = []
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for ele in content:
                if (
                    "image" in ele
                    or "image_url" in ele
                    or "video" in ele
                    or ele.get("type") in ("image", "image_url", "video")
                ):
                    vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> Tuple[Optional[List[Image.Image]], Optional[List[List[Image.Image]]], Optional[Dict[str, Any]]]:
    """
    Extract and load all images and videos from structured messages.

    This is the main entry point — equivalent to
    ``qwen_vl_utils.process_vision_info`` — adapted for ZDTaichu-5.0.

    Args:
        conversations: Qwen-style messages with structured ``content`` lists
            containing ``{"type": "image", "image": ...}`` and/or
            ``{"type": "video", "video": ...}`` elements.

    Returns:
        (image_inputs, video_inputs, video_kwargs)
          - image_inputs: list of PIL images, or None
          - video_inputs: list of frame-lists (each is ``List[PIL.Image]``),
                          or None
          - video_kwargs: dict with ``sample_fps_list`` and ``metadata_list``

    Example::

        from vision_utils import process_vision_info

        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": "clip.mp4", "fps": 2.0},
                {"type": "text", "text": "Describe this video."},
            ]}
        ]

        images, videos, video_kwargs = process_vision_info(messages)
        # images = None
        # videos = [[PIL.Image, PIL.Image, ...]]  (one list of frames per video)
        # video_kwargs = {"sample_fps_list": [2.0], "metadata_list": [...]}
    """
    vision_infos = extract_vision_info(conversations)

    image_inputs: List[Image.Image] = []
    video_inputs: List[List[Image.Image]] = []
    sample_fps_list: List[float] = []
    metadata_list: List[dict] = []

    for info in vision_infos:
        if "image" in info or "image_url" in info:
            image_inputs.append(fetch_image(info))

        elif "video" in info:
            frames, sample_fps, metadata = fetch_video(info)
            video_inputs.append(frames)
            sample_fps_list.append(sample_fps)
            metadata_list.append(metadata)

        else:
            raise ValueError(
                "Vision element must contain 'image', 'image_url', or 'video' key."
            )

    video_kwargs = {
        "sample_fps_list": sample_fps_list,
        "metadata_list": metadata_list,
    }

    return (
        image_inputs if image_inputs else None,
        video_inputs if video_inputs else None,
        video_kwargs,
    )


def build_text_from_messages(
    messages: List[Dict[str, Any]],
    image_token: str = "<|image_pad|>",
    video_token: str = "<|video_pad|>",
) -> List[Dict[str, Any]]:
    """
    Convert structured messages (with typed content lists) into plain-text
    messages that ``apply_chat_template`` can handle.

    Each ``{"type": "image", ...}`` is replaced with ``image_token``.
    Each ``{"type": "video", ...}`` is replaced with ``video_token``.
    Text elements are concatenated.

    Returns:
        New message list with plain string ``content`` fields.
    """
    output = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            output.append(msg)
            continue

        parts = []
        for ele in content:
            typ = ele.get("type", "text")
            if typ == "text":
                parts.append(ele.get("text", ""))
            elif typ in ("image", "image_url"):
                parts.append(image_token)
            elif typ == "video":
                parts.append(video_token)
        output.append({**msg, "content": "".join(parts)})
    return output


__all__ = [
    "fetch_image",
    "fetch_video",
    "smart_nframes",
    "extract_vision_info",
    "process_vision_info",
    "build_text_from_messages",
]
