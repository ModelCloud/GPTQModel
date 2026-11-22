# SPDX-FileCopyrightText: 2026 Microsoft
# SPDX-License-Identifier: Apache-2.0
# Adapted from microsoft/Mage-VL@d88b153285f1633a61b2f693c59c8576693af185.
"""Video frame extraction helpers for Mage-VL native video input.

Helpers (decord-first / opencv-fallback decoding) are used by
``MageVLVideoProcessor`` defined below.

The helpers were ported from the training pipeline with minor cleanups:
  - dropped wrapper-only imports
  - consolidated timestamp helpers
  - kept decord-first / opencv-fallback decoding identical

Public API:
  - format_timestamp(seconds) -> "MM:SS.xx"
  - choose_target_frames(duration, max_frames, fixed_num_frames=None,
        target_fps=None) -> int
  - select_frame_indices(frame_count, target_count) -> list[int]
  - smart_resize(h, w, patch_size=16, min_pixels=None, max_pixels=None,
        align_patch_size=None) -> (h, w)
  - extract_video_frames(video_path, ...) -> (frames_np, frame_indices,
        timestamps_dict)
  - extract_video_frames_to_pil(video_path, ...) -> (frames_pil, frame_indices,
        timestamps_dict)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


# =============================================================================
# Timestamp helpers
# =============================================================================

def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:09.6f}"


def time_str_to_seconds(t: str) -> float:
    """Convert ``MM:SS.xx`` back to a float number of seconds.

    Inverse of :func:`format_timestamp`.
    """
    minute, sec = t.split(":")
    return int(minute) * 60 + float(sec)


# =============================================================================
# Frame-count / index selection
# =============================================================================

def choose_target_frames(
    duration_seconds: float,
    max_frames: int,
    fixed_num_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
) -> int:
    """Choose target frame count based on video duration in seconds.

    Sampling strategy:
      - if ``target_fps`` is set, sample at that fps (capped by ``max_frames``)
      - elif ``fixed_num_frames`` is set, use that exact count
      - else duration < 10s  -> 8 frames
      -      duration < 30s  -> 16 frames
      -      otherwise        -> ``max_frames`` (default 32)
    """
    if target_fps is not None and target_fps > 0:
        return min(max(1, int(duration_seconds * target_fps)), max_frames)
    if fixed_num_frames is not None:
        return fixed_num_frames
    if duration_seconds < 10:
        return 8
    if duration_seconds < 30:
        return 16
    return max_frames


def select_frame_indices(frame_count: int, target_count: int) -> List[int]:
    if frame_count <= target_count:
        return list(range(frame_count))
    return torch.linspace(0, frame_count - 1, target_count).round().long().tolist()


# =============================================================================
# Spatial resize
# =============================================================================

def smart_resize(height, width, patch_size=16, min_pixels=None, max_pixels=None, align_patch_size=None):
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid size: height={height}, width={width}")
    factor = align_patch_size or patch_size
    h_bar = max(factor, int(round(height / factor) * factor))
    w_bar = max(factor, int(round(width / factor) * factor))
    if max_pixels and h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif min_pixels and h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


# =============================================================================
# Frame extraction (decord first, opencv fallback)
# =============================================================================

def extract_video_frames(
    video_path: str,
    max_frames: int = 32,
    patch_size: int = 16,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    resize_frames: bool = True,
    fixed_num_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
) -> Tuple[List[np.ndarray], torch.Tensor, dict]:
    """Extract frames from a video.

    Sampling rule matches :func:`choose_target_frames`. Decoding tries decord
    first (better codec coverage) and falls back to OpenCV.

    Args:
        video_path: path to the input video file.
        max_frames: cap for long videos.
        patch_size: vision tower patch size for alignment.
        min_pixels: minimum pixel budget for resize.
        max_pixels: maximum pixel budget for resize.
        resize_frames: whether to apply :func:`smart_resize` (with
            ``align_patch_size = patch_size * 2``, i.e. 28 for spatial_merge=2).
        fixed_num_frames: see :func:`choose_target_frames`.
        target_fps: see :func:`choose_target_frames`.

    Returns:
        Tuple of:
          - ``frames``   : list of RGB ``np.ndarray`` (H, W, 3), dtype uint8.
          - ``frame_indices`` : 1D ``torch.Tensor[int64]`` of selected indices.
          - ``timestamps`` : ``dict[str(frame_idx) -> "MM:SS.xx"]``.

    Notes:
        Lazy imports of ``decord`` and ``cv2`` keep the module importable in
        environments where neither is installed (e.g. unit tests that only
        exercise the helpers above).
    """
    frames: List[np.ndarray] = []
    timestamps: dict = {}
    frame_indices: List[int] = []

    # Prefer decord because of broader codec support.
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(video_path)
        frame_count = len(vr)
        fps = vr.get_avg_fps()
        if not fps or fps <= 0:
            fps = 30.0

        duration = frame_count / fps
        target_count = choose_target_frames(
            duration, max_frames, fixed_num_frames, target_fps
        )
        selected_indices = select_frame_indices(frame_count, target_count)

        # One-shot batch decode + torchvision BICUBIC+antialias resize.
        # Mirrors qwen_vl_utils.fetch_video, replacing per-frame cv2 INTER_AREA/LINEAR.
        arr = vr.get_batch(selected_indices).asnumpy()  # [N,H,W,3] uint8 RGB
        H, W = arr.shape[1], arr.shape[2]
        if resize_frames and (min_pixels or max_pixels):
            resized_h, resized_w = smart_resize(
                H, W, patch_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                align_patch_size=patch_size * 2,
            )
            if (resized_h, resized_w) != (H, W):
                from torchvision.transforms import InterpolationMode as _IM
                from torchvision.transforms.functional import resize
                video_t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                video_t = resize(
                    video_t,
                    [resized_h, resized_w],
                    interpolation=_IM.BICUBIC,
                    antialias=True,
                )
                arr = video_t.permute(0, 2, 3, 1).contiguous().numpy()

        frames = list(arr)
        frame_indices = list(selected_indices)
        for frame_idx in selected_indices:
            timestamps[str(int(frame_idx))] = format_timestamp(int(frame_idx) / fps)

        return frames, torch.tensor(frame_indices, dtype=torch.int64), timestamps
    except Exception as e:
        logger.warning(
            f"decord failed to open {video_path}: {e}; falling back to OpenCV"
        )

    # OpenCV fallback.
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"OpenCV also failed to open video, skipped: {video_path}")
        return frames, torch.tensor(frame_indices, dtype=torch.int64), timestamps

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_count > 0:
        duration = frame_count / fps
        target_count = choose_target_frames(
            duration, max_frames, fixed_num_frames, target_fps
        )
        selected_indices = select_frame_indices(frame_count, target_count)

        for frame_idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if resize_frames and (min_pixels or max_pixels):
                resized_h, resized_w = smart_resize(
                    frame.shape[0],
                    frame.shape[1],
                    patch_size,
                    min_pixels,
                    max_pixels,
                    align_patch_size=patch_size * 2,
                )
                if (resized_h, resized_w) != (frame.shape[0], frame.shape[1]):
                    interp = (
                        cv2.INTER_AREA
                        if resized_h < frame.shape[0] or resized_w < frame.shape[1]
                        else cv2.INTER_LINEAR
                    )
                    frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interp)

            frames.append(frame)
            timestamps[str(frame_idx)] = format_timestamp(frame_idx / fps)
            frame_indices.append(frame_idx)
    else:
        # Unknown frame count: read sequentially then sample.
        frame_idx = 0
        temp_frames: List[Tuple[int, np.ndarray]] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            temp_frames.append((frame_idx, frame))
            frame_idx += 1

        if temp_frames:
            duration = len(temp_frames) / fps
            target_count = choose_target_frames(
                duration, max_frames, fixed_num_frames, target_fps
            )
            selected_indices = select_frame_indices(len(temp_frames), target_count)

            for idx in selected_indices:
                frame_idx, frame = temp_frames[idx]
                if resize_frames and (min_pixels or max_pixels):
                    resized_h, resized_w = smart_resize(
                        frame.shape[0],
                        frame.shape[1],
                        patch_size,
                        min_pixels,
                        max_pixels,
                        align_patch_size=patch_size * 2,
                    )
                    if (resized_h, resized_w) != (frame.shape[0], frame.shape[1]):
                        interp = (
                            cv2.INTER_AREA
                            if resized_h < frame.shape[0] or resized_w < frame.shape[1]
                            else cv2.INTER_LINEAR
                        )
                        frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interp)

                frames.append(frame)
                timestamps[str(frame_idx)] = format_timestamp(frame_idx / fps)
                frame_indices.append(frame_idx)

    cap.release()
    return frames, torch.tensor(frame_indices, dtype=torch.int64), timestamps


def extract_video_frames_to_pil(
    video_path: str,
    max_frames: int = 32,
    patch_size: int = 16,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    resize_frames: bool = True,
    fixed_num_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
):
    """Same as :func:`extract_video_frames` but returns a list of PIL Images."""
    from PIL import Image  # local import: PIL is mandatory for the processor

    frames_np, frame_indices, timestamps = extract_video_frames(
        video_path=video_path,
        max_frames=max_frames,
        patch_size=patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        resize_frames=resize_frames,
        fixed_num_frames=fixed_num_frames,
        target_fps=target_fps,
    )
    frames_pil = [Image.fromarray(frame) for frame in frames_np]
    return frames_pil, frame_indices, timestamps


# =============================================================================
# patch_positions construction (row-major + 2x2 block-layout reorder)
# =============================================================================
# Block-layout reorder mirroring the training pipeline, kept here so the
# VideoProcessor is self-contained.

def _convert_positions_to_block_layout(
    positions: torch.Tensor,
    t: int,
    h: int,
    w: int,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Reorder ``[t*h*w, 3]`` row-major positions to 2x2 block layout."""
    sms = spatial_merge_size
    if sms == 1:
        return positions
    device = positions.device
    total = t * h * w
    indices = torch.arange(total, device=device).view(t, h, w)
    h_m, w_m = h // sms, w // sms
    indices = (
        indices.view(t, h_m, sms, w_m, sms)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(total)
    )
    return positions[indices]


def build_patch_positions(
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    frame_indices: Optional[List[Optional[torch.Tensor]]] = None,
) -> torch.Tensor:
    """Build block-layout ``[t,h,w]`` patch positions for one or many videos/images.

    Args:
        grid_thw: ``[num_samples, 3]`` LongTensor (T, H_p, W_p) per sample.
        spatial_merge_size: vision tower spatial-merge size (default 2).
        frame_indices: optional list (one entry per row of ``grid_thw``) of
            real frame indices to use as the t-coordinate. Each entry should
            be a 1-D LongTensor of length ``T`` for that sample. When provided
            this matches the training pipeline,
            where ``t`` is the original frame number in the source video so
            the vision tower's 3-D RoPE encodes the actual temporal position
            rather than a 0..T-1 dense index. Pass ``None`` for an entry to
            fall back to dense ``arange(T)`` for that sample.

    Returns:
        ``[sum(T*H_p*W_p), 3]`` Int64Tensor in block layout, ready to feed
        ``forward(... patch_positions=...)``.
    """
    out = []
    for sample_idx, row in enumerate(grid_thw):
        t_v, h_v, w_v = int(row[0]), int(row[1]), int(row[2])
        h_coords = torch.arange(h_v, dtype=torch.int64).repeat_interleave(w_v).repeat(t_v)
        w_coords = torch.arange(w_v, dtype=torch.int64).repeat(h_v).repeat(t_v)
        # t-coords: prefer real frame_indices (training convention) when given.
        sample_frame_idx = None
        if frame_indices is not None and sample_idx < len(frame_indices):
            sample_frame_idx = frame_indices[sample_idx]
        if sample_frame_idx is not None:
            fi = torch.as_tensor(sample_frame_idx, dtype=torch.int64)
            if fi.numel() != t_v:
                raise ValueError(
                    f"frame_indices[{sample_idx}] has length {fi.numel()} but "
                    f"grid_thw[{sample_idx}, 0] = {t_v}"
                )
            t_coords = fi.repeat_interleave(h_v * w_v)
        else:
            # Each frame's t coordinate runs 0..t_v-1 (each value repeated h_v*w_v).
            t_coords = torch.arange(t_v, dtype=torch.int64).repeat_interleave(h_v * w_v)
        pp = torch.stack([t_coords, h_coords, w_coords], dim=1)
        pp = _convert_positions_to_block_layout(pp, t_v, h_v, w_v, spatial_merge_size)
        out.append(pp)
    return torch.cat(out, dim=0)


# =============================================================================
# MageVLVideoProcessor
# =============================================================================
# A thin processor that wraps `Qwen2VLImageProcessor` to convert raw video
# files (or pre-decoded frame lists) into the tensor bundle needed by the
# MageVL model.
#
# Output (BatchFeature):
#   - pixel_values_videos : [sum(T*H_p*W_p), C, P, P]      patch tensor
#   - video_grid_thw      : [num_videos, 3]                (T_eff, H_p, W_p)
#   - patch_positions     : [sum(T*H_p*W_p), 3]            block layout
#   - frame_timestamps    : list[list[float]]              per-video per-frame seconds
#
# Aligned with the modeling code, we deliberately
# DO NOT emit `second_per_grid_ts`.

class MageVLVideoProcessor:
    """Decode + sample + patch-ify videos for MageVL.

    Designed to be standalone (does not inherit ``transformers.ProcessorMixin``)
    so it can be unit-tested without the full Processor stack.
    """

    # Canonical defaults.
    DEFAULT_MAX_FRAMES = 384
    DEFAULT_PATCH_SIZE = 16
    DEFAULT_SPATIAL_MERGE_SIZE = 2
    DEFAULT_TEMPORAL_PATCH_SIZE = 1  # this checkpoint ships tps=1
    DEFAULT_MIN_PIXELS = 256 * 28 * 28
    DEFAULT_MAX_PIXELS = 1605632

    def __init__(
        self,
        image_processor=None,
        max_frames: int = DEFAULT_MAX_FRAMES,
        fixed_num_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
        patch_size: int = DEFAULT_PATCH_SIZE,
        spatial_merge_size: int = DEFAULT_SPATIAL_MERGE_SIZE,
        temporal_patch_size: int = DEFAULT_TEMPORAL_PATCH_SIZE,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        resize_frames: bool = True,
    ):
        """
        Args:
            image_processor: a `Qwen2VLImageProcessor` instance. If ``None`` an
                instance is built from the other kwargs at first call.
            max_frames / fixed_num_frames / target_fps: see
                :func:`choose_target_frames`.
            patch_size: vision tower patch size (default 16).
            spatial_merge_size: vision tower spatial merge factor (default 2).
            temporal_patch_size: temporal-patch grouping; this checkpoint
                ships ``temporal_patch_size=1`` so each pv row is one single
                patch (3*16*16=768) and ``Σ t·h·w == total_patches``
                naturally. Override only if loading a non-default processor.
            min_pixels / max_pixels: smart_resize budget.
            resize_frames: whether to resize frames before patching.
        """
        self._image_processor = image_processor
        self.max_frames = max_frames
        self.fixed_num_frames = fixed_num_frames
        self.target_fps = target_fps
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.resize_frames = resize_frames

    # ------------------------------------------------------------------ utils

    @property
    def image_processor(self):
        """Lazy-build the underlying `Qwen2VLImageProcessor`."""
        if self._image_processor is None:
            from ..native.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

            self._image_processor = Qwen2VLImageProcessor(
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                patch_size=self.patch_size,
                merge_size=self.spatial_merge_size,
                temporal_patch_size=self.temporal_patch_size,
            )
        return self._image_processor

    @staticmethod
    def _coerce_video_input(video):
        """Normalise a single video input to ``(frames_pil, timestamps_seconds)``.

        Accepts:
          - ``str`` path to a video file,
          - ``list[PIL.Image]`` (already decoded; timestamps default to None),
          - ``list[np.ndarray]`` (RGB uint8; converted to PIL).
        """
        from PIL import Image

        if isinstance(video, str):
            return None  # signal: use video path through extract_video_frames_to_pil
        if isinstance(video, list) and len(video) > 0:
            first = video[0]
            if isinstance(first, Image.Image):
                return list(video), None
            if isinstance(first, np.ndarray):
                return [Image.fromarray(f) for f in video], None
        raise TypeError(
            f"Unsupported video input type: {type(video).__name__}. "
            "Expected file path, list[PIL.Image], or list[np.ndarray]."
        )

    # ---------------------------------------------------------------- __call__

    def __call__(
        self,
        videos,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ):
        """Process one or several videos.

        Args:
            videos: a single video or a list of videos. Each video may be a
                path, a list of PIL frames, or a list of np.ndarray RGB frames.
            return_tensors: only ``"pt"`` is supported (mirrors the underlying
                image processor).
            **kwargs: ignored / reserved for transformers ProcessorMixin
                compatibility (e.g. ``do_rescale``).

        Returns:
            A dict-like object with keys:
              - ``pixel_values_videos`` : Tensor ``[N_total_patches, C, P, P]``
              - ``video_grid_thw``      : Tensor ``[num_videos, 3]`` (T, H_p, W_p)
              - ``patch_positions``     : Tensor ``[N_total_patches, 3]`` block layout
              - ``frame_timestamps``    : ``list[list[float]]`` per video
        """
        if return_tensors not in (None, "pt"):
            raise ValueError(
                f"return_tensors={return_tensors!r} not supported; only 'pt' is."
            )

        # Normalise to a list of videos.
        if not isinstance(videos, (list, tuple)) or (
            len(videos) > 0
            and (isinstance(videos[0], str) is False)
            and not isinstance(videos[0], list)
        ):
            # Heuristic: a single video as `list[PIL.Image]` should not be
            # treated as a batch of single-frame videos. We detect that case
            # by checking the inner element type.
            from PIL import Image

            if isinstance(videos, list) and len(videos) > 0 and isinstance(
                videos[0], (Image.Image, np.ndarray)
            ):
                videos = [videos]
            elif isinstance(videos, str):
                videos = [videos]
        if not isinstance(videos, (list, tuple)):
            videos = [videos]

        per_video_pixel_values = []
        per_video_grid_thw = []
        per_video_patch_positions = []
        frame_timestamps_all: List[List[float]] = []

        for video in videos:
            # 1) Decode + sample
            if isinstance(video, str):
                frames_pil, frame_indices, timestamps = extract_video_frames_to_pil(
                    video_path=video,
                    max_frames=self.max_frames,
                    patch_size=self.patch_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    resize_frames=self.resize_frames,
                    fixed_num_frames=self.fixed_num_frames,
                    target_fps=self.target_fps,
                )
                # Reconstruct fps from any two timestamps, fall back to 30.
                seconds_seq: List[float] = []
                if len(frames_pil) > 0:
                    fi_list = frame_indices.tolist()
                    for fi in fi_list:
                        ts = timestamps.get(str(int(fi)))
                        if ts is None:
                            seconds_seq.append(0.0)
                        else:
                            seconds_seq.append(time_str_to_seconds(ts))
                # Real frame indices in the source video (training convention
                # for the t-axis of patch_positions).
                frame_indices_t = frame_indices.to(torch.int64)
            else:
                pre_decoded = self._coerce_video_input(video)
                if pre_decoded is None:
                    raise ValueError("Mage-VL video needs a path or predecoded image/array frames.")
                frames_pil, _ = pre_decoded
                seconds_seq = [float(i) for i in range(len(frames_pil))]
                # Without the original video we have no real indices; fall back
                # to dense ``arange(T)``.
                frame_indices_t = torch.arange(len(frames_pil), dtype=torch.int64)

            if len(frames_pil) == 0:
                raise ValueError(f"No frames decoded from video: {video!r}")

            # 2) Patch-ify via Qwen2VLImageProcessor.
            #    Video frames go
            #    through the *image* path, one frame == one image. The
            #    resulting `image_grid_thw` has shape ``[N, 3]`` with each row
            #    ``[1, H_p, W_p]``. We then merge into a single video grid
            #    ``[1, T=N, H_p, W_p]`` (smart_resize guarantees same H/W).
            #
            #    Important: this checkpoint ships an image processor with
            #    ``temporal_patch_size=1``, so each pv row encodes ONE single
            #    patch (3*16*16 = 768). The Mage-ViT embedding
            #    layer reshapes pv via ``view(-1, 3, 16, 16)`` and produces
            #    exactly ``pv.shape[0]`` patches, so the cu_seqlens check
            #    ``Σ t·h·w == total_patches`` is satisfied with the natural
            #    per-frame grid below. The lazy-built fallback in
            #    ``image_processor`` honors ``temporal_patch_size=1`` to keep
            #    standalone tests aligned with the checkpoint convention.
            ip = self.image_processor
            data = ip(images=frames_pil, return_tensors="pt")
            pixel_values = data["pixel_values"]
            image_grid_thw = data["image_grid_thw"]  # [N, 3]

            if not torch.all(image_grid_thw[:, 1] == image_grid_thw[0, 1]) or not torch.all(
                image_grid_thw[:, 2] == image_grid_thw[0, 2]
            ):
                raise RuntimeError(
                    "Frames yielded inconsistent (H_p, W_p); smart_resize should "
                    f"prevent this. Got grid_thw={image_grid_thw.tolist()}"
                )

            T_eff = int(image_grid_thw[:, 0].sum().item())  # sum of per-frame t (each is 1)
            H_p = int(image_grid_thw[0, 1].item())
            W_p = int(image_grid_thw[0, 2].item())
            video_grid_thw = torch.tensor(
                [[T_eff, H_p, W_p]], dtype=image_grid_thw.dtype
            )
            pixel_values_videos = pixel_values  # already [T_eff*H_p*W_p, C, P, P]

            # 3) patch_positions in block layout (over the merged video grid).
            #    Use REAL frame_indices for the t-axis (training convention).
            patch_positions = build_patch_positions(
                video_grid_thw,
                spatial_merge_size=self.spatial_merge_size,
                frame_indices=[frame_indices_t],
            )

            per_video_pixel_values.append(pixel_values_videos)
            per_video_grid_thw.append(video_grid_thw)
            per_video_patch_positions.append(patch_positions)
            frame_timestamps_all.append(seconds_seq)

        out_pixel_values = torch.cat(per_video_pixel_values, dim=0)
        out_grid_thw = torch.cat(per_video_grid_thw, dim=0)
        out_patch_positions = torch.cat(per_video_patch_positions, dim=0)

        try:
            from transformers.feature_extraction_utils import BatchFeature

            return BatchFeature(
                data={
                    "pixel_values_videos": out_pixel_values,
                    "video_grid_thw": out_grid_thw,
                    "patch_positions": out_patch_positions,
                    "frame_timestamps": frame_timestamps_all,
                }
            )
        except Exception:
            return {
                "pixel_values_videos": out_pixel_values,
                "video_grid_thw": out_grid_thw,
                "patch_positions": out_patch_positions,
                "frame_timestamps": frame_timestamps_all,
            }


__all__ = [
    "format_timestamp",
    "time_str_to_seconds",
    "choose_target_frames",
    "select_frame_indices",
    "smart_resize",
    "extract_video_frames",
    "extract_video_frames_to_pil",
    "build_patch_positions",
    "MageVLVideoProcessor",
]
