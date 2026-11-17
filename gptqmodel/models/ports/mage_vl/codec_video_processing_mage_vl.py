# SPDX-FileCopyrightText: 2026 Microsoft
# SPDX-License-Identifier: Apache-2.0
# Adapted from microsoft/Mage-VL@d88b153285f1633a61b2f693c59c8576693af185.
"""Package-owned traditional codec preprocessing; neural DCVC fails closed.

This module is the codec analogue of ``video_processing_magevl.py``.
It is invoked when a user calls::

    processor(messages=..., video_backend="codec", max_pixels=...)

and is responsible for:

  - Decoding the video and assembling canvas images via ``cv-preinfer``
    (PyPI: ``codec-video-prep``, requires ``ffmpeg`` on PATH).
  - Running the bundled ``Qwen2VLImageProcessor`` on those canvases with a
    pixel budget that is *aligned* to the canvas dimensions (so the
    smart_resize step never desynchronises ``image_grid_thw`` from the
    codec-emitted ``src_patch_position`` array).
  - Producing the per-patch ``patch_positions`` table that
    ``modeling_magevl.py`` reads for the 2D-MRoPE block layout.

The result is a ``BatchFeature``-shaped dict containing the same keys that
the frame-sampling video path produces (``pixel_values`` /
``image_grid_thw`` / ``patch_positions``), so downstream
``modeling_magevl.py`` consumes it without changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Optional


try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

import numpy as np
import torch
from PIL import Image


VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

# Matches exactly one chat-template video placeholder block, so that image
# blocks elsewhere in the prompt are left untouched.
_VIDEO_BLOCK_RE = re.compile(
    re.escape(VISION_START) + r"\s*" + re.escape(VIDEO_PAD) + r"\s*" + re.escape(VISION_END)
)


# ----------------------------------------------------------------- config

@dataclass
class DcvcConfig:
    """DCVC-RT neural-codec knobs (nested under ``CodecConfig.dcvc``).

    Only used when ``CodecConfig.engine == "dcvc-rt"``. Checkpoints default to
    the ``DCVC_INTRA_TAR`` / ``DCVC_INTER_TAR`` env vars (environment-specific,
    so better left out of the model config).
    """
    qp: int = 42                          # quantization (0-63); scoring only
    reset_interval: int = 64              # feature-adaptor reset period (frames)
    intra_period: int = -1                # -1 => single I-frame at t=0, rest P
    max_side: int = 0                     # >0 downscales frames before DCVC scoring;
                                          # 0 => full-resolution scoring (matches the
                                          # hevc baseline, which reads h264 bits at
                                          # native stream resolution). Faithful but
                                          # slower to encode.
    seq_len_frames: int = 0               # 0 => CodecConfig.num_sampled_frames()
    patch: int = 16                       # MUST match image processor patch_size
    canvas_token_side: Optional[int] = None  # None => derive from max_pixels
    # ---- readiness grouping knobs (used by the bundled process_video_bitcost_
    # readiness pipeline). Defaults reproduce EXACTLY the offline/cluster DCVC
    # sweep + hevc baseline generation params, so the release DCVC path yields
    # the same assets we evaluated. (These intentionally differ from the release
    # hevc _run_cv_preinfer knobs, e.g. max_group_frames=128 vs CodecConfig's 64.)
    readiness_sum_threshold_mode: str = "auto"
    readiness_coverage_bins: int = 3
    readiness_delta_ratio: float = 0.05
    bitcost_grid: str = "sub"             # DCVC bitmap is at 16px granularity
    bitcost_pct: int = 99
    decode_backsearch_max: int = 16
    max_group_frames: int = 128
    intra_ckpt: str = field(default_factory=lambda: os.getenv("DCVC_INTRA_TAR", ""))
    inter_ckpt: str = field(default_factory=lambda: os.getenv("DCVC_INTER_TAR", ""))
    device: str = "cuda:0"
    pkg_dir: str = ""                     # dir with dcvc_rt_engine.py + dcvc_readiness_gen.py + codec_tools/


@dataclass
class CodecConfig:
    """All knobs for the codec preprocessing pipeline.

    ``max_pixels`` is shared with the image_processor / video_processor pixel
    budget. The processor sets it from the user's ``max_pixels=`` kwarg, so
    canvas size and HF smart_resize budget stay consistent.
    """

    target_canvas: int = 32
    group_size: int = 32
    images_per_group: int = 4
    patch: int = 14
    max_pixels: int = 150000
    min_group_frames: int = 8
    max_group_frames: int = 64
    spatial_mask_mode: str = "off"
    cache_root: Path = field(default_factory=lambda: Path(
        os.getenv(
            "ONLINE_CODEC_CACHE_DIR",
            os.path.join(
                os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "online_codec",
            ),
        )
    ))
    timeout_seconds: int = 7200

    # ---- codec engine selection ------------------------------------------
    # "hevc"    : HEVC/h264 bit-cost readiness (external cv-preinfer binary; the
    #             default). "cv-preinfer" is accepted as a legacy alias.
    # "dcvc-rt" : retained metadata only; execution explicitly rejected.
    # Selectable per call via ``codec_config={"engine": "dcvc-rt", ...}``.
    engine: str = "hevc"

    # DCVC-RT knobs, nested. In the model config / codec_config this is a plain
    # dict (``"dcvc": {"qp": 21, ...}``); __post_init__ coerces it to DcvcConfig.
    dcvc: "DcvcConfig" = field(default_factory=DcvcConfig)

    def __post_init__(self):
        if isinstance(self.dcvc, dict):
            # Filter to known DcvcConfig fields — preprocessor_config.json's
            # codec.dcvc also carries selection knobs (per_frame_cap_ratio,
            # bottom_atten, threshold_scale, ...) read directly by the readiness
            # pipeline via codec_dcvc_config, which are not DcvcConfig fields.
            _known = {f.name for f in _dc_fields(DcvcConfig)}
            self.dcvc = DcvcConfig(**{k: v for k, v in self.dcvc.items() if k in _known})

    def validate(self) -> None:
        if self.target_canvas <= 0:
            raise ValueError("CodecConfig.target_canvas must be > 0")
        if self.target_canvas % self.images_per_group != 0:
            raise ValueError(
                "CodecConfig.target_canvas must be divisible by images_per_group"
            )
        if self.group_size % self.images_per_group != 0:
            raise ValueError(
                "CodecConfig.group_size must be divisible by images_per_group"
            )

    def num_sampled_frames(self) -> int:
        return (self.target_canvas // self.images_per_group) * self.group_size


# ---------------------------------------------------------- text/position

def _format_timestamp(seconds: float, decimals: int) -> str:
    return f"<{seconds:.{decimals}f} seconds>"


def convert_positions_to_block_layout(
    positions: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Reorder a (T*H*W, 3) patch position table into 2D-MRoPE block layout."""
    sms = int(spatial_merge_size)
    if sms == 1:
        return positions
    total = int(t) * int(h) * int(w)
    indices = torch.arange(total, device=positions.device).view(t, h, w)
    h_m, w_m = int(h) // sms, int(w) // sms
    indices = (
        indices.view(t, h_m, sms, w_m, sms)
        .permute(0, 1, 3, 2, 4).contiguous().view(total)
    )
    return positions[indices]


def codec_positions_for_processor(
    src_positions: np.ndarray, image_grid_thw: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    positions = torch.from_numpy(src_positions).long().to(device)
    expected_total = int(image_grid_thw.prod(dim=1).sum().item())
    if expected_total != positions.shape[0]:
        raise ValueError(
            "codec patch position length mismatch: "
            f"thw_total={expected_total}, positions={positions.shape[0]}"
        )
    chunks, offset = [], 0
    for row in image_grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        n = t * h * w
        chunks.append(convert_positions_to_block_layout(positions[offset: offset + n], t, h, w))
        offset += n
    return torch.cat(chunks, dim=0)


def _timestamp_runs(
    patch_positions: torch.Tensor, fps: float, decimals: int, spatial_merge_size: int = 2,
) -> list[tuple[str, int]]:
    t_values = patch_positions[:, 0]
    unique_t, counts = torch.unique_consecutive(t_values, return_counts=True)
    merge_factor = int(spatial_merge_size) ** 2
    runs = []
    for t_val, count in zip(unique_t.tolist(), counts.tolist()):
        if int(t_val) < 0:
            continue
        token_count = int(count) // merge_factor
        if token_count <= 0:
            continue
        runs.append((_format_timestamp(float(t_val) / float(fps), decimals), token_count))
    return runs


def rewrite_text_with_codec_positions(
    text: str, patch_positions: torch.Tensor, fps: float, decimals: int,
) -> str:
    """Replace the vision span in a chat-template string with codec-aware tokens."""
    parts = []
    for timestamp, token_count in _timestamp_runs(patch_positions, fps, decimals):
        parts.extend([timestamp, VISION_START, IMAGE_PAD * token_count, VISION_END, "\n"])
    vision_text = "".join(parts)
    match = _VIDEO_BLOCK_RE.search(text)
    if match is None:
        return text
    tail_start = match.end()
    if tail_start < len(text) and text[tail_start] == "\n":
        tail_start += 1
    return text[:match.start()] + vision_text + text[tail_start:]


def drop_padding_canvases(
    images: list[Image.Image], src_positions: np.ndarray,
) -> tuple[list[Image.Image], np.ndarray, int]:
    """Drop fully-padding canvases (all-negative timestamps) and their patches."""
    n_canvas = len(images)
    if n_canvas == 0:
        return images, src_positions, 0
    total_patches = src_positions.shape[0]
    if total_patches % n_canvas != 0:
        raise ValueError(
            f"src_positions length {total_patches} not divisible by canvas count {n_canvas}"
        )
    ppc = total_patches // n_canvas
    positions = src_positions.reshape(n_canvas, ppc, 3)
    canvas_t = positions[..., 0]
    keep_mask = (canvas_t >= 0).any(axis=1)
    if bool((keep_mask & ~((canvas_t >= 0).all(axis=1))).any()):
        raise ValueError("encountered half-padding canvas; padding is expected to be canvas-granular")
    dropped = int(n_canvas - int(keep_mask.sum()))
    if dropped == 0:
        return images, src_positions, 0
    kept_images = [img for img, keep in zip(images, keep_mask.tolist()) if keep]
    kept_positions = positions[keep_mask].reshape(-1, 3)
    return kept_images, kept_positions, dropped


# ------------------------------------------------------- cv-preinfer driver

def _get_video_total_frames(video_url: str) -> int:
    import cv2  # type: ignore[import-not-found]
    cap = cv2.VideoCapture(video_url)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return max(1, total)


def _cache_dir_for(video_url: str, cfg: CodecConfig) -> Path:
    raw = (
        f"{video_url}|eng={cfg.engine}|tc={cfg.target_canvas}|gs={cfg.group_size}"
        f"|ipg={cfg.images_per_group}|patch={cfg.patch}"
        f"|mp={cfg.max_pixels}|mask={cfg.spatial_mask_mode}"
    )
    key = hashlib.md5(raw.encode()).hexdigest()
    return cfg.cache_root / f"{Path(video_url).stem}_{key}"


def _load_codec_result(out_dir: Path) -> dict:
    with open(out_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    canvas_files = meta.get("canvas_files")
    if not canvas_files:
        for ext in ("npy", "jpg", "png"):
            hits = sorted(p.name for p in out_dir.glob(f"canvas_*.{ext}"))
            if hits:
                canvas_files = hits
                break
        canvas_files = canvas_files or []
    images = []
    for name in canvas_files:
        fp = out_dir / name
        if name.endswith(".npy"):
            images.append(Image.fromarray(np.load(fp)))
        else:
            images.append(Image.open(fp).convert("RGB"))
    src_positions = np.load(out_dir / "src_patch_position.npy")
    fps = float(meta.get("fps") or 30.0)
    return {"images": images, "src_positions": src_positions, "fps": fps,
            "out_dir": str(out_dir), "meta": meta}


def _codec_binary() -> str:
    binary = shutil.which("cv-preinfer")
    if binary is None:
        raise RuntimeError(
            "Mage-VL traditional HEVC preprocessing needs installed codec-video-prep "
            "with cv-preinfer on PATH. Alternatively use predecoded frames "
            "(video_backend='frames'). Neural DCVC is not qualified."
        )
    return binary


def _run_cv_preinfer(video_url: str, out_dir: Path, cfg: CodecConfig) -> dict:
    bin_name = _codec_binary()
    tmp_dir = Path(tempfile.mkdtemp(dir=str(cfg.cache_root), prefix=f".tmp_{out_dir.name[:48]}_"))
    num_sampled = min(cfg.num_sampled_frames(), _get_video_total_frames(video_url))
    cmd = [
        bin_name, "--video", video_url, "--out_dir", str(tmp_dir),
        "--num_sampled_frames", str(num_sampled),
        "--grouping_mode", "readiness",
        "--group_size", str(cfg.group_size),
        "--images_per_group", str(cfg.images_per_group),
        "--patch", str(cfg.patch),
        "--max_pixels", str(cfg.max_pixels),
        "--readiness_sum_threshold", "0",
        "--min_group_frames", str(cfg.min_group_frames),
        "--max_group_frames", str(cfg.max_group_frames),
        "--avoid_keyframes",
        "--canvas_format", "jpg",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=cfg.timeout_seconds)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-2000:]
            raise RuntimeError(f"online codec failed rc={result.returncode}: {detail}")
        _load_codec_result(tmp_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return _load_codec_result(out_dir)


# ------------------------------------------------------- dcvc-rt driver
def _run_dcvc_rt(video_url: str, out_dir: Path, cfg: CodecConfig) -> dict:
    raise NotImplementedError(
        "Mage-VL neural DCVC preprocessing is not qualified in GPT-QModel. "
        "No checkpoint Python or bundled scripts are executed. Use predecoded frames "
        "(video_backend='frames') or install codec-video-prep for traditional HEVC preprocessing."
    )


def process_codec_video(video_url: str, cfg: CodecConfig) -> dict:
    """Public entrypoint: video URL + config -> dict(images, src_positions, fps, ...).

    Result is cached on disk under ``cfg.cache_root``; concurrent workers
    coordinate via a flock-protected sentinel.

    Soft-warning behaviour (B-mode):
      - If the video has fewer frames than needed to fill ``target_canvas``,
        we emit a one-time UserWarning describing the shortfall but proceed
        normally (cv-preinfer will produce fewer canvases than requested).
      - If the video is so short that cv-preinfer cannot form a single
        group (``< min_group_frames``), we emit a clearer warning and let
        cv-preinfer's own error propagate.
    """
    cfg.validate()
    if cfg.engine == "dcvc-rt":
        return _run_dcvc_rt(video_url, cfg.cache_root, cfg)
    out_dir = _cache_dir_for(video_url, cfg)
    if (out_dir / "meta.json").exists() and (out_dir / "src_patch_position.npy").exists():
        return _load_codec_result(out_dir)

    _codec_binary()
    _maybe_warn_short_video(video_url, cfg)

    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.cache_root / f".{out_dir.name}.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if (out_dir / "meta.json").exists() and (out_dir / "src_patch_position.npy").exists():
            return _load_codec_result(out_dir)
        if cfg.engine in ("hevc", "cv-preinfer"):
            return _run_cv_preinfer(video_url, out_dir, cfg)
        raise ValueError(
            f"unknown codec engine: {cfg.engine!r} (use 'hevc' or 'dcvc-rt')"
        )
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _maybe_warn_short_video(video_url: str, cfg: CodecConfig) -> None:
    """Soft-warn (B-mode) when a video is too short to fill target_canvas.

    Logic:
      * needed_frames  = num_sampled_frames() = (target_canvas/ipg)*group_size
      * usable_frames  = min(needed_frames, total_frames)
      * expected_canv  = (usable_frames // group_size) * images_per_group
    If ``expected_canv < target_canvas`` we warn. If
    ``total_frames < min_group_frames`` we warn more loudly (cv-preinfer
    will fail downstream and that error is allowed to propagate).
    """
    try:
        total_frames = _get_video_total_frames(video_url)
    except Exception:
        return  # don't fail on probe errors; cv-preinfer will report its own

    needed = cfg.num_sampled_frames()
    usable = min(needed, total_frames)
    expected_canv = (usable // cfg.group_size) * cfg.images_per_group

    if total_frames < cfg.min_group_frames:
        warnings.warn(
            f"[codec] video {video_url!r} has only {total_frames} frames "
            f"(< min_group_frames={cfg.min_group_frames}); cv-preinfer cannot "
            f"form even a single group and will error out. Consider lowering "
            f"min_group_frames or using video_backend='frames' for this clip.",
            UserWarning,
            stacklevel=2,
        )
        return

    if expected_canv < cfg.target_canvas:
        warnings.warn(
            f"[codec] video {video_url!r} has {total_frames} frames; with "
            f"group_size={cfg.group_size}, images_per_group={cfg.images_per_group} "
            f"this yields ~{expected_canv} canvas(es) instead of the requested "
            f"target_canvas={cfg.target_canvas}. Inference will proceed with the "
            f"smaller canvas count.",
            UserWarning,
            stacklevel=2,
        )


# ----------------------------------------------------- processor wiring

def codec_image_processor_outputs(
    image_processor, images: list[Image.Image], max_pixels: int,
) -> dict:
    """Run ``Qwen2VLImageProcessor`` on codec canvases without smart_resize-ing.

    The codec emits canvases already aligned to the patch grid. To keep
    ``image_grid_thw`` consistent with ``src_patch_position``:
      - ``max_pixels`` is clamped up to the largest canvas (never shrinks)
      - ``min_pixels`` is clamped down to the smallest canvas (never upscales)

    Without the ``min_pixels`` clamp, ``Qwen2VLImageProcessor``'s default
    ``min_pixels=200704`` would grow any canvas below that threshold,
    producing extra patches and a chunk/index mismatch downstream.
    """
    canvas_pixels = [im.width * im.height for im in images]
    proc_max = max(int(max_pixels), max(canvas_pixels, default=int(max_pixels)))
    proc_min = min(canvas_pixels) if canvas_pixels else 1
    return image_processor(
        images=images, min_pixels=proc_min, max_pixels=proc_max, return_tensors="pt",
    )


__all__ = [
    "CodecConfig",
    "process_codec_video",
    "drop_padding_canvases",
    "codec_positions_for_processor",
    "rewrite_text_with_codec_positions",
    "codec_image_processor_outputs",
    "VISION_START", "VISION_END", "IMAGE_PAD",
]
