# SPDX-FileCopyrightText: 2026 Microsoft
# SPDX-License-Identifier: Apache-2.0
# Adapted from microsoft/Mage-VL@d88b153285f1633a61b2f693c59c8576693af185.
"""Mage-VL multi-modal processor.

Combines:
  - installed ``Qwen2VLImageProcessorPil``
  - ``MageVLVideoProcessor``  (this checkpoint, video_processing_*)
  - supplied normalized tokenizer or installed ``Qwen2Tokenizer``
  - ``chat_template.jinja``            (existing, emits <|video_pad|>)

Public API:
    proc = MageVLProcessor(image_processor, tokenizer, video_processor)
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=[text], videos=[mp4_or_frames], return_tensors="pt")
    out = model.generate(**inputs)

Design choices:
  - Video path is "in-processor, transformed to multi-image + per-frame
    timestamps" — model.forward sees the image path only.
  - The chat_template's <|vision_start|><|video_pad|><|vision_end|> placeholder
    is rewritten in __call__ to per-frame blocks:
        <X.X seconds><|vision_start|><|image_pad|>*n<|vision_end|>\n
  - We DO NOT emit `second_per_grid_ts`; see plan §0.5.
  - Backward-compatible: `images=...` / pure-text usage matches the existing
    Qwen2_5_VLProcessor output.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

import torch


# Special-token strings used by the checkpoint's tokenizer / chat_template.
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

# Checkpoint metadata describes the input representation. Execution policy,
# including cache locations and timeouts, is supplied by the caller.
_CHECKPOINT_CODEC_KEYS = frozenset({
    "target_canvas", "group_size", "images_per_group", "patch", "max_pixels",
    "min_group_frames", "max_group_frames", "spatial_mask_mode",
})


def _format_seconds_tag(seconds: float) -> str:
    """Match training format: ``<X.X seconds>`` (one decimal place)."""
    return f"<{float(seconds):.1f} seconds>"


def _expand_video_block_for_frames(
    n_per_frame: int,
    frame_seconds: Sequence[float],
) -> str:
    """Build the per-frame expanded text that replaces a single
    ``<|vision_start|><|video_pad|><|vision_end|>`` block.

    Output (one block per frame, newline-separated):
        ``<X.X seconds><|vision_start|><|image_pad|>*n_per_frame<|vision_end|>\\n``
    """
    parts: List[str] = []
    for sec in frame_seconds:
        parts.append(_format_seconds_tag(sec))
        parts.append(VISION_START)
        parts.append(IMAGE_PAD * n_per_frame)
        parts.append(VISION_END)
    return "".join(parts)


class MageVLProcessor:
    """Native multi-modal processor for MageVL.

    Selected directly by the Mage-VL definition, without checkpoint auto classes.
    """

    attributes = ["image_processor", "video_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template: Optional[str] = None,
        codec_config: Optional[dict] = None,
    ):
        if tokenizer is None:
            raise ValueError("Mage-VL processor requires a tokenizer.")
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor

        # Inherit chat_template from the tokenizer if not given (matches Qwen2_5_VLProcessor).
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        self.chat_template = chat_template

        # Cache the merge size from image_processor for token-count math.
        self.spatial_merge_size = int(
            getattr(image_processor, "merge_size", 2) if image_processor is not None else 2
        )

        # Codec config defaults (overridden per-call via ``codec_config=``).
        self._codec_config_defaults: dict = dict(codec_config or {})

    # ------------------------------------------------------------------ utils

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        import json

        from transformers import Qwen2Tokenizer, Qwen2VLImageProcessorPil
        from transformers.utils.hub import cached_file

        from .video_processing_mage_vl import MageVLVideoProcessor

        kwargs.pop("_from_auto", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        kwargs.pop("config", None)
        tokenizer = kwargs.pop("tokenizer", None)
        codec_override = kwargs.pop("codec_config", None)
        image_processor = Qwen2VLImageProcessorPil.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            tokenizer = Qwen2Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        hub_kwargs = {k: v for k, v in kwargs.items()
                      if k in {"cache_dir", "revision", "token", "local_files_only", "subfolder", "force_download"}}
        config_path = cached_file(pretrained_model_name_or_path, "preprocessor_config.json", **hub_kwargs)
        if config_path is None:
            raise FileNotFoundError("Mage-VL requires preprocessor_config.json.")
        with open(config_path, encoding="utf-8") as source:
            codec_defaults = json.load(source).get("codec")
        codec_defaults = {
            key: codec_defaults[key] for key in _CHECKPOINT_CODEC_KEYS
            if isinstance(codec_defaults, dict) and key in codec_defaults
        }
        video_path = cached_file(pretrained_model_name_or_path, "video_preprocessor_config.json",
                                 _raise_exceptions_for_missing_entries=False, **hub_kwargs)
        video_kwargs = {}
        if video_path is not None:
            with open(video_path, encoding="utf-8") as source:
                video_config = json.load(source)
            for key in ("max_frames", "fixed_num_frames", "target_fps", "resize_frames"):
                if key in video_config:
                    video_kwargs[key] = video_config[key]
        video_processor = MageVLVideoProcessor(
            image_processor=image_processor,
            min_pixels=image_processor.size["shortest_edge"],
            max_pixels=image_processor.size["longest_edge"],
            patch_size=image_processor.patch_size,
            spatial_merge_size=image_processor.merge_size,
            temporal_patch_size=image_processor.temporal_patch_size,
            **video_kwargs,
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor,
                   codec_config=codec_defaults if codec_override is None else codec_override)

    def save_pretrained(self, save_directory, **kwargs):
        import json
        from pathlib import Path

        if self.image_processor is None:
            raise ValueError("Saving a Mage-VL processor requires an image processor.")
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path, **kwargs)
        self.image_processor.save_pretrained(path)
        config_path = path / "preprocessor_config.json"
        config = json.loads(config_path.read_text())
        config["processor_class"] = type(self).__name__
        config["codec"] = {
            key: value for key, value in self._codec_config_defaults.items()
            if key in _CHECKPOINT_CODEC_KEYS
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        if self.video_processor is not None:
            video_config = {key: getattr(self.video_processor, key)
                            for key in ("max_frames", "fixed_num_frames", "target_fps", "resize_frames")}
            (path / "video_preprocessor_config.json").write_text(json.dumps(video_config, indent=2) + "\n")
        if self.chat_template:
            (path / "chat_template.jinja").write_text(self.chat_template)


    # ------------------------------------------------------------- chat helpers

    def apply_chat_template(self, messages, **kwargs):
        """Delegate to the tokenizer (which already has ``chat_template``)."""
        if self.chat_template and "chat_template" not in kwargs:
            kwargs["chat_template"] = self.chat_template
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    # ----------------------------------------------------------- main __call__

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images=None,
        videos=None,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = False,
        num_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
        # Codec video backend (in-processor codec preprocessing). When
        # ``video_backend="codec"`` and ``videos`` is set, the codec pipeline
        # (cv-preinfer) replaces the frame-sampling VideoProcessor. The codec
        # canvas pixel budget is taken from ``max_pixels`` so the user only
        # configures one pixel knob.
        video_backend: str = "frames",
        max_pixels: Optional[int] = None,
        codec_config: Optional[dict] = None,
        **kwargs,
    ):
        """Process an aligned (text, images, videos) batch.

        Behaviour:
          * ``videos is not None``: run the VideoProcessor, rewrite each
            ``<|video_pad|>`` block in ``text`` to per-frame ``<X.X seconds>``
            blocks, then alias the video patches as ``pixel_values`` /
            ``image_grid_thw`` so the model's image path consumes them.
          * ``images is not None``: passed through to the underlying
            ``image_processor``. (May coexist with ``videos``; expansion order
            in the prompt is determined by the chat_template / placeholders.)
          * Pure text: tokenize and return.

        Per-call frame-sampling overrides (apply only to ``videos`` path; do
        not mutate the underlying VideoProcessor's defaults):
          * ``num_frames``  : force exactly N frames per video
            (alias of ``fixed_num_frames``).
          * ``max_frames``  : cap on auto-selected frame count (long videos).
          * ``target_fps``  : sample at this FPS (capped by ``max_frames``).

        Returns a ``BatchFeature`` with at minimum ``input_ids`` and
        ``attention_mask``; plus ``pixel_values`` / ``image_grid_thw`` /
        ``patch_positions`` when visuals are present.
        """
        if text is None:
            raise ValueError("`text` is required.")
        if isinstance(text, str):
            text = [text]
        text = list(text)

        out: dict = {}

        # ---------------- VISUAL PLACEHOLDER ORDER ----------------
        # Record the prompt order of every visual placeholder *before* any
        # rewriting, so the visual tensors can be concatenated in exactly the
        # order the model consumes the <|image_pad|> slots.
        _pad_re = re.compile(re.escape(IMAGE_PAD) + "|" + re.escape(VIDEO_PAD))
        order_per_text = [
            ["image" if m.group(0) == IMAGE_PAD else "video" for m in _pad_re.finditer(s)]
            for s in text
        ]

        # Per-visual (pixel_values, image_grid_thw, patch_positions) triples,
        # kept separate until the final prompt-order assembly.
        image_slots: List[tuple] = []
        video_slots: List[tuple] = []

        # ---------------- IMAGE PATH ----------------
        # Runs *before* the video paths: at this point every <|image_pad|> in
        # `text` still belongs to a real image, because video placeholders are
        # still <|video_pad|>. Running it afterwards would let the image
        # expansion consume placeholders the video rewrite had just emitted.
        if images is not None:
            if self.image_processor is None:
                raise ValueError("images passed but no image_processor configured.")
            image_outputs = self.image_processor(images=images, return_tensors="pt")
            image_grid_thw = image_outputs["image_grid_thw"]

            # Expand each <|image_pad|> placeholder to the number of merged tokens.
            sms = self.spatial_merge_size
            merge_factor = sms * sms
            image_token_counts = (
                (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
                // merge_factor
            ).tolist()
            img_idx = 0

            def _expand_image_pads(s: str) -> str:
                nonlocal img_idx
                while IMAGE_PAD in s:
                    if img_idx >= len(image_token_counts):
                        break
                    n = int(image_token_counts[img_idx])
                    s = s.replace(IMAGE_PAD, "<|placeholder|>" * n, 1)
                    img_idx += 1
                return s.replace("<|placeholder|>", IMAGE_PAD)

            text = [_expand_image_pads(s) for s in text]

            from .video_processing_mage_vl import build_patch_positions
            image_pp = build_patch_positions(image_grid_thw, spatial_merge_size=sms)
            offset = 0
            for row in image_grid_thw:
                n = int(row[0]) * int(row[1]) * int(row[2])
                image_slots.append((
                    image_outputs["pixel_values"][offset: offset + n],
                    row.unsqueeze(0),
                    image_pp[offset: offset + n],
                ))
                offset += n

        # ---------------- CODEC VIDEO BACKEND ----------------
        # Codec path: replaces the frame-sampling VideoProcessor entirely.
        # Each video -> N canvases + src_patch_position; we rewrite the
        # <|vision_start|>...<|vision_end|> span in `text` based on the codec
        # patch_positions (one canvas worth of <|image_pad|>s per timestamp).
        if videos is not None and str(video_backend).lower() == "codec":
            from .codec_video_processing_mage_vl import (
                CodecConfig,
                codec_image_processor_outputs,
                codec_positions_for_processor,
                drop_padding_canvases,
                process_codec_video,
                rewrite_text_with_codec_positions,
            )

            # Normalise to list[video_url].
            if isinstance(videos, str):
                videos_list = [videos]
            else:
                videos_list = list(videos)

            # Build effective codec config: defaults < class-level < per-call.
            cfg_kwargs = dict(self._codec_config_defaults)
            if codec_config:
                merged_dcvc = dict(cfg_kwargs.get("dcvc") or {})
                merged_dcvc.update(codec_config.get("dcvc") or {})
                cfg_kwargs.update(codec_config)
                if merged_dcvc:
                    cfg_kwargs["dcvc"] = merged_dcvc
            # Unify pixel budget with image_processor.
            effective_max_pixels = int(
                max_pixels
                if max_pixels is not None
                else cfg_kwargs.get("max_pixels", getattr(self.image_processor, "max_pixels", 150000))
            )
            cfg_kwargs["max_pixels"] = effective_max_pixels
            cfg = CodecConfig(**cfg_kwargs)

            all_pixel_values, all_grid_thw, all_patch_positions = [], [], []
            rewritten_texts = list(text)
            if len(rewritten_texts) != len(videos_list):
                if len(rewritten_texts) == 1 and len(videos_list) >= 1:
                    rewritten_texts = rewritten_texts * len(videos_list)
                    order_per_text = order_per_text * len(videos_list)
                else:
                    raise ValueError(
                        f"codec video backend: got {len(rewritten_texts)} texts but {len(videos_list)} videos"
                    )

            for idx, video_url in enumerate(videos_list):
                payload = process_codec_video(video_url, cfg)
                imgs, src_positions, _ = drop_padding_canvases(
                    payload["images"], payload["src_positions"]
                )
                if not imgs:
                    raise RuntimeError(f"codec produced no usable canvases for {video_url}")
                image_data = codec_image_processor_outputs(
                    self.image_processor, imgs, max_pixels=effective_max_pixels
                )
                image_grid_thw = image_data["image_grid_thw"]
                patch_positions = codec_positions_for_processor(
                    src_positions, image_grid_thw, device=image_grid_thw.device,
                )
                rewritten_texts[idx] = rewrite_text_with_codec_positions(
                    rewritten_texts[idx], patch_positions,
                    fps=float(payload["fps"]), decimals=1,
                )
                all_pixel_values.append(image_data["pixel_values"])
                all_grid_thw.append(image_grid_thw)
                all_patch_positions.append(patch_positions)

            video_slots.extend(zip(all_pixel_values, all_grid_thw, all_patch_positions))
            text = rewritten_texts
            # Codec branch handled the video. Suppress the frame-sampling block below.
            videos = None

        # ---------------- VIDEO PATH ----------------
        # Process videos first so we can rewrite their placeholders into the
        # text before tokenization.
        video_outputs = None
        if videos is not None:
            if self.video_processor is None:
                raise ValueError("videos passed but no video_processor configured.")
            # Normalise to a list of videos.
            if isinstance(videos, (str,)):
                videos_list = [videos]
            elif isinstance(videos, list) and len(videos) > 0 and not isinstance(
                videos[0], (list, str)
            ):
                # list[PIL]/[np.ndarray] = single video
                videos_list = [videos]
            else:
                videos_list = list(videos)

            # Per-call sampling overrides: temporarily swap the
            # VideoProcessor's attributes, then restore. Lets users do
            #     processor(videos=[mp4], num_frames=8)
            # without mutating processor.video_processor.
            from copy import copy

            vp = copy(self.video_processor)
            if num_frames is not None:
                vp.fixed_num_frames = int(num_frames)
            if max_frames is not None:
                vp.max_frames = int(max_frames)
            if target_fps is not None:
                vp.target_fps = float(target_fps)
            video_outputs = vp(videos=videos_list, return_tensors="pt")

            # Rewrite each <|video_pad|> in `text` into per-frame blocks.
            video_grid_thw = video_outputs["video_grid_thw"]  # [num_videos, 3]
            frame_timestamps = video_outputs["frame_timestamps"]
            sms = self.spatial_merge_size

            # We iterate placeholders globally across all texts (matching how
            # Qwen2_5_VLProcessor sources `image_grid_thw` rows).
            video_idx = 0

            def _rewrite_one_text(s: str) -> str:
                nonlocal video_idx
                pattern = re.compile(
                    re.escape(VISION_START) + r"\s*" + re.escape(VIDEO_PAD) + r"\s*" + re.escape(VISION_END)
                )

                def _sub(_match):
                    nonlocal video_idx
                    if video_idx >= video_grid_thw.shape[0]:
                        raise ValueError(
                            "More <|video_pad|> placeholders in text than videos provided."
                        )
                    T_eff = int(video_grid_thw[video_idx, 0].item())
                    H_p = int(video_grid_thw[video_idx, 1].item())
                    W_p = int(video_grid_thw[video_idx, 2].item())
                    n_per_frame = (H_p * W_p) // (sms * sms)
                    frame_seconds = frame_timestamps[video_idx]
                    if len(frame_seconds) != T_eff:
                        # Defensive: pad/truncate so the count matches the grid.
                        if len(frame_seconds) < T_eff:
                            frame_seconds = list(frame_seconds) + [
                                frame_seconds[-1] if frame_seconds else 0.0
                            ] * (T_eff - len(frame_seconds))
                        else:
                            frame_seconds = list(frame_seconds[:T_eff])
                    expanded = _expand_video_block_for_frames(
                        n_per_frame, frame_seconds
                    )
                    video_idx += 1
                    # Strip trailing newline so we don't double-newline existing prompts.
                    return expanded.rstrip("\n")

                return pattern.sub(_sub, s)

            text = [_rewrite_one_text(s) for s in text]

            if video_idx != video_grid_thw.shape[0]:
                raise ValueError(
                    f"Provided {video_grid_thw.shape[0]} videos but only "
                    f"{video_idx} <|video_pad|> placeholders were found in text."
                )

            # Alias video tensors into the image path (NEW model only consumes the image path).
            # Option 1 (multi-image semantics, training-aligned): expand each
            # video_grid_thw row [T, H, W] into T rows of [1, H, W]. The
            # pixel_values rows are already laid out frame-by-frame (T*H*W per
            # video, with temporal_patch_size=1), so this row-expansion of
            # image_grid_thw is the only adjustment needed for the model's
            # forward to treat each frame as a separate image (matching the
            # multi-image inference path).
            vgthw = video_outputs["video_grid_thw"]
            video_pv = video_outputs["pixel_values_videos"]
            video_pp = video_outputs["patch_positions"]
            offset = 0
            for row in vgthw:
                T_v, H_v, W_v = int(row[0]), int(row[1]), int(row[2])
                n = T_v * H_v * W_v
                video_slots.append((
                    video_pv[offset: offset + n],
                    torch.tensor([[1, H_v, W_v]] * T_v, dtype=vgthw.dtype),
                    video_pp[offset: offset + n],
                ))
                offset += n

        # ---------------- ASSEMBLE VISUALS IN PROMPT ORDER ----------------
        # The model consumes <|image_pad|> slots in prompt order and
        # pixel_values in row order, so the visual tensors must be stacked in
        # the order their placeholders appear in the prompt.
        if image_slots or video_slots:
            flat_order = [kind for per_text in order_per_text for kind in per_text]
            n_img = sum(k == "image" for k in flat_order)
            n_vid = sum(k == "video" for k in flat_order)
            if n_img == len(image_slots) and n_vid == len(video_slots):
                img_it, vid_it = iter(image_slots), iter(video_slots)
                ordered = [next(img_it) if k == "image" else next(vid_it) for k in flat_order]
            else:
                raise ValueError("Mage-VL visual placeholders do not match the supplied images/videos.")
            out["pixel_values"] = torch.cat([s[0] for s in ordered], dim=0)
            out["image_grid_thw"] = torch.cat([s[1] for s in ordered], dim=0)
            out["patch_positions"] = torch.cat([s[2] for s in ordered], dim=0)


        # ---------------- VIDEO PATH FINAL EXPANSION ----------------
        # When `videos` was given (and possibly without `images`), the per-frame
        # rewrite above already produced runs of <|image_pad|> that need to be
        # treated like image placeholders (one per merged token). Because the
        # rewrite directly emits ``IMAGE_PAD * n_per_frame``, the texts are
        # already in their tokenize-ready form for the video portion. So nothing
        # more to do here — fall through to tokenize.

        # ---------------- TOKENIZE ----------------
        encoding = self.tokenizer(
            text,
            padding=padding,
            return_tensors=return_tensors,
            **{k: v for k, v in kwargs.items() if k in (
                "max_length", "truncation", "add_special_tokens",
                "return_attention_mask", "return_token_type_ids",
            )},
        )
        out["input_ids"] = encoding["input_ids"]
        out["attention_mask"] = encoding.get(
            "attention_mask",
            torch.ones_like(encoding["input_ids"]),
        )

        try:
            from transformers.feature_extraction_utils import BatchFeature

            return BatchFeature(data=out)
        except Exception:
            return out

    # ---------------------------------------------------------------- decoding

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


__all__ = [
    "MageVLProcessor",
    "VISION_START",
    "VISION_END",
    "IMAGE_PAD",
    "VIDEO_PAD",
]
