# Copyright (C) 2025 AIDC-AI
# SPDX-License-Identifier: Apache-2.0
# Source prompt/media processing adapted for direct native asset loading; see NOTICE.
"""Ovis2.5 chat/media preparation without checkpoint Python dispatch."""

import os
from typing import Dict, List, Union

import torch
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from ..native.siglip.image_processing_pil_siglip import SiglipImageProcessorPil
from .configuration_ovis2_5 import Ovis2_5_Config
from .vision_ovis2_5 import (
    IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER_ID,
    INDICATOR_IDS,
    VIDEO_PLACEHOLDER,
    VIDEO_PLACEHOLDER_ID,
    VISUAL_ATOM_ID,
    preprocess_media,
)


class Ovis2_5Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor, tokenizer, config):
        # The components are explicitly selected; ProcessorMixin's registry discovery is not used.
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.config = config
        self.chat_template = tokenizer.chat_template

    @classmethod
    def from_pretrained(cls, model_id_or_path, config=None, tokenizer=None, **kwargs):
        selected = {k: kwargs[k] for k in ("cache_dir", "revision", "token", "local_files_only", "subfolder", "force_download")
                    if k in kwargs}
        config = config or Ovis2_5_Config.from_pretrained(model_id_or_path, **selected)
        if tokenizer is None:
            from ....utils.hf import load_hf_tokenizer
            tokenizer = load_hf_tokenizer(model_id_or_path, trust_remote_code=False, config=config, **selected)
        image = SiglipImageProcessorPil.from_pretrained(model_id_or_path, **selected)
        return cls(image, tokenizer, config)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.image_processor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        # A standalone processor needs the merge stride and image budgets on reload.
        # Leave a model writer's existing (possibly quantized) config untouched.
        if not os.path.exists(os.path.join(save_directory, "config.json")):
            self.config.save_pretrained(save_directory)
        return []

    def __call__(self, messages=None, return_tensors="pt", **kwargs):
        if return_tensors != "pt":
            raise ValueError("Ovis2.5 processor returns PyTorch tensors")
        ids, pixels, grids = self.preprocess_inputs(messages, **kwargs)
        result = {"input_ids": ids, "attention_mask": ids.ne(self.tokenizer.pad_token_id)}
        if pixels is not None:
            result.update(pixel_values=pixels, grid_thws=grids)
        return BatchFeature(result)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if tokenize:
            return self(messages=messages, **kwargs)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

    def _merge_inputs(
        self, raw_input_ids, placeholder_id, grid_thws, indicator_begin_id, indicator_end_id
    ):
        if len([v for v in raw_input_ids if v == placeholder_id]) != len(grid_thws):
            raise ValueError("Ovis2.5 media count does not match prompt placeholders")
        input_ids = []
        prev_index = 0
        placeholder_indexes = [i for i, v in enumerate(raw_input_ids) if v == placeholder_id]
        for placeholder_index, grid_thw in zip(placeholder_indexes, grid_thws):
            input_ids.extend(raw_input_ids[prev_index:placeholder_index])
            num_image_atoms = grid_thw.prod().item()
            num_image_atoms //= self.config.vit_config.hidden_stride ** 2
            num_image_atoms //= self.config.vit_config.temporal_patch_size
            input_ids.extend([indicator_begin_id] + [VISUAL_ATOM_ID] * num_image_atoms + [indicator_end_id])
            prev_index = placeholder_index + 1
        input_ids.extend(raw_input_ids[prev_index:])
        return input_ids

    def _tokenize_with_visual_placeholder(self, text):
        placeholder = VIDEO_PLACEHOLDER if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER
        placeholder_id = VIDEO_PLACEHOLDER_ID if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER_ID
        chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in text.split(placeholder)]
        input_ids = chunks[0]
        for chunk in chunks[1:]:
            input_ids.append(placeholder_id)
            input_ids.extend(chunk)
        return input_ids

    def preprocess_inputs(
        self,
        messages: List[Union[str, Dict]],
        min_pixels=None,
        max_pixels=None,
        add_generation_prompt=True,
        enable_thinking=False
    ):
        max_pixels = self.config.max_pixels if max_pixels is None else max_pixels
        min_pixels = self.config.min_pixels if min_pixels is None else min_pixels
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking
        )
        input_ids = self._tokenize_with_visual_placeholder(text)
        images = []
        videos = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                images.extend([item["image"] for item in content if item.get("image") is not None])
                videos.extend([item["video"] for item in content if item.get("video") is not None])
        if images and videos:
            raise ValueError(
                "Multiple visual input data types detected (both image and video provided). "
                "This model supports only one type of visual input data at a time. "
                "Please provide either image or video, but not both."
            )

        pixel_values, grid_thws = None, None
        if images:
            pixel_values, grid_thws = zip(
                *(preprocess_media(self.image_processor, self.config.vit_config, image=image, min_pixels=min_pixels, max_pixels=max_pixels)
                  for image in images)
            )
            input_ids = self._merge_inputs(
                input_ids, IMAGE_PLACEHOLDER_ID, grid_thws, INDICATOR_IDS[0], INDICATOR_IDS[1]
            )
            pixel_values = torch.cat(pixel_values, dim=0)
            grid_thws = torch.cat(grid_thws, dim=0)
        elif videos:
            assert len(videos) == 1, "only support single video"
            pixel_values, grid_thws = preprocess_media(self.image_processor, self.config.vit_config,
                video=videos[0], min_pixels=min_pixels, max_pixels=max_pixels
            )
            input_ids = self._merge_inputs(
                input_ids, VIDEO_PLACEHOLDER_ID, grid_thws, INDICATOR_IDS[2], INDICATOR_IDS[3]
            )

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        return input_ids, pixel_values, grid_thws
