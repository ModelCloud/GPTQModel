# SPDX-FileCopyrightText: 2024 AIDC-AI
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned Ovis chat, resize and patchification, using native data-only components."""

from types import SimpleNamespace

import torch
from transformers import ProcessorMixin, SiglipImageProcessor

from .vision import (
    IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER_ID,
    INDICATOR_IDS,
    VIDEO_PLACEHOLDER,
    VIDEO_PLACEHOLDER_ID,
    VISUAL_ATOM_ID,
    VisualTokenizer,
)


class Ovis26Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "PreTrainedTokenizerBase"
    model_input_names = ["input_ids", "attention_mask", "pixel_values", "grid_thws"]

    def __init__(self, config, tokenizer, image_processor=None):
        self.config = config
        self.tokenizer = self.text_tokenizer = tokenizer
        self.image_processor = image_processor if image_processor is not None else SiglipImageProcessor()
        adapter = SimpleNamespace(
            vit=SimpleNamespace(config=config.vit_config),
            image_processor=self.image_processor,
            smart_resize=VisualTokenizer.smart_resize,
        )
        adapter.preprocess = lambda **kwargs: VisualTokenizer.preprocess(adapter, **kwargs)
        self.visual_tokenizer = adapter

    @classmethod
    def from_pretrained(cls, path, *, config=None, tokenizer=None, **kwargs):
        if config is None or tokenizer is None:
            raise ValueError("Ovis26Processor requires the selected config and normalized tokenizer")
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        image_processor = SiglipImageProcessor.from_pretrained(path, **kwargs)
        return cls(config, tokenizer, image_processor)

    def save_pretrained(self, path, **kwargs):
        self.image_processor.save_pretrained(path, **kwargs)
        self.tokenizer.save_pretrained(path, **kwargs)

    def __call__(self, messages, **kwargs):
        ids, pixels, grids = self.preprocess_inputs(messages, **kwargs)
        return {
            "input_ids": ids,
            "attention_mask": ids.ne(self.tokenizer.pad_token_id),
            "pixel_values": pixels,
            "grid_thws": grids,
        }

    def _merge_inputs(self, raw_input_ids, placeholder_id, grid_thws, indicator_begin_id, indicator_end_id):
        input_ids = []
        prev_index = 0
        placeholder_indexes = [i for i, v in enumerate(raw_input_ids) if v == placeholder_id]
        if len(placeholder_indexes) != len(grid_thws):
            raise ValueError("Visual placeholder count does not match supplied media")
        for placeholder_index, grid_thw in zip(placeholder_indexes, grid_thws):
            input_ids.extend(raw_input_ids[prev_index:placeholder_index])
            num_image_atoms = grid_thw.prod().item()
            num_image_atoms //= self.visual_tokenizer.vit.config.hidden_stride**2
            num_image_atoms //= self.visual_tokenizer.vit.config.temporal_patch_size
            input_ids.extend([indicator_begin_id] + [VISUAL_ATOM_ID] * num_image_atoms + [indicator_end_id])
            prev_index = placeholder_index + 1
        input_ids.extend(raw_input_ids[prev_index:])
        return input_ids

    def _tokenize_with_visual_placeholder(self, text):
        placeholder = VIDEO_PLACEHOLDER if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER
        placeholder_id = VIDEO_PLACEHOLDER_ID if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER_ID
        chunks = [self.text_tokenizer(chunk, add_special_tokens=False).input_ids for chunk in text.split(placeholder)]
        input_ids = chunks[0]
        for chunk in chunks[1:]:
            input_ids.append(placeholder_id)
            input_ids.extend(chunk)
        return input_ids

    def preprocess_inputs(
        self,
        messages: list,
        min_pixels=448 * 448,
        max_pixels=1792 * 1792,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        text = self.text_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking
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
                *(
                    self.visual_tokenizer.preprocess(image=image, min_pixels=min_pixels, max_pixels=max_pixels)
                    for image in images
                )
            )
            input_ids = self._merge_inputs(
                input_ids, IMAGE_PLACEHOLDER_ID, grid_thws, INDICATOR_IDS[0], INDICATOR_IDS[1]
            )
            pixel_values = torch.cat(pixel_values, dim=0)
            grid_thws = torch.cat(grid_thws, dim=0)
        elif videos:
            assert len(videos) == 1, "only support single video"
            pixel_values, grid_thws = self.visual_tokenizer.preprocess(
                video=videos[0], min_pixels=min_pixels, max_pixels=max_pixels
            )
            input_ids = self._merge_inputs(
                input_ids, VIDEO_PLACEHOLDER_ID, grid_thws, INDICATOR_IDS[2], INDICATOR_IDS[3]
            )

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        return input_ids, pixel_values, grid_thws
