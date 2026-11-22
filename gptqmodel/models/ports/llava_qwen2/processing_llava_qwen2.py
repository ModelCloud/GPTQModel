# Copyright 2023 Haotian Liu
# SPDX-License-Identifier: Apache-2.0
# Package-owned processor adaptation by ModelCloud.ai, 2026; see NOTICE.

import math

import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from ..native.clip.image_processing_clip import CLIPImageProcessor
from .configuration_llava_qwen2 import LlavaConfig
from .multimodal import IMAGE_TOKEN_INDEX, select_best_resolution


def image_processor_for_config(config):
    if config.mm_vision_tower.startswith("mobileclip"):
        size = int(config.mm_vision_tower.rsplit("_", 1)[-1])
        return CLIPImageProcessor(size={"shortest_edge": size}, crop_size={"height": size, "width": size},
                                  image_mean=[0.0] * 3, image_std=[1.0] * 3)
    size = (config.vision_config or {}).get("image_size", 336 if config.mm_vision_tower.endswith("336") else 224)
    if getattr(config, "s2", False):
        size = max(int(value) for value in getattr(config, "s2_scales", "336,672,1008").split(","))
    return CLIPImageProcessor(size={"shortest_edge": size}, crop_size={"height": size, "width": size})


def process_image(image, processor, config):
    image = image.convert("RGB")
    aspect = getattr(config, "image_aspect_ratio", "square")
    if aspect == "pad":
        size = max(image.size)
        padded = Image.new("RGB", (size, size), tuple(int(x * 255) for x in processor.image_mean))
        padded.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        return processor(padded, return_tensors="pt")["pixel_values"][0]
    if aspect == "anyres":
        width, height = select_best_resolution(image.size, config.image_grid_pinpoints)
        patch = processor.crop_size["height"]
        if width % patch or height % patch:
            raise ValueError("image_grid_pinpoints must contain whole vision image crops")
        scale = min(width / image.width, height / image.height)
        new_size = (min(math.ceil(image.width * scale), width), min(math.ceil(image.height * scale), height))
        padded = Image.new("RGB", (width, height), (0, 0, 0))
        padded.paste(image.resize(new_size), ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
        size = processor.size["shortest_edge"]
        patches = [image.resize((size, size))]
        patches += [padded.crop((x, y, x + patch, y + patch))
                    for y in range(0, height, patch) for x in range(0, width, patch)]
        return processor(patches, return_tensors="pt")["pixel_values"]
    return processor(image, return_tensors="pt")["pixel_values"][0]


class LlavaQwen2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"
    model_input_names = ["input_ids", "attention_mask", "images", "image_sizes"]

    def __init__(self, image_processor, tokenizer, config, **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, **kwargs)
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, config=None, tokenizer=None, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        if config is None:
            config = LlavaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            from ....utils.hf import load_hf_tokenizer
            tokenizer = load_hf_tokenizer(pretrained_model_name_or_path, model_config=config,
                                          trust_remote_code=False, **kwargs)
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        return cls(image_processor_for_config(config), tokenizer, config)

    def __call__(self, text, images=None, padding=True, return_tensors="pt", **kwargs):
        if return_tensors != "pt":
            raise ValueError("LLaVA-Qwen2 processor currently returns PyTorch tensors only")
        rows = []
        for prompt in ([text] if isinstance(text, str) else text):
            chunks = [self.tokenizer(chunk, **kwargs).input_ids for chunk in prompt.split("<image>")]
            bos = bool(chunks[0] and chunks[0][0] == self.tokenizer.bos_token_id)
            ids = chunks[0][:1] if bos else []
            for index, chunk in enumerate(chunks):
                if index:
                    ids.append(IMAGE_TOKEN_INDEX)
                ids.extend(chunk[int(bos):])
            rows.append(ids)
        if not rows or any(not row for row in rows):
            raise ValueError("Prompts must contain at least one token")
        if not padding and len({len(row) for row in rows}) != 1:
            raise ValueError("Batched PyTorch prompts of different lengths require padding")
        width = max(map(len, rows))
        pad = self.tokenizer.pad_token_id
        if pad is None and len({len(row) for row in rows}) != 1:
            raise ValueError("A padding token is required for unequal-length prompts")
        ids = torch.full((len(rows), width), pad if pad is not None else 0, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            section = slice(width - len(row), width) if self.config.tokenizer_padding_side == "left" else slice(0, len(row))
            ids[index, section] = torch.tensor(row)
            mask[index, section] = 1
        result = {"input_ids": ids, "attention_mask": mask}
        if images is not None:
            images = [images] if isinstance(images, Image.Image) else list(images)
            result["image_sizes"] = [image.size for image in images]
            values = [process_image(image, self.image_processor, self.config) for image in images]
            result["images"] = torch.stack(values) if values and all(x.shape == values[0].shape for x in values) else values
        return BatchFeature(result)

    def apply_chat_template(self, conversation, *, tokenize=False, return_dict=False, return_tensors="pt",
                            add_generation_prompt=False, padding=True, **kwargs):
        batched = bool(conversation and isinstance(conversation[0], (list, tuple)))
        conversations = conversation if batched else [conversation]
        prompts, image_groups = [], []
        for messages in conversations:
            rendered, images = [], []
            for message in messages:
                message = dict(message)
                content = message["content"]
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if item["type"] == "image":
                            image = item.get("image")
                            if not isinstance(image, Image.Image):
                                raise ValueError("Image chat content must contain an already-loaded PIL image")
                            images.append(image)
                            parts.append("<image>\n")
                        elif item["type"] == "text":
                            parts.append(item["text"])
                        else:
                            raise ValueError(f"Unsupported chat content type: {item['type']}")
                    message["content"] = "".join(parts)
                rendered.append(message)
            prompts.append(self.tokenizer.apply_chat_template(rendered, tokenize=False,
                           add_generation_prompt=add_generation_prompt, **kwargs))
            image_groups.append(images)
        if not tokenize:
            return prompts if batched else prompts[0]
        images = None
        if any(image_groups):
            size = self.image_processor.crop_size["height"]
            images = [image for group in image_groups for image in (group or [Image.new("RGB", (size, size))])]
        result = self(prompts, images=images, padding=padding, return_tensors=return_tensors)
        return result if return_dict else result["input_ids"]
