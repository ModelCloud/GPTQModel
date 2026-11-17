# Copyright 2024 AIDC-AI
# SPDX-License-Identifier: Apache-2.0
"""Source Ovis aspect-preserving images and conversation protocols."""

import copy

import torch
from transformers import CLIPImageProcessor, ProcessorMixin, SiglipImageProcessor
from transformers.feature_extraction_utils import BatchFeature

from .configuration_ovis import IGNORE_ID, IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS, IMAGE_TOKEN_ID, OvisConfig


def image_processor_for_config(config):
    visual = config.visual_tokenizer_config
    size = visual.backbone_config.image_size
    if visual.model_type == "clip_visual_tokenizer":
        return CLIPImageProcessor(size={"height": size, "width": size}, crop_size={"height": size, "width": size},
                                  do_center_crop=False)
    return SiglipImageProcessor(size={"height": size, "width": size})


def _square(image, processor, side):
    width, height = image.size
    if width >= height:
        width, height = side, max(1, int(height / width * side))
    else:
        width, height = max(1, int(width / height * side)), side
    values = processor.preprocess(image, size={"height": height, "width": width}, return_tensors="pt")["pixel_values"]
    result = values.new_zeros(1, 3, side, side)
    y, x = (side - height) // 2, (side - width) // 2
    result[:, :, y:y + height, x:x + width] = values
    return result


def image_placeholders(grid):
    ids = [IMAGE_INDICATOR_IDS[0], IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS[1]]
    if grid[0] * grid[1] > 1:
        for row in range(grid[0]):
            for col in range(grid[1]):
                ids.append(IMAGE_ATOM_ID)
                if col < grid[1] - 1:
                    ids.append(IMAGE_INDICATOR_IDS[2])
            if row < grid[0] - 1:
                ids.append(IMAGE_INDICATOR_IDS[3])
    return ids + [IMAGE_INDICATOR_IDS[4]]


def process_image(image, processor, config, max_partition=9, covering_threshold=0.9, convert_to_rgb=True):
    if convert_to_rgb:
        image = image.convert("RGB")
    visual = config.visual_tokenizer_config
    side = visual.backbone_config.image_size
    if config.ovis_version != "1.6":
        if visual.hd_booster is None:
            return _square(image, processor, side), [IMAGE_TOKEN_ID]
        width, height = image.size
        if visual.hd_booster == "s2wrapper-adaptive" and (height < side * 1.5 or width < side * 1.5):
            values = torch.full((1, 15, side, side), torch.inf)
            values[:, :3] = _square(image, processor, side)
        else:
            crops = [image] + [image.crop(box) for box in (
                (0, 0, width // 2, height // 2), (width // 2, 0, width, height // 2),
                (0, height // 2, width // 2, height), (width // 2, height // 2, width, height))]
            values = torch.cat([_square(crop, processor, side) for crop in crops], dim=1)
        return values, [IMAGE_TOKEN_ID]
    if max_partition < 1:
        raise ValueError("max_partition must be positive")

    def partition(grid):
        width, height = image.size
        return [(col * (width // grid[1]), row * (height // grid[0]),
                 width if col == grid[1] - 1 else (col + 1) * (width // grid[1]),
                 height if row == grid[0] - 1 else (row + 1) * (height // grid[0]))
                for row in range(grid[0]) for col in range(grid[1])]

    def coverage(box):
        width, height = sorted((box[2] - box[0], box[3] - box[1]), reverse=True)
        if width > side:
            height, width = height / width * side, side
        return width * height

    choices = []
    for rows in range(1, max_partition + 1):
        for cols in range(1, max_partition + 1):
            if rows * cols <= max_partition and rows <= image.height and cols <= image.width:
                grid = (rows, cols)
                ratio = sum(map(coverage, partition(grid))) / (image.width * image.height)
                choices.append((grid, ratio))
    good = [choice for choice in choices if choice[1] > covering_threshold]
    grid = min(good, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0] if good else min(
        choices, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0]
    crops = [image.crop(box) for box in partition(grid)]
    if len(crops) > 1:
        crops.insert(0, image)
    return torch.cat([_square(crop, processor, side) for crop in crops]), image_placeholders(grid)


class ConversationFormatter:
    def __init__(self, tokenizer, kind, version="1.6"):
        self.tokenizer, self.kind, self.version = tokenizer, kind, version

    def _tokenize_with_image_symbol(self, text):
        ids = []
        for index, part in enumerate(text.split("<image>")):
            if index:
                ids.append(IMAGE_TOKEN_ID)
            ids.extend(self.tokenizer(part, add_special_tokens=False).input_ids)
        return ids

    def format(self, conversations, generation_preface=None):
        conversations = copy.deepcopy(conversations)
        if not conversations:
            raise ValueError("Ovis requires a nonempty conversation")
        llama = self.kind == "Llama3ConversationFormatter"
        gemma = self.kind == "GemmaConversationFormatter"
        if gemma:
            roles = {"human": "<start_of_turn>user\n", "gpt": "<start_of_turn>model\n"}
            bos, end = "<bos>", "<end_of_turn>\n"
            if conversations[0]["from"] == "system":
                raise ValueError("Gemma does not support system prompt")
        else:
            roles = {role: (f"<|start_header_id|>{name}<|end_header_id|>\n\n" if llama else f"<|im_start|>{name}\n")
                     for role, name in (("system", "system"), ("human", "user"), ("gpt", "assistant"))}
            bos, end = ("<|begin_of_text|>", "<|eot_id|>") if llama else (
                "", "<|im_end|>" if self.version == "1.0" else "<|im_end|>\n")
            if conversations[0]["from"] != "system":
                conversations.insert(0, {"from": "system", "value": "You are a helpful and honest multimodal assistant."
                                         if llama else "You are a helpful assistant."})
        if generation_preface is not None:
            conversations.append({"from": "gpt", "value": generation_preface})
        prompt = bos
        ids = self._tokenize_with_image_symbol(bos)
        labels = [IGNORE_ID] * len(ids)
        for index, message in enumerate(conversations):
            role = roles[message["from"]]
            value = message["value"].strip() if (llama or gemma) else message["value"]
            text = role + value
            if index < len(conversations) - 1 or generation_preface is None:
                text += end
            if not (llama or gemma) and self.version == "1.0" and index < len(conversations) - 1:
                text += "\n"
            part = self._tokenize_with_image_symbol(text)
            targets = [IGNORE_ID] * len(part)
            if message["from"] == "gpt" and (llama or gemma or self.version == "1.0" or generation_preface is None):
                begin = len(self.tokenizer(role, add_special_tokens=False).input_ids)
                stop = -1 if gemma or (not llama and self.version != "1.0") else len(part)
                targets[begin:stop] = part[begin:stop]
            prompt += text
            ids.extend(part)
            labels.extend(targets)
        return prompt, ids, labels

    def format_query(self, query, generation_preface=""):
        return self.format([{"from": "human", "value": query}], generation_preface)[:2]


class OvisProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("CLIPImageProcessor", "SiglipImageProcessor")
    tokenizer_class = "AutoTokenizer"
    model_input_names = ["input_ids", "attention_mask", "pixel_values"]

    def __init__(self, image_processor, tokenizer, config, **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, **kwargs)
        self.config = config

    @classmethod
    def from_pretrained(cls, path, *, config=None, tokenizer=None, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = config if config is not None else OvisConfig.from_pretrained(path, **kwargs)
        if tokenizer is None:
            from ....utils.hf import load_hf_tokenizer
            tokenizer = load_hf_tokenizer(path, model_config=config, trust_remote_code=False, **kwargs)
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        image_cls = CLIPImageProcessor if config.visual_tokenizer_config.model_type == "clip_visual_tokenizer" else SiglipImageProcessor
        processor = image_cls.from_pretrained(path, **kwargs)
        if not processor.do_resize or getattr(processor, "do_center_crop", False) or not processor.do_rescale or not processor.do_normalize:
            raise ValueError("Unqualified Ovis image processor settings")
        return cls(processor, tokenizer, config)

    def preprocess_inputs(self, text_or_conversations, images=None, max_partition=9, generation_preface="",
                          return_labels=False, propagate_exception=True):
        conversations = [{"from": "human", "value": text_or_conversations}] if isinstance(text_or_conversations, str) else text_or_conversations
        prompt, raw_ids, raw_labels = ConversationFormatter(self.tokenizer, self.config.conversation_formatter_class,
                                                           self.config.ovis_version).format(
            conversations, generation_preface)
        images = [] if images is None else images
        if sum(value == IMAGE_TOKEN_ID for value in raw_ids) != len(images):
            raise ValueError("Ovis image placeholders and images must match")
        ids, labels, pixels = [], [], []
        index = 0
        for token, label in zip(raw_ids, raw_labels):
            if token != IMAGE_TOKEN_ID:
                ids.append(token)
                labels.append(label)
                continue
            values, placeholders = process_image(images[index], self.image_processor, self.config, max_partition)
            index += 1
            ids.extend(placeholders)
            labels.extend([IGNORE_ID] * len(placeholders))
            pixels.append(values)
        result = (prompt, torch.tensor(ids, dtype=torch.long), torch.cat(pixels) if pixels else None)
        return (*result, torch.tensor(labels, dtype=torch.long)) if return_labels else result

    def __call__(self, text, images=None, return_tensors="pt", **kwargs):
        if return_tensors != "pt":
            raise ValueError("Ovis processor returns PyTorch tensors only")
        _, ids, pixels = self.preprocess_inputs(text, images, **kwargs)
        return BatchFeature({"input_ids": ids.unsqueeze(0), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                             "pixel_values": [pixels]})

    def save_pretrained(self, path, **kwargs):
        self.image_processor.save_pretrained(path, **kwargs)
        self.tokenizer.save_pretrained(path, **kwargs)
        self.config.save_pretrained(path)
