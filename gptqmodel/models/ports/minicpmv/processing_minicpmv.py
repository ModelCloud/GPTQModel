# SPDX-FileCopyrightText: 2024 The HuggingFace Inc. team
# SPDX-FileCopyrightText: 2024 OpenBMB
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned MiniCPM-V media assembly with caller-normalized tokenizers.

Special image markers are processor vocabulary contracts, not tokenizer patches.
No checkpoint class names, Python modules or AutoProcessor registrations are used.
"""

import json
from pathlib import Path

import torch
from transformers import ProcessorMixin
from transformers.utils import cached_file

from .configuration_minicpm import MiniCPMVConfig
from .image_processing_minicpmv import MiniCPMVBatchFeature, MiniCPMVImageProcessor


class MiniCPMVProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BaseImageProcessor"
    tokenizer_class = "PreTrainedTokenizerBase"

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if self.tokenizer is None:
            raise ValueError("MiniCPM-V processor requires the caller's Tokenicer-normalized tokenizer")
        self.version = float(image_processor.version)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, tokenizer=None, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        if config is None:
            config = MiniCPMVConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            # The shared loader owns Tokenicer normalization and the default-deny
            # boundary; processors must not rediscover checkpoint Python.
            from ....utils.hf import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(str(pretrained_model_name_or_path), model_config=config, trust_remote_code=False)
        path = cached_file(pretrained_model_name_or_path, "preprocessor_config.json",
                           _raise_exceptions_for_missing_entries=False, **kwargs)
        settings = json.loads(Path(path).read_text()) if path is not None else {}
        if path is None and config.version >= 2.5:
            raise OSError("MiniCPM-V requires preprocessor_config.json")
        settings.update(version=config.version, image_feature_size=config.query_num,
                        patch_size=config.patch_size, slice_mode=config.slice_mode)
        settings.setdefault("scale_resolution", getattr(config, "scale_resolution", config.slice_config.scale_resolution)
                            if config.version != 1.0 else config.image_size)
        settings.setdefault("max_slice_nums", getattr(config, "max_slice_nums", config.slice_config.max_slice_nums))
        settings.setdefault("use_image_id", config.use_image_id)
        return cls(MiniCPMVImageProcessor(**settings), tokenizer)

    def save_pretrained(self, save_directory, **kwargs):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.image_processor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        destination = path / "processor_config.json"
        destination.write_text(json.dumps({"processor_class": type(self).__name__}, indent=2) + "\n")
        return [str(destination)]

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "tgt_sizes", "image_bound", "temporal_ids"]

    def _convert(self, text, max_length):
        ids = self.tokenizer.encode(text)
        add_bos = getattr(self.tokenizer, "add_bos_token", False)
        if self.version < 2.5 and not add_bos:
            ids = [self.tokenizer.bos_token_id] + ids
        if max_length is not None:
            ids = ids[:max_length]
        ids = torch.tensor(ids, dtype=torch.int32)
        def token(name):
            value = self.tokenizer.convert_tokens_to_ids(name)
            if value is None or value == self.tokenizer.unk_token_id:
                raise ValueError(f"MiniCPM-V tokenizer is missing image marker {name}")
            return value
        starts = ids == token("<image>")
        ends = ids == token("</image>")
        if self.version >= 2.5:
            starts |= ids == token("<slice>")
            ends |= ids == token("</slice>")
        left = torch.where(starts)[0] + 1
        right = torch.where(ends)[0]
        if len(left) != len(right) or bool((right <= left).any()):
            raise ValueError("MiniCPM-V image marker pair was truncated or malformed")
        return ids, torch.stack((left, right), dim=-1)

    def __call__(self, text, images=None, max_length=None, do_pad=True, max_slice_nums=None,
                 use_image_id=None, temporal_ids=None, return_tensors="pt", **kwargs):
        texts = [text] if isinstance(text, str) else text
        if images is None:
            images = [[] for _ in texts]
        if temporal_ids is not None and self.version != 4.5:
            raise ValueError("Temporal frame pooling requires MiniCPM-V 4.5")
        media = self.image_processor(images, do_pad=do_pad, max_slice_nums=max_slice_nums,
                                     temporal_ids=temporal_ids, return_tensors=return_tensors)
        if len(media["pixel_values"]) != len(texts):
            raise ValueError("MiniCPM-V images and text batch sizes differ")
        ids_list, bounds_list = [], []
        marker = "(<image>./</image>)"
        for i, prompt in enumerate(texts):
            parts = prompt.split(marker)
            if len(parts) - 1 != len(media["image_sizes"][i]):
                raise ValueError("MiniCPM-V image placeholders and images differ")
            final = parts[0]
            for j, size in enumerate(media["image_sizes"][i]):
                skip = j in media["skip_image_idx"][i]
                placeholder = "" if skip else self.image_processor.get_slice_image_placeholder(
                    size, j, max_slice_nums, use_image_id)
                final += placeholder + parts[j + 1]
            ids, bounds = self._convert(final, max_length)
            expected_bounds = len(media["temporal_ids"][i]) if self.version == 4.5 else len(media["pixel_values"][i])
            if len(bounds) != expected_bounds or bool(((bounds[:, 1] - bounds[:, 0]) != self.image_processor.image_feature_size).any()):
                raise ValueError("MiniCPM-V truncated or malformed image placeholders")
            ids_list.append(ids)
            bounds_list.append(bounds)
        length = max(len(ids) for ids in ids_list)
        padded = torch.zeros(len(texts), length, dtype=torch.int32)
        # Upstream uses zero left padding even when EOS is a distinct token.
        mask = torch.zeros_like(padded, dtype=torch.bool)
        for i, ids in enumerate(ids_list):
            padded[i, -len(ids):] = ids
            mask[i, -len(ids):] = ids.ne(0)
            bounds_list[i] += length - len(ids)
        return MiniCPMVBatchFeature({
            "input_ids": padded, "attention_mask": mask, "image_bound": bounds_list,
            "pixel_values": media["pixel_values"], "image_sizes": media["image_sizes"],
            "tgt_sizes": media["tgt_sizes"], "temporal_ids": media["temporal_ids"],
        })

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
