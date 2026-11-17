# SPDX-FileCopyrightText: 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""Pinned Prismatic image transforms and native-tokenizer processor binding."""

import torch
from PIL import Image
from transformers import BaseImageProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature


class PrismaticImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(self, use_fused_vision_backbone=False, image_resize_strategy="letterbox",
                 input_sizes=None, interpolations=None, means=None, stds=None, **kwargs):
        for key in list(kwargs):
            if key.startswith("tvf_") or key in {"auto_map", "processor_class", "image_processor_type"}:
                kwargs.pop(key)
        super().__init__(**kwargs)
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy
        self.input_sizes = input_sizes or [[3, 224, 224]]
        self.interpolations = interpolations or ["bicubic"] * len(self.input_sizes)
        self.means = means or [[0.5, 0.5, 0.5]] * len(self.input_sizes)
        self.stds = stds or [[0.5, 0.5, 0.5]] * len(self.input_sizes)
        count = 2 if use_fused_vision_backbone else 1
        if any(len(v) != count for v in (self.input_sizes, self.interpolations, self.means, self.stds)):
            raise ValueError("OpenVLA requires one image transform per vision backbone")
        if image_resize_strategy not in {"letterbox", "resize-naive", "resize-crop"}:
            raise ValueError(f"Unsupported OpenVLA image resize strategy: {image_resize_strategy}")
        if any(len(s) != 3 or s[0] != 3 or s[1] != s[2] or s[1] <= 0 for s in self.input_sizes):
            raise ValueError("OpenVLA requires positive square RGB input sizes")
        if count == 2 and self.input_sizes[0] != self.input_sizes[1]:
            raise ValueError("Fused OpenVLA image transforms must have equal output shapes")
        if any(len(m) != 3 for m in self.means) or any(len(s) != 3 or min(s) <= 0 for s in self.stds):
            raise ValueError("OpenVLA requires three means and positive standard deviations per image")

    def apply_transform(self, image):
        try:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as tvf
        except ImportError as exc:
            raise ImportError(
                "OpenVLA image processing requires torchvision; install with `pip install 'gptqmodel[vision]'` "
                "using a torchvision build compatible with your PyTorch installation."
            ) from exc

        if not isinstance(image, Image.Image):
            raise TypeError("OpenVLA image processing accepts PIL images")
        image = image.convert("RGB")
        if self.image_resize_strategy == "letterbox":
            width, height = image.size
            side = max(width, height)
            horizontal, vertical = (side - width) // 2, (side - height) // 2
            # The source applies one common letterbox using the final backbone's mean.
            fill = tuple(int(x * 255) for x in self.means[-1])
            image = tvf.pad(image, (horizontal, vertical, horizontal, vertical), fill=fill)
        outputs = []
        for size, interpolation, mean, std in zip(self.input_sizes, self.interpolations, self.means, self.stds):
            resized = tvf.resize(
                image, size[-2:] if self.image_resize_strategy == "resize-naive" else size[-1],
                interpolation=InterpolationMode(interpolation), antialias=True,
            )
            resized = tvf.center_crop(resized, size[-2:])
            outputs.append(tvf.normalize(tvf.to_tensor(resized), mean, std))
        return torch.cat(outputs, dim=0)

    def preprocess(self, images, return_tensors=None, **kwargs):
        del kwargs
        images = images if isinstance(images, (list, tuple)) else [images]
        if not images:
            raise ValueError("OpenVLA requires at least one image")
        pixels = torch.stack([self.apply_transform(image) for image in images])
        return BatchFeature({"pixel_values": pixels.float().numpy()}, tensor_type=return_tensors)


class PrismaticProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, tokenizer=None, config=None, **kwargs):
        from ....utils.hf import load_hf_tokenizer
        from .configuration_prismatic import OpenVLAConfig

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        if config is None:
            config = OpenVLAConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            tokenizer = load_hf_tokenizer(
                pretrained_model_name_or_path, model_config=config, trust_remote_code=False, **kwargs,
            )
        image_processor = PrismaticImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(self, text=None, images=None, padding=False, truncation=None, max_length=None,
                 return_tensors="pt", **kwargs):
        text_inputs = self.tokenizer(text, padding=padding, truncation=truncation, max_length=max_length,
                                     return_tensors=return_tensors, **kwargs)
        if images is None:
            return BatchFeature(dict(text_inputs))
        pixels = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        if len(pixels) != len(text_inputs["input_ids"]):
            raise ValueError("OpenVLA requires the same number of images and text rows")
        return BatchFeature({**text_inputs, "pixel_values": pixels})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))
