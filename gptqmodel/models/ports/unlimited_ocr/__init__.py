# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native generative task; the package path also binds the legacy alias."""

from transformers import PreTrainedConfig, UnlimitedOcrConfig, UnlimitedOcrImageProcessor
from transformers import TokenizersBackend as UnlimitedOcrTokenizer
from transformers import UnlimitedOcrForConditionalGeneration as NativeUnlimitedOcrForConditionalGeneration
from transformers import UnlimitedOcrProcessor as NativeUnlimitedOcrProcessor


class UnlimitedOcrConfigLoader:
    """Canonicalize only the published legacy alias before native conversion lookup."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        payload, kwargs = PreTrainedConfig.get_config_dict(path, **kwargs)
        if payload.get("model_type") not in {"unlimited_ocr", "unlimited-ocr"}:
            raise ValueError("Unlimited-OCR requires unlimited_ocr or unlimited-ocr model_type")
        payload["model_type"] = "unlimited_ocr"
        return UnlimitedOcrConfig.from_dict(payload, **kwargs)


class UnlimitedOcrProcessor:
    """Bind exact native media components without checkpoint processor discovery."""

    @classmethod
    def from_pretrained(cls, path, *, config, tokenizer, **kwargs):
        return NativeUnlimitedOcrProcessor(
            image_processor=UnlimitedOcrImageProcessor.from_pretrained(path, **kwargs),
            tokenizer=tokenizer, patch_size=config.vision_config.sam_config.patch_size,
            downsample_ratio=4,
        )


class UnlimitedOcrForConditionalGeneration:
    """Loader boundary only; successful construction returns the exact native class."""

    @staticmethod
    def _validate_attention(config, kwargs):
        text = config.text_config
        implementation = kwargs.get("attn_implementation", config._attn_implementation)
        if isinstance(implementation, dict):
            implementation = implementation.get("text_config", implementation.get("", None))
        if implementation is None:
            implementation = text._attn_implementation
        if implementation == "eager" and text.num_key_value_heads not in (1, text.num_attention_heads):
            raise ValueError(
                "Transformers 5 Unlimited-OCR eager attention does not repeat grouped KV heads; "
                "use attn_implementation='sdpa' for GQA. Released MHA and MQA are unaffected."
            )

    @classmethod
    def _from_config(cls, config, **kwargs):
        cls._validate_attention(config, kwargs)
        return NativeUnlimitedOcrForConditionalGeneration._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        if kwargs.get("config") is None:
            kwargs["config"] = UnlimitedOcrConfigLoader.from_pretrained(path)
        cls._validate_attention(kwargs["config"], kwargs)
        return NativeUnlimitedOcrForConditionalGeneration.from_pretrained(path, **kwargs)


__all__ = ["UnlimitedOcrConfigLoader", "UnlimitedOcrForConditionalGeneration", "UnlimitedOcrProcessor", "UnlimitedOcrTokenizer"]
