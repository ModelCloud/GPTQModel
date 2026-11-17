# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Transformers aliases, with explicit native media component selection.

No modeling implementation is copied. See README.md for the reviewed native
version and the pinned checkpoint whose historical processor names differ.
"""

from transformers import (
    Lfm3VlConfig,
    Lfm3VlForConditionalGeneration,
    Lfm3VlProcessor,
    Qwen2VLImageProcessor,
    Qwen3VLVideoProcessor,
)


class Lfm3VlNativeProcessor(Lfm3VlProcessor):
    """Construct native components without checkpoint processor discovery."""

    @staticmethod
    def get_possibly_dynamic_module(module_name):
        # The metadata reader also understands audio-tokenizer references.
        # LFM3-VL has no such component: do not let metadata load another model.
        raise ValueError(f"LFM3-VL does not accept checkpoint-selected processor components: {module_name}")

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, tokenizer=None, config=None, **kwargs):
        del config
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        if tokenizer is None:
            raise ValueError("LFM3-VL processor requires the tokenizer selected by the model loader")
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_id_or_path, trust_remote_code=False, **kwargs,
        )
        video_processor = Qwen3VLVideoProcessor.from_pretrained(
            model_id_or_path, trust_remote_code=False, **kwargs,
        )
        processor_dict, processor_kwargs = cls.get_processor_dict(
            model_id_or_path, trust_remote_code=False, **kwargs,
        )
        return Lfm3VlProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=processor_kwargs.get("chat_template", processor_dict.get("chat_template")),
        )


__all__ = ["Lfm3VlConfig", "Lfm3VlForConditionalGeneration", "Lfm3VlNativeProcessor"]
