# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 LFM3-VL, with explicit media component selection.

The model, config and processor are the vendored 5.23 classes. The pinned
checkpoint's historical ``Lfm3Vl{Image,Video}Processor`` names map to the
vendored Qwen media processors.
"""

from ..native.lfm3_vl.configuration_lfm3_vl import Lfm3VlConfig
from ..native.lfm3_vl.modeling_lfm3_vl import Lfm3VlForConditionalGeneration
from ..native.lfm3_vl.processing_lfm3_vl import Lfm3VlProcessor


class Lfm3VlNativeProcessor(Lfm3VlProcessor):
    """Construct native components without checkpoint processor discovery."""

    @staticmethod
    def get_possibly_dynamic_module(module_name):
        # The metadata reader also understands audio-tokenizer references.
        # LFM3-VL has no such component: do not let metadata load another model.
        raise ValueError(f"LFM3-VL does not accept checkpoint-selected processor components: {module_name}")

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, tokenizer=None, config=None, **kwargs):
        # These media processors import torchvision. Resolve them only when the
        # processor is actually built so config/model loading stays dependency-light.
        from ..native.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from ..native.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

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
