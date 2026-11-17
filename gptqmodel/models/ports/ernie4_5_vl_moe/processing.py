# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Select native media components without checkpoint processor discovery."""

from pathlib import Path

from transformers import Ernie4_5_VLMoeProcessor


class ErnieVLProcessor(Ernie4_5_VLMoeProcessor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, config=None, tokenizer=None, **kwargs):
        from transformers import Ernie4_5_VLMoeImageProcessorPil, Ernie4_5_VLMoeVideoProcessor

        from ....utils.hf import load_hf_tokenizer
        from .configuration import ErnieVLConfig

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        if config is None:
            config = ErnieVLConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if tokenizer is None:
            tokenizer = load_hf_tokenizer(pretrained_model_name_or_path, model_config=config, **kwargs)
        image_values, _ = Ernie4_5_VLMoeImageProcessorPil.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
        if "shortest_edge" not in image_values.get("size", {}):
            image_values["size"] = {
                "shortest_edge": image_values.pop("min_pixels", 3136),
                "longest_edge": image_values.pop("max_pixels", 4816896),
            }
        video_values, _ = Ernie4_5_VLMoeVideoProcessor.get_video_processor_dict(pretrained_model_name_or_path, **kwargs)
        if "shortest_edge" not in video_values.get("size", {}):
            video_values["size"] = dict(Ernie4_5_VLMoeVideoProcessor.size)
        video_values.setdefault("draw_on_frames", True)
        video_values.setdefault("font", str(Path(__file__).with_name("Roboto-Regular.ttf")))
        image_processor = Ernie4_5_VLMoeImageProcessorPil.from_dict(image_values)
        video_processor = Ernie4_5_VLMoeVideoProcessor.from_dict(video_values)
        return cls(
            image_processor=image_processor, video_processor=video_processor, tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )
