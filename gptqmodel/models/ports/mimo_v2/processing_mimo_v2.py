# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Direct native Qwen2-VL patch processing declared by the pinned MiMo assets.

The checkpoint declares Qwen2_5_VLProcessor/Qwen2VLImageProcessor. Selection is
fixed here, never delegated to checkpoint processor/tokenizer Python. MiMo's
decoder consumes ordinary positions and different video argument names.
"""

from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor

from .configuration_mimo_v2 import MiMoV2Config


class MiMoV2Processor(Qwen2_5_VLProcessor):
    @classmethod
    def from_pretrained(cls, path, *, config=None, tokenizer=None, **kwargs):
        from ....utils.hf import load_hf_tokenizer

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = config or MiMoV2Config.from_pretrained(path, **kwargs)
        if tokenizer is None:
            tokenizer = load_hf_tokenizer(path, model_config=config, trust_remote_code=False, **kwargs)
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if config.vision_config:
            image = Qwen2VLImageProcessor.from_pretrained(path, **kwargs)
            # The same pinned preprocessor_config supplies both patch layouts.
            video = Qwen2VLVideoProcessor.from_pretrained(path, **kwargs)
        else:
            image, video = Qwen2VLImageProcessor(), Qwen2VLVideoProcessor()
        return cls(image_processor=image, video_processor=video, tokenizer=tokenizer,
                   chat_template=getattr(tokenizer, "chat_template", None))

    def __call__(self, *args, audio_codes=None, audio_embeds=None, **kwargs):
        if kwargs.get("audio") is not None:
            raise ValueError("MiMo waveform preprocessing requires the separately loaded audio codec; pass audio_codes")
        kwargs.setdefault("return_mm_token_type_ids", False)
        inputs = super().__call__(*args, **kwargs)
        if "pixel_values_videos" in inputs:
            inputs["video_pixel_values"] = inputs.pop("pixel_values_videos")
        inputs.pop("second_per_grid_ts", None)
        inputs.pop("mm_token_type_ids", None)
        if audio_codes is not None:
            inputs["audio_codes"] = audio_codes
        if audio_embeds is not None:
            inputs["audio_embeds"] = audio_embeds
        return inputs
