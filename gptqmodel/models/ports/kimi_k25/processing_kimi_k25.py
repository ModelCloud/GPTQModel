# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Load released media_proc_cfg data with fixed native image/video classes."""

from ..native.kimi_k25.image_processing_kimi_k25 import Kimi_K25ImageProcessor
from ..native.kimi_k25.processing_kimi_k25 import Kimi_K25Processor as NativeKimiProcessor
from ..native.kimi_k25.video_processing_kimi_k25 import Kimi_K25VideoProcessor


def _legacy_media_options(data):
    for name in ("fixed_output_tokens", "in_patch_limit_video", "max_num_frames_each_video"):
        if data.get(name) is not None:
            raise ValueError(f"Native Kimi-K2.5 media loading does not support {name}={data[name]!r}")
    if data.get("timestamp_mode", "hh:mm:ss.fff") != "hh:mm:ss.fff":
        raise ValueError("Native Kimi-K2.5 media loading requires timestamp_mode='hh:mm:ss.fff'")
    common = {
        "patch_size": data["patch_size"],
        "merge_size": data["merge_kernel_size"],
        "image_mean": data["image_mean"],
        "image_std": data["image_std"],
        "size": {"max_height": data["patch_limit_on_one_side"], "max_width": data["patch_limit_on_one_side"]},
    }
    return (
        dict(common, max_patches=data["in_patch_limit"]),
        dict(common, max_patches=data["in_patch_limit_each_frame"],
             temporal_patch_size=data["temporal_merge_kernel_size"], fps=data["sample_fps"]),
    )


class Kimi_K25Processor(NativeKimiProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        from .tokenization_kimi import TikTokenTokenizer

        if isinstance(tokenizer, TikTokenTokenizer):
            # Native Transformers distinguishes video from image placeholders; released vocabulary does not.
            tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|kimi_k25_video_placeholder|>"]},
                replace_extra_special_tokens=False,
            )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        config = kwargs.pop("config", None)
        tokenizer = kwargs.pop("tokenizer", None)
        image_processor = kwargs.pop("image_processor", None)
        video_processor = kwargs.pop("video_processor", None)
        if tokenizer is None:
            from tokenicer import Tokenicer

            from .native import Kimi_K25Config
            from .tokenization_kimi import TikTokenTokenizer

            if config is None:
                config = Kimi_K25Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = TikTokenTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = Tokenicer.load(tokenizer, model_config=config, trust_remote_code=False)

        if image_processor is None or video_processor is None:
            image_data, _ = Kimi_K25ImageProcessor.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
            legacy = image_data.get("media_proc_cfg")
            if legacy is not None:
                image_options, video_options = _legacy_media_options(legacy)
                if image_processor is None:
                    image_processor = Kimi_K25ImageProcessor(**image_options)
                if video_processor is None:
                    video_processor = Kimi_K25VideoProcessor(**video_options)
            else:
                if image_processor is None:
                    image_processor = Kimi_K25ImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
                if video_processor is None:
                    video_processor = Kimi_K25VideoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return [image_processor, tokenizer, video_processor]
