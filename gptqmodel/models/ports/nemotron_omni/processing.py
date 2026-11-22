# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bind native components directly; metadata never selects processor code."""

from transformers import TokenizersBackend

from ..native.nemotron_h_omni.processing_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3Processor as NativeProcessor,
)
from .configuration import NemotronOmniConfig


NemotronOmniTokenizer = TokenizersBackend


class NemotronOmniProcessor(NativeProcessor):
    @classmethod
    def from_pretrained(cls, path, *, config=None, tokenizer=None, **kwargs):
        # The media components import torchvision. Keep the tokenizer/config
        # importable in a base install by resolving them only for processor loads.
        from ..native.nemotron_h_omni.image_processing_nemotron_h_omni import (
            NemotronH_Omni_Reasoning_V3ImageProcessor,
        )
        from ..native.nemotron_h_omni.video_processing_nemotron_h_omni import (
            NemotronH_Omni_Reasoning_V3VideoProcessor,
        )
        from ..native.parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor

        for name in ("trust_remote_code", "code_revision", "_from_auto"):
            kwargs.pop(name, None)
        config = config or NemotronOmniConfig.from_pretrained(path, **kwargs)
        if tokenizer is None:
            tokenizer = TokenizersBackend.from_pretrained(path, **kwargs)
        image = NemotronH_Omni_Reasoning_V3ImageProcessor.from_pretrained(path, **kwargs)
        video = NemotronH_Omni_Reasoning_V3VideoProcessor.from_pretrained(
            path, norm_mean=image.image_mean, norm_std=image.image_std,
            patch_size=image.patch_size, downsample_ratio=config.downsample_ratio,
            video_target_num_patches=getattr(config.vision_config, "video_target_num_patches", 1024),
            **kwargs,
        )
        audio = config.audio_config
        feature_extractor = None
        if audio is not None:
            feature_extractor = ParakeetFeatureExtractor(
                feature_size=audio.num_mel_bins, sampling_rate=getattr(audio, "sampling_rate", 16000),
            )
        return cls(
            image_processor=image, video_processor=video, tokenizer=tokenizer,
            feature_extractor=feature_extractor, chat_template=tokenizer.chat_template,
            video_temporal_patch_dim=config.video_temporal_patch_size,
            audio_sampling_rate=getattr(audio, "sampling_rate", 16000),
            audio_subsampling_factor=getattr(audio, "subsampling_factor", 8),
        )
