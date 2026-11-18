#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Union

from transformers import PretrainedConfig, Qwen3Config, WhisperConfig
from transformers.utils import logging

from .modeling_navit_siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)


class MiniCPMVSliceConfig(PretrainedConfig):
    model_type = "minicpmv"

    def __init__(
        self,
        patch_size=14,
        max_slice_nums=9,
        scale_resolution=448,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "minicpmv":
            config_dict = config_dict["slice_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MiniCPMTTSConfig(PretrainedConfig):
    model_type = "minicpmtts"

    def __init__(
        self,
        llm_dim: int = 2560,
        llm_intermediate_size: int = 768,
        llm_down_scale: bool = False,
        llm_dim_model_base: int = 256,
        projector_type: str = "mlp",
        hidden_act: str = "silu",
        aug_loss_weight: bool = False,
        aug_layer_loss_weight: bool = False,
        filter_tts_loss: bool = False,
        tts_filter_loss_fix: bool = False,
        long_weight: float = 0.1,
        short_weight: float = 0.1,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 20,
        num_key_value_heads: int = 12,
        max_position_embeddings: int = 4096,
        num_audio_tokens: int = 4097,
        num_text_tokens: int = 21178,
        num_mel_bins: int = 100,
        num_vq: int = 1,
        use_llm_hidden_state: bool = False,
        audio_bos_token_id: int = 21132,
        text_eos_token_id: int = 21133,
        use_text: bool = True,
        streaming: bool = False,
        streaming_text_chunk_min: int = 3,
        streaming_text_chunk_max: int = 7,
        streaming_text_reserved_len: int = 300,
        streaming_audio_chunk_size: int = 50,
        attn_implementation: str = "sdpa",
        condition_type: str = "llm_hidden",
        backbone_model: str = "llama",
        audio_tokenizer_type: str = "wavtokenizer",
        audio_tokenizer_sample_rate: int = 24000,
        streaming_sliding_window: bool = False,
        streaming_sliding_window_max_text_len: int = 500,
        streaming_sliding_window_average_speed: int = 5,
        streaming_sliding_window_fast_speed: int = 7,
        streaming_sliding_window_slow_speed: int = 3,
        streaming_sliding_window_audio_frame_rate: int = 50,
        streaming_sliding_window_audio_init_text_length: int = 10,
        streaming_sliding_window_audio_window_size: int = 300,
        normalize_projected_hidden: bool = False,
        interleaved: bool = False,
        attention_type: str = "sliding_recompute",
        recomputed_chunks: int = 1,
        window_size: int = 2,
        **kwargs,
    ):
        self.top_p = kwargs.pop("top_p", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        super().__init__(**kwargs)

        self.llm_dim = llm_dim
        self.llm_hidden_size = llm_dim
        self.llm_intermediate_size = llm_intermediate_size
        self.llm_down_scale = llm_down_scale
        self.llm_dim_model_base = llm_dim_model_base
        self.projector_type = projector_type
        self.aug_loss_weight = aug_loss_weight
        self.aug_layer_loss_weight = aug_layer_loss_weight
        self.tts_filter_loss_fix = tts_filter_loss_fix
        self.filter_tts_loss = filter_tts_loss
        self.long_weight = long_weight
        self.short_weight = short_weight
        self.hidden_act = hidden_act

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens
        self.num_mel_bins = num_mel_bins
        self.num_vq = num_vq
        self.use_llm_hidden_state = use_llm_hidden_state
        self.audio_bos_token_id = audio_bos_token_id
        self.text_eos_token_id = text_eos_token_id
        self.use_text = use_text
        self.streaming = streaming
        self.streaming_text_chunk_min = streaming_text_chunk_min
        self.streaming_text_chunk_max = streaming_text_chunk_max
        self.streaming_text_reserved_len = streaming_text_reserved_len
        self.streaming_audio_chunk_size = streaming_audio_chunk_size
        self.attn_implementation = attn_implementation
        self.condition_type = condition_type
        self.backbone_model = backbone_model
        self.audio_tokenizer_type = audio_tokenizer_type
        self.audio_tokenizer_sample_rate = audio_tokenizer_sample_rate

        self.streaming_sliding_window = streaming_sliding_window
        self.streaming_sliding_window_max_text_len = streaming_sliding_window_max_text_len
        self.streaming_sliding_window_average_speed = streaming_sliding_window_average_speed
        self.streaming_sliding_window_fast_speed = streaming_sliding_window_fast_speed
        self.streaming_sliding_window_slow_speed = streaming_sliding_window_slow_speed
        self.streaming_sliding_window_audio_frame_rate = streaming_sliding_window_audio_frame_rate
        self.streaming_sliding_window_audio_init_text_length = streaming_sliding_window_audio_init_text_length
        self.streaming_sliding_window_audio_window_size = streaming_sliding_window_audio_window_size

        self.normalize_projected_hidden = normalize_projected_hidden

        self.interleaved = interleaved
        self.attention_type = attention_type
        self.recomputed_chunks = recomputed_chunks
        self.window_size = window_size


class MiniCPMOConfig(Qwen3Config):
    model_type = "minicpmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 980,
        "intermediate_size": 4304,
        "model_type": "siglip",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    def __init__(
        self,
        use_cache=True,
        query_num=64,
        image_size=448,
        drop_vision_last_layer=True,
        batch_vision_input=True,
        slice_config=None,
        vision_config=None,
        audio_config=None,
        tts_config=None,
        use_image_id=True,
        vision_batch_size=16,
        audio_pool_step=None,
        audio_chunk_length=1.0,
        stream_input=False,
        listen_speak_type="asr",
        init_vision=True,
        init_audio=True,
        init_tts=True,
        **kwargs,
    ):
        self.version = str(kwargs.pop("version", "4.5"))
        if self.version not in ("4.5", "2.6"):
            raise ValueError(f"Unsupported MiniCPM-o version: {self.version}")
        for name, value, expected in (
            ("vision", vision_config, ("siglip", "siglip_vision_model")),
            ("audio", audio_config, ("whisper",)),
            ("tts", tts_config, ("minicpmtts",) if self.version == "4.5" else ("conditional_chattts",)),
        ):
            if isinstance(value, dict) and value.get("model_type", expected[0]) not in expected:
                raise ValueError(f"Foreign MiniCPM-o {name} architecture")
        self.use_cache = use_cache
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        self.vision_batch_size = vision_batch_size
        self.audio_pool_step = audio_pool_step if audio_pool_step is not None else (2 if self.version == "2.6" else 5)
        self.chunk_input = kwargs.pop("chunk_input", False)
        self.audio_chunk_length = audio_chunk_length
        self.stream_input = stream_input
        self.listen_speak_type = listen_speak_type

        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts

        if isinstance(slice_config, MiniCPMVSliceConfig):
            slice_config = slice_config.to_dict()
        if slice_config is None:
            self.slice_config = MiniCPMVSliceConfig(max_slice_nums=1)
        else:
            self.slice_config = MiniCPMVSliceConfig(**slice_config)
        self.slice_mode = True

        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if vision_config is None:
            self.vision_config = SiglipVisionConfig(**self.default_vision_config)
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif isinstance(vision_config, SiglipVisionConfig):
            self.vision_config = vision_config

        if audio_config is None:
            self.audio_config = WhisperConfig()
        elif isinstance(audio_config, dict):
            self.audio_config = WhisperConfig(**audio_config)
        elif isinstance(audio_config, WhisperConfig):
            self.audio_config = audio_config

        if self.version == "2.6":
            from .tts_26 import ConditionalChatTTSConfig

            self.tts_config = ConditionalChatTTSConfig(**(tts_config or {}))
        elif tts_config is None:
            self.tts_config = MiniCPMTTSConfig()
        elif isinstance(tts_config, dict):
            self.tts_config = MiniCPMTTSConfig(**tts_config)
        elif isinstance(tts_config, MiniCPMTTSConfig):
            self.tts_config = tts_config

        self.patch_size = self.vision_config.patch_size

        super().__init__(**kwargs)
