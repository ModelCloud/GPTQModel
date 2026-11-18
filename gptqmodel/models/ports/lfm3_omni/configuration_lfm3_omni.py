# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# GPT-QModel adaptation: package-owned loading and Transformers 5 execution.
# Pinned source and modifications: NOTICE.
from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.utils import auto_docstring

from .configuration_lfm3 import Lfm3Config
from .configuration_lfm3_vl import Lfm3VlVisionConfig


def _config_class(model_type):
    classes = {"lfm2": Lfm2Config, "lfm3": Lfm3Config, "lfm3_vl_vision": Lfm3VlVisionConfig,
               "lfm3_omni_encoder": Lfm3OmniEncoderConfig}
    if model_type not in classes:
        raise ValueError(f"Unsupported LFM3 Omni component model_type: {model_type!r}")
    return classes[model_type]


# `dtype` is dropped so that the top-level dtype governs every component of a legacy export
_LEGACY_SUBCONFIG_DROPPED_KEYS = (
    "architectures",
    "transformers_version",
    "is_causal",
    "auto_map",
    "_name_or_path",
    "dtype",
)


def _subconfig_dict(values: dict, default_model_type: str) -> dict:
    if values.get("model_type", default_model_type) != default_model_type:
        raise ValueError(f"Expected {default_model_type} component, got {values.get('model_type')!r}")
    values = {key: value for key, value in values.items() if key not in _LEGACY_SUBCONFIG_DROPPED_KEYS}
    values.setdefault("model_type", default_model_type)
    return values


@auto_docstring(checkpoint="LiquidAI/LFM3-Omni-2B-Preview")
@strict
class Lfm3OmniEncoderConfig(PreTrainedConfig):
    r"""
    text_config (`Lfm2Config` or `dict`, *optional*):
        The configuration of the LFM2 model run over the downsampled spectrogram. Defaults to the 17-layer,
        768-wide model of LFM3-Audio.
    num_mel_bins (`int`, *optional*, defaults to 128):
        The number of mel bins of the input features.
    conv_channels (`int`, *optional*, defaults to 256):
        The number of channels of the downsampling convolutions.
    hidden_size (`int`, *optional*, defaults to 768):
        The width of the encoder output, equal to `text_config.hidden_size`.
    sliding_window (`int`, *optional*, defaults to 32):
        The number of past frames each frame attends to, itself included.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated normal initializer of the convolutions.
    """

    model_type = "lfm3_omni_encoder"
    sub_configs = {"text_config": Lfm2Config}

    text_config: dict | PreTrainedConfig | None = None
    num_mel_bins: int = 128
    conv_channels: int = 256
    hidden_size: int = 768
    sliding_window: int = 32
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            # the encoder consumes embeddings only: no end-of-sequence id in its two-token vocabulary
            self.text_config = {"eos_token_id": None, **_subconfig_dict(self.text_config, "lfm2")}
            self.text_config = _config_class(self.text_config["model_type"])(**self.text_config)
        elif self.text_config is None:
            self.text_config = _config_class("lfm2")(
                vocab_size=2,
                hidden_size=self.hidden_size,
                intermediate_size=2560,
                num_hidden_layers=17,
                num_attention_heads=8,
                num_key_value_heads=4,
                layer_types=(["conv", "conv", "full_attention"] * 5) + ["full_attention", "full_attention"],
                conv_L_cache=3,
                rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=None,
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="LiquidAI/LFM3-Omni-2B-Preview")
@strict
class Lfm3OmniConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Lfm3OmniForConditionalGeneration`].

    text_config (`Lfm3Config` or `dict`, *optional*):
        The configuration of the LFM3 language model.
    audio_encoder_config (`Lfm3OmniEncoderConfig` or `dict`, *optional*):
        The configuration of the speech encoder.
    depth_config (`Lfm3Config` or `dict`, *optional*):
        The configuration of the depth decoder, a bidirectional LFM3 model over the codebook axis of one speech
        frame. Defaults to the 3-layer, 1024-wide model of LFM3-Audio.
    num_codebooks (`int`, *optional*, defaults to 12):
        The number of codebooks per speech frame.
    codebook_size (`int`, *optional*, defaults to 2017):
        The size of each codebook, the codec's codes plus the end-of-audio code.
    audio_eos_code (`int`, *optional*, defaults to 2016):
        The end-of-audio code. A frame holding it in every codebook ends a speech segment.
    maskgit_schedule (`list[int]`, *optional*, defaults to `[4, 4, 4]`):
        The number of codebooks revealed by each pass of the depth decoder when sampling a frame.
    interleaved_n_text (`int`, *optional*, defaults to 6):
        The number of text tokens per block of the interleaved generation mode.
    interleaved_n_audio (`int`, *optional*, defaults to 12):
        The number of speech frames per block of the interleaved generation mode.
    audio_token_id (`int`, *optional*, defaults to 125031):
        The id of the placeholder token standing for one encoded input speech frame.
    audio_out_token_id (`int`, *optional*, defaults to 125032):
        The id of the placeholder token standing for one output speech frame.
    audio_start_token_id (`int`, *optional*, defaults to 125012):
        The id of `<|audio_start|>`, which switches generation to speech until an end-of-audio frame.
    text_end_token_id (`int`, *optional*, defaults to 125014):
        The id of `<|text_end|>`, which ends the text of an interleaved answer.
    mixed_start_token_id (`int`, *optional*, defaults to 125015):
        The id of `<|mixed_start|>`, which switches generation to interleaved text and speech.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the language model head to the input embeddings.
    text_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the text loss in the training loss.
    audio_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the speech loss in the training loss.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated normal initializer of the projector and depth decoder.

    vision_config (`Lfm3VlVisionConfig` or `dict`, *optional*):
        The configuration of the image encoder.
    video_token_id (`int`, *optional*, defaults to 125029):
        The id of the placeholder token standing for one encoded video patch group.
    image_token_id (`int`, *optional*, defaults to 125030):
        The placeholder for one projected image token.
    projector_hidden_size (`int`, *optional*, defaults to 4096):
        The intermediate dimension of the image projector.
    projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The image projector activation.
    projector_bias (`bool`, *optional*, defaults to `True`):
        Whether the image projector linear layers have biases.
    projector_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon of the image projector layer normalization.
    downsample_factor (`int`, *optional*, defaults to 2):
        The side length of each image patch block merged into one language token.
    """

    model_type = "lfm3_omni"
    sub_configs = {
        "text_config": Lfm3Config,
        "audio_encoder_config": Lfm3OmniEncoderConfig,
        "depth_config": Lfm3Config,
        "vision_config": Lfm3VlVisionConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    audio_encoder_config: dict | PreTrainedConfig | None = None
    depth_config: dict | PreTrainedConfig | None = None
    num_codebooks: int = 12
    codebook_size: int = 2017
    audio_eos_code: int = 2016
    maskgit_schedule: list[int] | None = None
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    audio_token_id: int = 125_031
    audio_out_token_id: int = 125_032
    audio_start_token_id: int = 125_012
    text_end_token_id: int = 125_014
    mixed_start_token_id: int = 125_015
    tie_word_embeddings: bool = True
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    initializer_range: float = 0.02

    # the ids `generate` and `transcribe_live` index the text logits with
    _token_id_fields = (
        "audio_token_id",
        "audio_out_token_id",
        "audio_start_token_id",
        "text_end_token_id",
        "mixed_start_token_id",
    )
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 125_030
    video_token_id: int = 125_029
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    projector_layer_norm_eps: float = 1e-6
    downsample_factor: int = 2

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = Lfm3VlVisionConfig(**_subconfig_dict(self.vision_config, "lfm3_vl_vision"))
        elif self.vision_config is None:
            self.vision_config = _config_class("lfm3_vl_vision")()
        if "lfm" in kwargs or "format_version" in kwargs:
            raise ValueError(
                "LFM3-Omni reads the canonical checkpoint layout only; training exports (`lfm` / `format_version`) "
                "are not supported, load a canonical Hugging Face checkpoint."
            )
        if isinstance(self.text_config, dict):
            self.text_config = _subconfig_dict(self.text_config, "lfm3")
            self.text_config = _config_class(self.text_config["model_type"])(**self.text_config)
        elif self.text_config is None:
            self.text_config = _config_class("lfm3")()
        self.validate_vision_fields()
        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config = Lfm3OmniEncoderConfig(
                **_subconfig_dict(self.audio_encoder_config, "lfm3_omni_encoder"))
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = self.sub_configs["audio_encoder_config"]()
        if isinstance(self.depth_config, dict):
            # the depth decoder consumes embeddings only; its token ids are placeholders
            self.depth_config = {
                "pad_token_id": None,
                "bos_token_id": None,
                "eos_token_id": None,
                **_subconfig_dict(self.depth_config, "lfm3"),
            }
            self.depth_config = _config_class(self.depth_config["model_type"])(**self.depth_config)
        elif self.depth_config is None:
            self.depth_config = _config_class("lfm3")(
                vocab_size=1,
                pad_token_id=None,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=3,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=32,
                max_position_embeddings=self.num_codebooks,
                layer_types=["full_attention"] * 3,
                rope_parameters={
                    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
                },
                sliding_window=None,
                use_sandwich_norm=False,
                use_cache=False,
                bos_token_id=None,
                eos_token_id=None,
            )
        steps = kwargs.pop("maskgit_steps", 3)
        if self.maskgit_schedule is None:
            if not 1 <= steps <= self.num_codebooks:
                raise ValueError(
                    f"maskgit_steps must be between 1 and num_codebooks={self.num_codebooks}, got {steps}"
                )
            self.maskgit_schedule = [self.num_codebooks // steps] * steps
            self.maskgit_schedule[-1] += self.num_codebooks - sum(self.maskgit_schedule)
        self.maskgit_schedule = list(self.maskgit_schedule)
        self._validate_speech_fields()
        super().__post_init__(**kwargs)

    def validate_speech_fields(self):
        """Part of `@strict`-powered validation, so `validate()` and `save_pretrained` refuse later edits too."""
        self._validate_speech_fields()

    def _validate_speech_fields(self):
        if min(self.maskgit_schedule, default=0) < 1 or sum(self.maskgit_schedule) != self.num_codebooks:
            raise ValueError(
                f"maskgit_schedule {self.maskgit_schedule} must be positive counts summing to "
                f"num_codebooks={self.num_codebooks}"
            )
        # the end-of-audio code indexes every codebook; a code outside them could never be sampled or embedded
        if not 0 <= self.audio_eos_code < self.codebook_size:
            raise ValueError(
                f"audio_eos_code={self.audio_eos_code} must lie in [0, codebook_size={self.codebook_size})"
            )
        if self.interleaved_n_text < 1 or self.interleaved_n_audio < 1:
            raise ValueError("interleaved_n_text and interleaved_n_audio must be positive")
        vocab_size = self.text_config.vocab_size
        for name in self._token_id_fields:
            if not 0 <= getattr(self, name) < vocab_size:
                raise ValueError(f"{name}={getattr(self, name)} must lie in [0, text_config.vocab_size={vocab_size})")

    def validate_vision_fields(self):
        """Part of `@strict`-powered validation, so `validate()` and `save_pretrained` refuse later edits too."""
        if self.downsample_factor != self.vision_config.spatial_merge_size:
            raise ValueError("downsample_factor must match vision_config.spatial_merge_size")
        # an `inputs_embeds` forward embeds both placeholder ids to find them
        vocab_size = self.text_config.vocab_size
        for name in ("image_token_id", "video_token_id"):
            if not 0 <= getattr(self, name) < vocab_size:
                raise ValueError(f"{name}={getattr(self, name)} must lie in [0, text_config.vocab_size={vocab_size})")


__all__ = ["Lfm3OmniConfig", "Lfm3OmniEncoderConfig"]
