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

import math

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring

from .configuration_lfm3 import Lfm3Config


@auto_docstring(checkpoint="LiquidAI/LFM3-Audio-2B-Preview")
@strict
class LiquidCodecConfig(PreTrainedConfig):
    r"""
    Configuration of the [`LiquidCodecModel`] bundled with LFM3-Audio-2B and LFM3-Omni-2B in their
    `audio_tokenizer/` directory.

    encoder_dim (`int`, *optional*, defaults to 512):
        The width of the encoder and decoder LFM3 models.
    encoder_conv_dim (`int`, *optional*, defaults to 256):
        The number of channels of the STFT front end.
    stft_n_fft (`int`, *optional*, defaults to 1280):
        The FFT size of the analysis and synthesis transforms.
    stft_hop_length (`int`, *optional*, defaults to 320):
        The hop of the analysis and synthesis transforms.
    stft_win_length (`int`, *optional*, defaults to 1280):
        The window length of the analysis and synthesis transforms.
    stft_frame_stack (`int`, *optional*, defaults to 6):
        The number of STFT frames per codec frame.
    quantizer_dim (`int`, *optional*, defaults to 256):
        The width of the latent handed to the quantizer.
    fsq_levels (`list[int]`, *optional*, defaults to `[8, 7, 6, 6]`):
        The number of levels of each finite-scalar-quantization dimension; their product is the codebook size.
    n_q (`int`, *optional*, defaults to 12):
        The number of codebooks.
    decoder_window (`int`, *optional*, defaults to 30):
        The sliding attention window of the decoder LFM3 model, in 75 Hz frames.
    sampling_rate (`int`, *optional*, defaults to 24000):
        The sampling rate of the audio.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated normal initializer of the weights.
    """

    model_type = "liquid_codec"

    encoder_dim: int = 512
    encoder_conv_dim: int = 256
    stft_n_fft: int = 1280
    stft_hop_length: int = 320
    stft_win_length: int = 1280
    stft_frame_stack: int = 6
    quantizer_dim: int = 256
    fsq_levels: list[int] | None = None
    n_q: int = 12
    decoder_window: int = 30
    sampling_rate: int = 24_000
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        # the released configs nest the fields under `model` and dump the two LFM3 sub-configs
        dropped = ("encoder_lfm_config", "decoder_lfm_config", "n_output_samples")
        for key, value in kwargs.pop("model", {}).items():
            if key in dropped:
                continue
            if key not in self.__dataclass_fields__:
                raise ValueError(f"Unknown Liquid codec setting `{key}`")
            setattr(self, key, value)
        for key in dropped:
            kwargs.pop(key, None)
        if self.fsq_levels is None:
            self.fsq_levels = [8, 7, 6, 6]
        self.fsq_levels = list(self.fsq_levels)
        super().__post_init__(**kwargs)

    @property
    def codebook_size(self) -> int:
        return math.prod(self.fsq_levels)

    @property
    def frame_size(self) -> int:
        return self.stft_hop_length * self.stft_frame_stack

    def lfm_config(self, sliding_window: int | None = None) -> PreTrainedConfig:
        attention_type = "full_attention" if sliding_window is None else "sliding_attention"
        return Lfm3Config(
            # the backbones are not sub-configs, so the requested attention implementation is handed over here
            attn_implementation=self._attn_implementation,
            vocab_size=1,
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
            hidden_size=self.encoder_dim,
            intermediate_size=2304,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=131_072,
            norm_eps=1e-5,
            conv_L_cache=7,
            conv_bias=False,
            use_sandwich_norm=True,
            layer_types=[attention_type if i in (2, 4, 6) else "conv" for i in range(8)],
            sliding_window=sliding_window,
            rope_parameters={
                attention_type: {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
            },
            use_cache=False,
            tie_word_embeddings=False,
        )


__all__ = ["LiquidCodecConfig"]
