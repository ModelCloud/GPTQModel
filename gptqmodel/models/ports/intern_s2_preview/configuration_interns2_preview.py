# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
# Pinned upstream: see NOTICE. The text/vision configs are the native Qwen3.5-MoE ones.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig


# The released text_config declares model_type="qwen3_5_moe_text"; its fields,
# defaults and RoPE validation are exactly native Qwen3.5-MoE.
InternS2PreviewTextConfig = Qwen3_5MoeTextConfig


class InternS2PreviewVisionConfig(Qwen3_5MoeVisionConfig):
    """Native Qwen3.5-MoE vision fields under the released vision model_type."""

    model_type = "intern_s2_preview"


class InternS2PreviewTimeSeriesConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternS2PreviewTimeSeriesModel`]. It is used to instantiate a
    InternS2PreviewTimeSeries model according to the specified arguments, defining the model architecture.

    Args:
        ts_adapt_in_dim (`int`, *optional*, defaults to 256):
            The input dimension of the time series adapter.
        ts_adapt_out_dim (`int`, *optional*, defaults to 1024):
            The output dimension of the time series adapter.
        ts_hidden_dim (`int`, *optional*, defaults to 1024):
            The hidden dimension of the time series model.
        ts_cnn_channels (`list[int]`, *optional*, defaults to [1, 32, 64, 128, 128]):
            The channels of the time series CNN.
        ts_cnn_kernel_sizes (`list[int]`, *optional*, defaults to [3, 5, 5, 5]):
            The kernel sizes of the time series CNN.
        ts_cnn_strides (`list[int]`, *optional*, defaults to [2, 4, 4, 5]):
            The strides of the time series CNN.
        ts_cnn_paddings (`list[int]`, *optional*, defaults to [1, 2, 2, 2]):
            The paddings of the time series CNN.
        ts_concat_subsampling_in_channels (`int`, *optional*, defaults to 128):
            The input channels of the time series concat subsampling.
        ts_concat_subsampling_concat_size (`int`, *optional*, defaults to 2):
            The concat size of the time series concat subsampling.
        **super_kwargs:
            Additional keyword arguments passed along to the base class `WhisperConfig`.
    """

    model_type = "interns2_preview_time_series"
    base_config_key = "ts_config"

    def __init__(
        self,
        activation_dropout: float = 0.0,
        activation_function: str = "gelu",
        attention_dropout: float = 0.0,
        d_model: int = 768,
        dropout: float = 0.0,
        encoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 3072,
        encoder_layerdrop: float = 0.0,
        encoder_layers: int = 17,
        max_source_positions: int = 1500,
        num_mel_bins: int = 80,
        out_hidden_size: int = 2048,
        scale_embedding: bool = False,
        ts_adapt_in_dim: int = 256,
        ts_adapt_out_dim: int = 1024,
        ts_hidden_dim: int = 1024,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)

        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.attention_dropout = attention_dropout
        self.d_model = d_model
        self.dropout = dropout
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layerdrop = encoder_layerdrop
        self.encoder_layers = encoder_layers
        self.max_source_positions = max_source_positions
        self.num_mel_bins = num_mel_bins
        self.out_hidden_size = out_hidden_size
        self.scale_embedding = scale_embedding
        self.ts_adapt_in_dim = ts_adapt_in_dim
        self.ts_adapt_out_dim = ts_adapt_out_dim
        self.ts_hidden_dim = ts_hidden_dim

        assert self.ts_adapt_out_dim == self.ts_hidden_dim, "ts_adapt_out_dim should be equal to ts_hidden_dim"


class InternS2PreviewConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternS2PreviewModel`]. It is used to instantiate a
    Qwen3.5-MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3.5-35B-A3B-Instruct [Qwen/Qwen3.5-35B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3_5TextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3_5VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 248056):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 248057):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 248053):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 248054):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import InternS2PreviewForConditionalGeneration, InternS2PreviewConfig

    >>> # Initializing a Qwen3.5-MoE style configuration
    >>> configuration = InternS2PreviewConfig()

    >>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
    >>> model = InternS2PreviewForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "intern_s2_preview"
    sub_configs = {
        "vision_config": InternS2PreviewVisionConfig,
        "text_config": InternS2PreviewTextConfig,
        "ts_config": InternS2PreviewTimeSeriesConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        ts_config=None,
        ts_token_id=248093,
        ts_start_id=248091,
        ts_end_id=248092,
        **kwargs,
    ):
        if isinstance(ts_config, dict):
            self.ts_config = self.sub_configs["ts_config"](**ts_config)
        elif ts_config is None:
            self.ts_config = self.sub_configs["ts_config"]()
        else:
            self.ts_config = ts_config

        self.ts_token_id = ts_token_id
        self.ts_start_id = ts_start_id
        self.ts_end_id = ts_end_id
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)
        self.architectures = ["InternS2PreviewForConditionalGeneration"]


__all__ = ["InternS2PreviewConfig", "InternS2PreviewTextConfig", "InternS2PreviewVisionConfig"]
