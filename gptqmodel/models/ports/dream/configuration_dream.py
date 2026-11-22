# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
# Modified by ModelCloud.ai for GPT-QModel: Transformers 5 RoPE metadata and geometry validation.
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
"""Dream model configuration"""

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class DreamConfig(PreTrainedConfig):
    model_type = "Dream"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,  # cache not used in diffusion
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        mask_token_id=151666,
        pad_token_id=151643,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        # Preserve legacy checkpoint metadata; TF5 RoPE utilities consume rope_parameters.
        # BC: if there is a 'type' field, move it to 'rope_type'.
        self.rope_parameters = dict(rope_parameters or rope_scaling or {})
        self.rope_parameters.setdefault("rope_type", self.rope_parameters.pop("type", "default"))
        self.rope_parameters.setdefault("rope_theta", rope_theta)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 2:
            raise ValueError("Dream requires an even head dimension dividing hidden_size")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("Dream attention heads must be divisible by key/value heads")
