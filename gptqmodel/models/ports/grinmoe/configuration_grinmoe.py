# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch GRINMoE model"""

from transformers.configuration_utils import PretrainedConfig


class GRINMoEConfig(PretrainedConfig):
    model_type = "grinmoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=6400,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=16,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.01,
        input_jitter_noise=0.01,
        attention_bias = False,
        lm_head_bias = False,
        **kwargs,
    ):
        if num_experts_per_tok != 2 or num_local_experts < 2:
            raise ValueError("GRIN-MoE SparseMixer requires top-2 routing and at least two experts")
        kv_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        if hidden_size % num_attention_heads or num_attention_heads % kv_heads:
            raise ValueError("GRIN-MoE requires divisible attention and key/value head dimensions")
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("GRIN-MoE rotary head dimension must be even")
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
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

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.input_jitter_noise = input_jitter_noise

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # Upstream keeps a full KV cache, including when eager attention masks a window.
        self.layer_types = ["full_attention"] * num_hidden_layers
