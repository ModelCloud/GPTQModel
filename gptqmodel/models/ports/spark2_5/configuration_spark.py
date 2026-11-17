# Modified by ModelCloud.ai: validated package-local Transformers 5 configuration.
# coding=utf-8
# Copyright 2026 The XHToken team and the HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy

from transformers import PreTrainedConfig


class Spark2_5Config(PreTrainedConfig):
    model_type = "spark2_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_k_v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.out_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embedding": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        headwise_attn_output_gate=False,
        gate_attn_act_mode="sigmoid",
        sliding_window=None,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_key_value_heads < 1 or num_attention_heads < 1:
            raise ValueError("Spark2.5 attention head counts must be positive")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
            )
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.gate_attn_act_mode = gate_attn_act_mode
        self.sliding_window = sliding_window
        self.rope_parameters = deepcopy(rope_parameters) if rope_parameters is not None else {}

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match num_hidden_layers ({num_hidden_layers})"
            )
        self.layer_types = list(layer_types)
        if any(t not in ("full_attention", "sliding_attention") for t in self.layer_types):
            raise ValueError("Spark2.5 layer_types must be full_attention or sliding_attention")
        if "sliding_attention" in self.layer_types and (sliding_window is None or sliding_window < 1):
            raise ValueError("sliding_attention requires a positive sliding_window")
        if (
            min(
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                self.head_dim,
            )
            < 1
        ):
            raise ValueError("Spark2.5 dimensions must be positive")
        if hidden_act != "gelu":
            raise ValueError("Spark2.5 supports hidden_act='gelu' only")
        if gate_attn_act_mode not in ("sigmoid", "silu"):
            raise ValueError("gate_attn_act_mode must be sigmoid or silu")
        for layer_type in set(self.layer_types):
            params = self.rope_parameters.get(layer_type, {})
            unknown = set(params) - {"rope_theta", "partial_rotary_factor", "rope_type"}
            if unknown or params.get("rope_type", "default") != "default":
                raise ValueError("Spark2.5 supports unscaled per-layer RoPE only")
            fraction = self.get_partial_rotary_factor(layer_type)
            rotary_dim = int(self.head_dim * fraction)
            if not 0 < fraction <= 1 or rotary_dim < 2 or rotary_dim % 2:
                raise ValueError("partial_rotary_factor must produce a positive even rotary dimension")
            if self.get_rope_theta(layer_type) <= 0:
                raise ValueError("rope_theta must be positive")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_rope_theta(self, layer_type):
        params = self.rope_parameters.get(layer_type, {})
        return params.get("rope_theta", 10000)

    def get_partial_rotary_factor(self, layer_type):
        params = self.rope_parameters.get(layer_type, {})
        return params.get("partial_rotary_factor", 1.0)


__all__ = ["Spark2_5Config"]
