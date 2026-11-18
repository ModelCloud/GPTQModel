# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Transformers 5 by ModelCloud.ai.
"""InternLM2 and InternLM2.5 checkpoint configuration."""

import copy
import math

from transformers import PreTrainedConfig


class InternLM2Config(PreTrainedConfig):
    model_type = "internlm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=103168, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None,
        hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02,
        rms_norm_eps=1e-6, use_cache=True, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, bias=True,
        rope_theta=10000, rope_scaling=None, rope_parameters=None,
        attn_implementation=None, **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.bias = bias
        self.rope_theta = rope_theta
        self.rope_parameters = copy.deepcopy(rope_parameters if rope_parameters is not None else rope_scaling)
        super().__init__(
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings, attn_implementation=attn_implementation or "eager", **kwargs,
        )
        if num_attention_heads <= 0 or hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_key_value_heads <= 0 or num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if hidden_size // num_attention_heads % 2:
            raise ValueError("InternLM2 rotary head dimension must be even")
        if pretraining_tp < 1 or hidden_size % pretraining_tp or vocab_size % pretraining_tp:
            raise ValueError("pretraining_tp must divide hidden_size and vocab_size")
        if self.rope_parameters is not None:
            kind = self.rope_parameters.get("rope_type", self.rope_parameters.get("type", "default"))
            if kind not in ("default", "linear", "dynamic"):
                raise ValueError(f"Unsupported InternLM2 rope type: {kind}")
            factor = self.rope_parameters.get("factor", 1.0)
            if not isinstance(factor, (float, int)) or not math.isfinite(factor) or factor < 1:
                raise ValueError("InternLM2 rope factor must be finite and >= 1")
            if kind == "dynamic" and hidden_size // num_attention_heads <= 2:
                raise ValueError("Dynamic NTK requires a head dimension greater than 2")
