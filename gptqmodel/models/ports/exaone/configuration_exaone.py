# Copyright 2021 The LG AI Research EXAONE Lab. All rights reserved.
# Copyright 2026 The LG AI Research and HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""EXAONE 3 configuration adapted for Transformers 5; see PROVENANCE.md."""

from copy import deepcopy

from transformers import PretrainedConfig


class ExaoneConfig(PretrainedConfig):
    model_type = "exaone"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "hidden_act": "activation_function",
        "rms_norm_eps": "layer_norm_epsilon",
    }

    def __init__(
        self, vocab_size=102400, max_position_embeddings=2048, hidden_size=2048,
        num_layers=32, num_attention_heads=32, num_key_value_heads=None,
        intermediate_size=None, activation_function="silu", rope_theta=10000.0,
        rope_scaling=None, rope_parameters=None, embed_dropout=0.0,
        attention_dropout=0.0, layer_norm_epsilon=1e-5, initializer_range=0.02,
        use_cache=True, bos_token_id=0, eos_token_id=2, pad_token_id=None,
        tie_word_embeddings=True, **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = kwargs.pop("num_hidden_layers", num_layers)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = kwargs.pop("head_dim", hidden_size // num_attention_heads)
        if hidden_size % num_attention_heads or self.head_dim != hidden_size // num_attention_heads:
            raise ValueError("EXAONE 3 requires head_dim = hidden_size / num_attention_heads")
        if self.head_dim % 2 or num_attention_heads % self.num_key_value_heads:
            raise ValueError("EXAONE 3 requires even head_dim and divisible grouped-query heads")
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.activation_function = kwargs.pop("hidden_act", activation_function)
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = kwargs.pop("rms_norm_eps", layer_norm_epsilon)
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        parameters = deepcopy(rope_parameters if rope_parameters is not None else rope_scaling) or {}
        parameters.setdefault("rope_type", parameters.pop("type", "default"))
        parameters.setdefault("rope_theta", rope_theta)
        if parameters["rope_type"] not in ("default", "linear", "dynamic", "yarn", "longrope", "llama3"):
            raise ValueError(f"Unsupported EXAONE RoPE type: {parameters['rope_type']}")
        if parameters["rope_type"] == "dynamic" and self.head_dim <= 2:
            raise ValueError("EXAONE dynamic RoPE requires head_dim > 2")
        self.rope_parameters = parameters
        kwargs.update(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                      pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings)
        super().__init__(**kwargs)
