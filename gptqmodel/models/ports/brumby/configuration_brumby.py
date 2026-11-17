# Copyright 2024 HuggingFace Inc. team.
# Copyright 2025 Manifest AI.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by ModelCloud.ai, 2026: Transformers 5 config and explicit validation.
"""Brumby config adapted from the pinned Manifest AI source in PROVENANCE.md."""

from transformers.configuration_utils import PreTrainedConfig


class BrumbyConfig(PreTrainedConfig):
    model_type = "brumby"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=40,
        layer_types=None,
        chunk_size=512,
        switch_over_seq_len=8192,
        prefill_chunk_size=1024,
        use_exp=False,
        p=2,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.layer_types = layer_types or [
            "sliding_attention" if self.sliding_window is not None and i >= max_window_layers else "full_attention"
            for i in range(num_hidden_layers)
        ]
        self.chunk_size = chunk_size
        self.prefill_chunk_size = prefill_chunk_size
        self.switch_over_seq_len = switch_over_seq_len
        self.use_exp = use_exp
        self.p = p
        self.rope_parameters = dict(rope_parameters or rope_scaling or {})
        self.rope_parameters.setdefault("rope_type", self.rope_parameters.pop("type", "default"))
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        if p != 2:
            raise ValueError("Brumby supports only degree-2 power retention")
        if num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("Brumby head_dim must be positive and even")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive or None")
        if len(self.layer_types) != num_hidden_layers or set(self.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("layer_types must describe every Brumby decoder layer")
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
