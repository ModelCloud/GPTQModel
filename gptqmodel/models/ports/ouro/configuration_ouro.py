# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2025 ByteDance and the Ouro authors.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ouro config adapted from the Apache-2.0 ByteDance release."""

from ..native.qwen2.configuration_qwen2 import Qwen2Config


class OuroConfig(Qwen2Config):
    model_type = "ouro"

    def __init__(
        self,
        total_ut_steps=4,
        early_exit_threshold=1.0,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=32768,
        max_window_layers=28,
        **kwargs,
    ):
        if not isinstance(total_ut_steps, int) or isinstance(total_ut_steps, bool) or total_ut_steps < 1:
            raise ValueError("total_ut_steps must be a positive integer")
        self.total_ut_steps = total_ut_steps
        self.early_exit_threshold = early_exit_threshold
        kwargs.setdefault("rope_theta", 10000.0)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("pad_token_id", 2)
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            **kwargs,
        )
        self.head_dim = getattr(self, "head_dim", None) or hidden_size // num_attention_heads
        if num_hidden_layers < 1 or self.head_dim % 2 or num_attention_heads % self.num_key_value_heads:
            raise ValueError("Ouro requires positive depth, even head_dim and divisible Q/KV head counts")
        if len(self.layer_types) != num_hidden_layers or set(self.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("Ouro layer_types must describe each physical attention layer")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("sliding_attention requires a positive sliding_window")
        if self.rope_parameters.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("Released Ouro attention rotates the complete head")


__all__ = ["OuroConfig"]
