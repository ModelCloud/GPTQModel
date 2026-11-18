# Modifications copyright(c) 2025 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: LicenseRef-Instella-Research-Only-RAIL-MS
# Modified for Transformers 5 by ModelCloud.ai. See LICENSE and NOTICES.
"""Instella configuration adapted from the pinned AMD checkpoint implementation."""

from transformers import PreTrainedConfig


class InstellaConfig(PreTrainedConfig):
    model_type = "instella"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=50304, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None,
        hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02,
        use_cache=True, pad_token_id=1, bos_token_id=None, eos_token_id=50279,
        tie_word_embeddings=False, rope_theta=10000.0,
        attention_bias=False, attention_dropout=0.0, rms_norm_eps=1e-5, **kwargs,
    ):
        # RoPE is stored as standardized `rope_parameters`. Released Instella checkpoints carry
        # the raw `rope_scaling` field instead; it is converted here, once, at parse time.
        rope = kwargs.pop("rope_parameters", None)
        checkpoint_rope = kwargs.pop("rope_scaling", None)
        if rope is not None:
            rope_theta = rope.get("rope_theta", rope_theta)
        else:
            rope = checkpoint_rope
        kind = rope.get("rope_type", rope.get("type")) if isinstance(rope, dict) else None
        if kind == "default":
            rope = None
        super().__init__(
            pad_token_id=pad_token_id, bos_token_id=bos_token_id,
            eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 2:
            raise ValueError("Instella requires an even head dimension and hidden_size divisible by num_attention_heads")
        if self.num_key_value_heads <= 0 or num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        self.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        if rope is not None:
            factor = rope.get("factor") if isinstance(rope, dict) else None
            if kind not in ("linear", "dynamic") or not isinstance(factor, (float, int)) or factor <= 1:
                raise ValueError("Instella rope_scaling requires type linear/dynamic and factor > 1")
            if kind == "dynamic" and hidden_size // num_attention_heads <= 2:
                raise ValueError("Dynamic NTK scaling requires head_dim > 2")
            self.rope_parameters = {"rope_type": kind, "rope_theta": rope_theta, "factor": factor}
