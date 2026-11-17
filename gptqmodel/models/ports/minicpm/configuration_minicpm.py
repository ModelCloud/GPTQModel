# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 OpenBMB. Adapted for Transformers 5 by ModelCloud.ai, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Original MiniCPM configuration; source revisions and attribution in README.md."""

import math

from transformers import PreTrainedConfig


class MiniCPMConfig(PreTrainedConfig):
    model_type = "minicpm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None,
        hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02,
        rms_norm_eps=1e-6, use_cache=True, pad_token_id=None, bos_token_id=1,
        eos_token_id=2, pretraining_tp=1, tie_word_embeddings=True,
        rope_theta=10000.0, rope_scaling=None, rope_parameters=None, attention_bias=False,
        attention_dropout=0.0, scale_emb=1, dim_model_base=1, scale_depth=1,
        num_experts=0, num_experts_per_tok=0, **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
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
        self.rope_theta = rope_theta
        self.rope_parameters = dict(rope_parameters or rope_scaling or {})
        self.rope_parameters.setdefault("rope_type", self.rope_parameters.get("type", "default"))
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_theta = self.rope_parameters["rope_theta"]
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "max_position_embeddings", "pretraining_tp"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 2:
            raise ValueError("MiniCPM needs an even head dimension dividing hidden_size")
        if num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        for value in (hidden_size, intermediate_size, self.num_key_value_heads * self.head_dim, vocab_size):
            if value % pretraining_tp:
                raise ValueError("pretraining_tp must divide projection dimensions and vocab_size")
        if num_experts < 0 or (num_experts and not 1 <= num_experts_per_tok <= num_experts):
            raise ValueError("MoE requires 1 <= num_experts_per_tok <= num_experts")
        if not num_experts and num_experts_per_tok:
            raise ValueError("num_experts_per_tok requires num_experts")
        for name in ("scale_emb", "dim_model_base", "scale_depth", "rope_theta", "rms_norm_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.rope_parameters["rope_type"] != "default":
            kind = self.rope_parameters["rope_type"]
            if kind not in ("linear", "dynamic"):
                raise ValueError("MiniCPM supports only linear or dynamic rope_scaling")
            factor = self.rope_parameters.get("factor")
            if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 1:
                raise ValueError("rope_scaling.factor must be finite and greater than one")
            if kind == "dynamic" and self.head_dim <= 2:
                raise ValueError("dynamic RoPE requires head_dim > 2")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads
