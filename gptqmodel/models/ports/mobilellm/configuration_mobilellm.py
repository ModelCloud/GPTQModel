# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Modifications Copyright (c) 2026 ModelCloud.ai.
# Licensed under the FAIR Noncommercial Research License; see LICENSE and NOTICE.

from copy import deepcopy

from transformers import PreTrainedConfig


class MobileLLMConfig(PreTrainedConfig):
    model_type = "mobilellm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=32000, hidden_size=576, intermediate_size=1536,
        num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
        head_dim=None, hidden_act="silu", max_position_embeddings=2048,
        initializer_range=0.02, rms_norm_eps=1e-5, use_cache=True,
        pad_token_id=None, bos_token_id=1, eos_token_id=2,
        pretraining_tp=1, tie_word_embeddings=False, rope_theta=10000.0,
        rope_scaling=None, rope_parameters=None, attention_bias=False,
        attention_dropout=0.0, mlp_bias=False, share_embedding=True,
        layer_sharing=False, **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        dimensions = (vocab_size, hidden_size, intermediate_size, num_hidden_layers,
                      num_attention_heads, self.num_key_value_heads, max_position_embeddings)
        if any(not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in dimensions):
            raise ValueError("MobileLLM model dimensions must be positive integers")
        self.head_dim = hidden_size // num_attention_heads if head_dim is None else head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.share_embedding = share_embedding
        self.layer_sharing = layer_sharing
        if not 0 <= attention_dropout <= 1:
            raise ValueError("MobileLLM attention_dropout must be between zero and one")
        params = deepcopy(rope_parameters if rope_parameters is not None else rope_scaling) or {}
        self.rope_theta = params.get("rope_theta", rope_theta)
        if self.rope_theta <= 0:
            raise ValueError("MobileLLM rope_theta must be positive")
        rope_type = params.get("rope_type", params.get("type", "default"))
        if rope_type not in {"default", "linear", "dynamic"}:
            raise ValueError(f"Unsupported MobileLLM RoPE type: {rope_type}")
        factor = params.get("factor", 1.0)
        if not isinstance(factor, (int, float)) or factor < 1:
            raise ValueError("MobileLLM RoPE factor must be at least one")
        self.rope_parameters = {"rope_type": rope_type, "rope_theta": self.rope_theta}
        if rope_type != "default":
            self.rope_parameters["factor"] = factor
        if (not isinstance(self.head_dim, int) or self.head_dim < 1
                or self.head_dim % 2 or self.head_dim * num_attention_heads != hidden_size
                or num_attention_heads % self.num_key_value_heads):
            raise ValueError("MobileLLM requires even head_dim and divisible attention/KV head geometry")
        if rope_type == "dynamic" and self.head_dim <= 2:
            raise ValueError("Dynamic NTK RoPE requires head_dim > 2")
        if (not isinstance(pretraining_tp, int) or pretraining_tp < 1
                or any(width % pretraining_tp for width in
                       (hidden_size, intermediate_size, self.num_key_value_heads * self.head_dim, vocab_size))):
            raise ValueError("MobileLLM pretraining_tp must divide projection widths and vocabulary")
        kwargs.update(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                      eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings)
        super().__init__(**kwargs)
