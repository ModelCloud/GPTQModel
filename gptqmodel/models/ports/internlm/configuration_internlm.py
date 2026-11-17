# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""InternLM 1 configuration adapted from the pinned upstream source in README.md."""

from transformers import PreTrainedConfig


class InternLMConfig(PreTrainedConfig):
    model_type = "internlm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=103168, hidden_size=4096, intermediate_size=11008,
                 num_hidden_layers=32, num_attention_heads=32, hidden_act="silu",
                 max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=1e-6,
                 use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                 tie_word_embeddings=False, bias=True, rotary=None,
                 attn_implementation=None, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.bias = bias
        self.rotary = dict(rotary) if rotary is not None else {"base": 10000, "type": "dynamic"}
        if hidden_size % num_attention_heads:
            raise ValueError("InternLM hidden_size must be divisible by num_attention_heads")
        self.head_dim = hidden_size // num_attention_heads
        if self.head_dim % 2 or self.head_dim < 4:
            raise ValueError("InternLM requires an even head dimension of at least four")
        if self.rotary.get("type") not in {"origin", "dynamic"}:
            raise ValueError("InternLM rotary type must be origin or dynamic")
        if self.rotary.get("base", 0) <= 0 or self.rotary.get("scaling_factor", 1.0) <= 0:
            raise ValueError("InternLM rotary base and scaling_factor must be positive")
        if max_position_embeddings <= 0:
            raise ValueError("InternLM max_position_embeddings must be positive")
        # Legacy checkpoints may serialize their eager implementation selector.
        kwargs.pop("num_key_value_heads", None)
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings,
                         attn_implementation=attn_implementation, **kwargs)
