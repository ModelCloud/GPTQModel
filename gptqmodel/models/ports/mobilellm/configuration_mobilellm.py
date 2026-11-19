# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/mobilellm/SPEC.md; see NOTICE.
"""MobileLLM configuration."""

import copy

from transformers.configuration_utils import PreTrainedConfig


_ROPE_TYPES = ("default", "linear", "dynamic")


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"MobileLLM `{name}` must be a positive integer, got {value!r}")


class MobileLLMConfig(PreTrainedConfig):
    model_type = "mobilellm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        share_embedding=True,
        layer_sharing=False,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        for name, value in (
            ("vocab_size", vocab_size),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("max_position_embeddings", max_position_embeddings),
        ):
            _positive_int(name, value)
        if not 0 <= attention_dropout <= 1:
            raise ValueError(f"MobileLLM `attention_dropout` must be in [0, 1], got {attention_dropout!r}")

        source = rope_parameters if rope_parameters is not None else rope_scaling
        source = copy.deepcopy(source) if source is not None else {}
        rope_theta = source.get("rope_theta", rope_theta)
        rope_type = source.get("rope_type", source.get("type", "default"))
        factor = source.get("factor", 1.0)
        if not isinstance(rope_theta, (int, float)) or isinstance(rope_theta, bool) or not rope_theta > 0:
            raise ValueError(f"MobileLLM `rope_theta` must be positive, got {rope_theta!r}")
        if rope_type not in _ROPE_TYPES:
            raise ValueError(f"MobileLLM supports RoPE types {_ROPE_TYPES}, got {rope_type!r}")
        if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor < 1:
            raise ValueError(f"MobileLLM RoPE `factor` must be a number >= 1, got {factor!r}")

        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1 or head_dim % 2:
            raise ValueError(f"MobileLLM `head_dim` must be a positive even integer, got {head_dim!r}")
        if head_dim * num_attention_heads != hidden_size:
            raise ValueError("MobileLLM requires head_dim * num_attention_heads == hidden_size")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("MobileLLM requires num_attention_heads divisible by num_key_value_heads")
        if rope_type == "dynamic" and head_dim <= 2:
            raise ValueError("MobileLLM dynamic RoPE requires head_dim > 2")

        if not isinstance(pretraining_tp, int) or isinstance(pretraining_tp, bool) or pretraining_tp < 1:
            raise ValueError(f"MobileLLM `pretraining_tp` must be a positive integer, got {pretraining_tp!r}")
        for name, value in (
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("num_key_value_heads * head_dim", num_key_value_heads * head_dim),
            ("vocab_size", vocab_size),
        ):
            if value % pretraining_tp:
                raise ValueError(f"MobileLLM `pretraining_tp`={pretraining_tp} must divide {name}={value}")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
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
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        parameters = {"rope_type": rope_type, "rope_theta": rope_theta}
        if rope_type != "default":
            parameters["factor"] = factor
        self.rope_parameters = parameters
        self.rope_theta = rope_theta


__all__ = ["MobileLLMConfig"]
