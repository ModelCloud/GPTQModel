# SPDX-FileCopyrightText: 2024 NVIDIA Corporation
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the two checkpoint-compatible DeciLM families."""

import copy

from transformers import PreTrainedConfig


class DeciLMConfig(PreTrainedConfig):
    model_type = "deci"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None,
        num_key_value_heads_per_layer=None, block_configs=None, hidden_act="silu",
        max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=1e-6,
        use_cache=True, pad_token_id=None, bos_token_id=1, eos_token_id=2,
        tie_word_embeddings=False, rope_theta=10000.0, rope_scaling=None,
        rope_parameters=None, attention_bias=False, mlp_bias=False,
        attention_dropout=0.0, pretraining_tp=1, model_type="deci", **kwargs,
    ):
        if model_type not in ("deci", "nemotron-nas"):
            raise ValueError(f"Unsupported DeciLM model_type: {model_type}")
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (num_key_value_heads if model_type == "nemotron-nas"
                                    else num_key_value_heads or num_attention_heads)
        self.num_key_value_heads_per_layer = copy.deepcopy(num_key_value_heads_per_layer)
        if num_key_value_heads_per_layer is None and model_type == "deci":
            self.num_key_value_heads_per_layer = [self.num_key_value_heads] * num_hidden_layers
        self.block_configs = copy.deepcopy(block_configs)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.pretraining_tp = pretraining_tp
        if not isinstance(pretraining_tp, int) or pretraining_tp < 1:
            raise ValueError("pretraining_tp must be a positive integer")
        self.rope_theta = rope_theta
        self.rope_scaling = copy.deepcopy(rope_scaling)
        params = copy.deepcopy(rope_parameters or rope_scaling or {})
        params["rope_type"] = params.pop("type", params.get("rope_type", "default"))
        params.setdefault("rope_theta", rope_theta)
        self.rope_parameters = params
        if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 2:
            raise ValueError("DeciLM requires integral, even attention head dimensions")
        self.head_dim = hidden_size // num_attention_heads
        # NAS uses no global FFN width or KV count. Never hand either sentinel
        # to a generic Llama/DynamicCache constructor.
        if self.block_configs is not None:
            if len(self.block_configs) != num_hidden_layers:
                raise ValueError("block_configs must have one entry per decoder layer")
            for block in self.block_configs:
                for name in ("attention", "ffn"):
                    part = block[name]
                    if part.get("no_op", False) and part.get("replace_with_linear", False):
                        raise ValueError("A subblock cannot be both no_op and linear")
                attn = block["attention"]
                if not attn.get("no_op") and not attn.get("replace_with_linear"):
                    group = attn.get("n_heads_in_group")
                    if not isinstance(group, int) or group <= 0 or num_attention_heads % group:
                        raise ValueError("n_heads_in_group must divide num_attention_heads")
                    window, sinks = attn.get("window_length"), attn.get("num_sink_tokens")
                    if window is not None and (window < 0 or (window == 0 and not attn.get("unshifted_sink"))):
                        raise ValueError("Invalid attention window")
                    if sinks is not None and (window is None or sinks < 0):
                        raise ValueError("Sink attention requires a window and nonnegative sink count")
                    if sinks is not None and not attn.get("unshifted_sink") and not 0 < sinks < window:
                        raise ValueError("Shifted sinks require 0 < num_sink_tokens < window_length")
                    if attn.get("unshifted_sink") and attn.get("use_prefill_window_in_sink_attention"):
                        raise ValueError("Unshifted sinks cannot use the prefill window flag")
                ffn = block["ffn"]
                if not ffn.get("no_op") and not ffn.get("replace_with_linear") and (ffn.get("ffn_mult") or 0) <= 0:
                    raise ValueError("Active NAS FFN requires positive ffn_mult")
        elif model_type == "nemotron-nas":
            # Permit the default configuration used by HF's diff serializer,
            # but construction rejects an unspecified NAS architecture.
            pass
        else:
            heads = self.num_key_value_heads_per_layer
            if heads is None or len(heads) != num_hidden_layers:
                raise ValueError("deci requires num_key_value_heads_per_layer for every layer")
            if any(not isinstance(h, int) or h <= 0 or num_attention_heads % h for h in heads):
                raise ValueError("Every KV head count must divide num_attention_heads")
            if intermediate_size is None or intermediate_size <= 0:
                raise ValueError("deci requires a positive intermediate_size")
        super().__init__(
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings, **kwargs,
        )

    def to_diff_dict(self):
        # HF's default diff implementation instantiates the config with no
        # args; that would invent heterogeneous layer metadata.
        return self.to_dict()

    def to_dict(self):
        payload = super().to_dict()
        # PreTrainedConfig writes the class attribute. This package-owned
        # config deliberately represents both serialized architecture aliases.
        payload["model_type"] = self.model_type
        return payload

    def layer_config(self, index):
        if self.block_configs is not None:
            return self.block_configs[index]
        if self.model_type == "nemotron-nas":
            raise ValueError("nemotron-nas requires checkpoint block_configs")
        return {
            "attention": {"n_heads_in_group": self.num_attention_heads // self.num_key_value_heads_per_layer[index]},
            "ffn": {"intermediate_size": self.intermediate_size},
        }
