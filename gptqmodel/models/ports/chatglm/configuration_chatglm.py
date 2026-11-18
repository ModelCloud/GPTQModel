# SPDX-FileCopyrightText: 2023 THUDM
# SPDX-FileCopyrightText: 2024 Zhipu AI
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Adapted from THUDM ChatGLM sources.
from transformers import PretrainedConfig


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "max_position_embeddings": "seq_length",
        "vocab_size": "padded_vocab_size",
    }
    def __init__(
        self,
        num_layers=None,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        original_rope=True,
        rope_ratio=None,
        **kwargs
    ):
        if kwargs.pop("position_encoding_2d", False) or "inner_hidden_size" in kwargs:
            raise ValueError(
                "Original ChatGLM1 transformer.layers / 2D-position checkpoints are unsupported; "
                "this local port owns the ChatGLM2/3 and GLM4 transformer.encoder layouts."
            )
        # Legacy vocab_size can count only unpadded tokens. Explicit padded
        # rows remain authoritative when reading checkpoint metadata; the
        # alias above keeps subsequent HF resize mutations synchronized.
        kwargs.pop("vocab_size", None)
        hidden_layers = kwargs.pop("num_hidden_layers", None)
        if num_layers is None:
            num_layers = hidden_layers if hidden_layers is not None else 28
        elif hidden_layers is not None and num_layers != hidden_layers:
            raise ValueError("num_layers and num_hidden_layers describe conflicting ChatGLM stacks")
        # Legacy max_length described the context window, not a generation limit.
        kwargs.pop("max_length", None)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("use_cache", True)
        self.original_rope = original_rope
        self.rope_ratio = rope_ratio
        if kv_channels is None:
            kv_channels = hidden_size // num_attention_heads
        if num_attention_heads < 1 or kv_channels < 4 or kv_channels % 4:
            raise ValueError("ChatGLM attention requires positive heads and kv_channels divisible by four")
        if multi_query_attention and (
            multi_query_group_num < 1 or num_attention_heads % multi_query_group_num
        ):
            raise ValueError("multi_query_group_num must divide num_attention_heads")
        if quantization_bit:
            raise ValueError("Upstream ChatGLM CUDA quantization_bit checkpoints are unsupported; use GPT-QModel quantization")
        if rope_ratio is not None and rope_ratio <= 0:
            raise ValueError("rope_ratio must be positive")
        self.num_layers = num_layers
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.use_cache = kwargs.pop("use_cache", True)
        super().__init__(**kwargs)
    @property
    def num_key_value_heads(self):
        return self.multi_query_group_num if self.multi_query_attention else self.num_attention_heads

    @property
    def head_dim(self):
        return self.kv_channels
