# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the released PanGu-α checkpoint layout."""

from transformers import PreTrainedConfig


class GPTPanguConfig(PreTrainedConfig):
    model_type = "gpt_pangu"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_hidden_layers": "num_layers", "num_attention_heads": "num_heads"}

    def __init__(
        self,
        vocab_size=40000,
        max_position_embeddings=1024,
        hidden_size=2560,
        intermediate_size=None,
        num_layers=32,
        num_heads=32,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=9,
        eos_token_id=9,
        tie_word_embeddings=True,
        **kwargs,
    ):
        # The released names remain canonical; standard HF aliases serialize safely.
        num_layers = kwargs.pop("num_hidden_layers", num_layers)
        num_heads = kwargs.pop("num_attention_heads", num_heads)
        if min(vocab_size, max_position_embeddings, hidden_size, num_layers, num_heads) <= 0:
            raise ValueError("GPT-Pangu sizes, layer count and head count must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if intermediate_size is not None and intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive or None")
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scale_attn_weights = scale_attn_weights
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        kwargs.update(bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings)
        super().__init__(**kwargs)
