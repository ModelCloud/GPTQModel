# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""TeleChat configuration, adapted to Transformers 5."""

from transformers import PreTrainedConfig


class Telechat2Config(PreTrainedConfig):
    model_type = "telechat"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_hidden_layers": "n_layer", "num_attention_heads": "n_head"}

    def __init__(
        self, vocab_size=160256, hidden_size=4096, n_layer=30, n_head=32,
        ffn_hidden_size=12288, num_key_value_heads=None, layer_norm_epsilon=1e-5,
        initializer_range=0.02, use_cache=True, bos_token_id=1, eos_token_id=2,
        pad_token_id=3, tie_word_embeddings=False,
        apply_residual_connection_post_layernorm=False, hidden_dropout=0.0,
        attention_dropout=0.0, training_seqlen=8192, base_seqlen=None,
        rope_theta=None, embed_layernorm=False, telechat_rope_policy=None, telechat_rope_precision=None, **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = kwargs.pop("n_embed", hidden_size)
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_key_value_heads = num_key_value_heads or n_head
        self.ffn_hidden_size = ffn_hidden_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.training_seqlen = training_seqlen
        self.seq_length = kwargs.pop("seq_length", training_seqlen)
        self.base_seqlen = base_seqlen if base_seqlen is not None else 4096
        self.rope_theta = rope_theta if rope_theta is not None else 10000.0
        self.telechat_rope_policy = telechat_rope_policy or ("modern" if rope_theta is not None else "legacy")
        self.telechat_rope_precision = telechat_rope_precision or (
            str(kwargs.get("dtype", kwargs.get("torch_dtype", "float16"))).removeprefix("torch.")
            if self.telechat_rope_policy == "modern" else "float16"
        )
        self.embed_layernorm = embed_layernorm
        if self.telechat_rope_policy not in {"legacy", "modern"}:
            raise ValueError("telechat_rope_policy must be legacy or modern")
        if self.hidden_size % n_head or self.hidden_size // n_head <= 2 or (self.hidden_size // n_head) % 2:
            raise ValueError("TeleChat requires an even head dimension greater than two")
        if n_head % self.num_key_value_heads or min(n_layer, training_seqlen, self.base_seqlen) < 1:
            raise ValueError("Invalid TeleChat layer, KV-head or sequence geometry")
        kwargs.update(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                      tie_word_embeddings=tie_word_embeddings)
        super().__init__(**kwargs)
