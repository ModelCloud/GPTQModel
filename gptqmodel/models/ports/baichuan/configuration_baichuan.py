# Copyright 2023 Baichuan Inc. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0; see the repository LICENSE.
"""Baichuan 1/2 config, adapted to Transformers 5."""

from transformers import PretrainedConfig


class BaichuanConfig(PretrainedConfig):
    model_type = "baichuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=125696, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, hidden_act="silu",
        max_position_embeddings=4096, model_max_length=None,
        initializer_range=0.02, rms_norm_eps=1e-6, use_cache=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
        z_loss_weight=0.0, position_embedding_type=None, normalize_lm_head=None,
        **kwargs,
    ):
        # Released configs lack these two discriminators. 7B/13B select
        # RoPE/ALiBi; the expanded Baichuan 2 vocabulary selects NormHead.
        self.position_embedding_type = position_embedding_type or (
            "alibi" if hidden_size == 5120 else "rope"
        )
        self.normalize_lm_head = vocab_size == 125696 if normalize_lm_head is None else normalize_lm_head
        if self.position_embedding_type not in ("rope", "alibi"):
            raise ValueError("Baichuan position_embedding_type must be rope or alibi")
        if hidden_size % num_attention_heads:
            raise ValueError("Baichuan hidden_size must be divisible by num_attention_heads")
        if self.position_embedding_type == "rope" and hidden_size // num_attention_heads % 2:
            raise ValueError("Baichuan RoPE requires an even head dimension")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.model_max_length = model_max_length or max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.z_loss_weight = z_loss_weight
        # Cache fields from a serialized config are derived, never GQA knobs.
        heads = kwargs.pop("num_key_value_heads", num_attention_heads)
        if heads != num_attention_heads:
            raise ValueError("Baichuan fused W_pack checkpoints use full multi-head attention")
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


# Original Baichuan-7B used the capitalized spelling in its auto_map.
BaiChuanConfig = BaichuanConfig
