# SPDX-FileCopyrightText: 2022 EleutherAI and the HuggingFace Inc. team
# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the pinned NVIDIA tri-mode diffusion decoder (see NOTICE)."""

from copy import deepcopy
from typing import Any

from transformers import PreTrainedConfig


class NemotronLabsDiffusionConfig(PreTrainedConfig):
    model_type = "nemotron_labs_diffusion"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"llama_4_scaling_beta"}

    def __init__(
        self, vocab_size=131072, hidden_size=4096, intermediate_size=14336,
        num_hidden_layers=34, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, hidden_act="silu", max_position_embeddings=262144,
        initializer_range=0.02, rms_norm_eps=1e-5, use_cache=False,
        pad_token_id=None, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
        rope_theta=1000000.0, rope_parameters=None, attention_bias=False,
        attention_dropout=0.0, mlp_bias=False, sliding_window=None,
        mask_token_id=-1, dlm_paradigm="bidirectional", block_size=32,
        dlm_loss_weight=None, ar_loss_weight=1.0, dp_varying_mask_ratio=False,
        **kwargs,
    ):
        kwargs.pop("attn_class", None)
        legacy_rope = kwargs.pop("rope_scaling", None)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters: dict[str, Any] = deepcopy(rope_parameters or legacy_rope or {"rope_type": "default"})
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault("llama_4_scaling_beta", 0.0)
        self.rope_parameters.setdefault("original_max_position_embeddings", max_position_embeddings)
        if self.rope_parameters.get("rope_type", self.rope_parameters.get("type")) == "default":
            self.ignore_keys_at_rope_validation = self.ignore_keys_at_rope_validation | {"original_max_position_embeddings"}
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window
        self.mask_token_id = mask_token_id
        self.dlm_paradigm = dlm_paradigm
        self.block_size = block_size
        self.dlm_loss_weight = dlm_loss_weight
        self.ar_loss_weight = ar_loss_weight
        self.dp_varying_mask_ratio = dp_varying_mask_ratio
        if dlm_paradigm not in {"bidirectional", "autoregressive", "block_diff"}:
            raise ValueError(f"Unsupported DLM paradigm: {dlm_paradigm}")
        if num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Attention requires divisible Q/KV heads and an even head_dim")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        kwargs.update(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                      eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings)
        super().__init__(**kwargs)
