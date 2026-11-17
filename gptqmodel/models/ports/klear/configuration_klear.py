# SPDX-FileCopyrightText: 2025 Kwai-Klear Team, Kuaishou
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Klear configuration adapted from the pinned sources in README.md."""

from copy import deepcopy

from transformers.configuration_utils import PreTrainedConfig


class KlearConfig(PreTrainedConfig):
    model_type = "Klear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=151936, hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=24, num_attention_heads=32, num_key_value_heads=4,
        hidden_act="silu", max_position_embeddings=32768, initializer_range=0.02,
        rms_norm_eps=1e-6, use_cache=True, tie_word_embeddings=False,
        rope_theta=10000.0, rope_scaling=None, rope_parameters=None,
        attention_bias=False, use_sliding_window=False, sliding_window=4096,
        attention_dropout=0.0, decoder_sparse_step=1, moe_intermediate_size=768,
        num_experts_per_tok=8, num_experts=128, norm_topk_prob=True,
        output_router_logits=False, router_aux_loss_coef=0.001, mlp_only_layers=None,
        routed_scaling_factor=2.5, n_shared_experts=1, head_dim=None,
        pad_token_id=None, bos_token_id=None, eos_token_id=None, **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.layer_types = [
            "sliding_attention" if self.sliding_window is not None else "full_attention"
        ] * num_hidden_layers
        self.rope_parameters = deepcopy(rope_parameters if rope_parameters is not None else rope_scaling) or {}
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.standardize_rope_params()
        self.validate_rope()
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = list(mlp_only_layers or [])
        # Kept for checkpoint compatibility; upstream does not apply this factor.
        self.routed_scaling_factor = routed_scaling_factor
        self.n_shared_experts = n_shared_experts
        if num_attention_heads % num_key_value_heads or self.head_dim % 2:
            raise ValueError("Klear requires divisible Q/KV heads and an even head_dim")
        if decoder_sparse_step < 1 or num_experts < 0:
            raise ValueError("Klear requires a positive decoder_sparse_step and nonnegative num_experts")
        if num_experts and not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")


__all__ = ["KlearConfig"]
