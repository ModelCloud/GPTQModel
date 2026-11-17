# Copyright 2024 Databricks, Inc. All rights reserved.
# Modifications Copyright 2026 ModelCloud.ai.
# Licensed under the Databricks Open Model License; see LICENSE.txt and NOTICE.txt.
"""Configuration for ModelCloud's split-QKV, per-expert converted DBRX."""

from copy import deepcopy

from transformers import PreTrainedConfig
from transformers.configuration_utils import SubConfigSpec


class DbrxAttentionConfig(PreTrainedConfig):
    base_config_key = "attn_config"

    def __init__(self, attn_pdrop=0.0, clip_qkv=None, kv_n_heads=1, rope_theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta


class DbrxFFNConfig(PreTrainedConfig):
    base_config_key = "ffn_config"

    def __init__(
        self, ffn_act_fn=None, ffn_hidden_size=3584, moe_num_experts=4, moe_top_k=1,
        moe_jitter_eps=None, moe_loss_weight=0.01, moe_normalize_expert_weights=1,
        uniform_expert_assignment=False, **kwargs,
    ):
        super().__init__(**kwargs)
        self.ffn_act_fn = deepcopy(ffn_act_fn) if ffn_act_fn is not None else {"name": "silu"}
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment


class DbrxConfig(PreTrainedConfig):
    model_type = "dbrx_converted"
    sub_configs_defaults = {
        "attn_config": SubConfigSpec(config_class=DbrxAttentionConfig),
        "ffn_config": SubConfigSpec(config_class=DbrxFFNConfig),
    }
    attribute_map = {
        "num_attention_heads": "n_heads", "hidden_size": "d_model",
        "num_hidden_layers": "n_layers", "max_position_embeddings": "max_seq_len",
    }

    def __init__(
        self, d_model=2048, n_heads=16, n_layers=24, max_seq_len=2048, vocab_size=32000,
        resid_pdrop=0.0, emb_pdrop=0.0, attn_config=None, ffn_config=None, use_cache=True,
        initializer_range=0.02, output_router_logits=False, router_aux_loss_coef=0.05, **kwargs,
    ):
        if kwargs.get("tie_word_embeddings", False):
            raise ValueError("tie_word_embeddings is not supported for converted DBRX.")
        kwargs.setdefault("tie_word_embeddings", False)
        for key in ("pad_token_id", "bos_token_id", "eos_token_id"):
            kwargs.setdefault(key, None)
        self.attn_config = (
            attn_config if isinstance(attn_config, DbrxAttentionConfig)
            else DbrxAttentionConfig(**(attn_config or {}))
        )
        self.ffn_config = (
            ffn_config if isinstance(ffn_config, DbrxFFNConfig)
            else DbrxFFNConfig(**(ffn_config or {}))
        )
        super().__init__(**kwargs)
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.max_seq_len, self.vocab_size = max_seq_len, vocab_size
        self.resid_pdrop, self.emb_pdrop = resid_pdrop, emb_pdrop
        self.use_cache, self.initializer_range = use_cache, initializer_range
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.num_key_value_heads = self.attn_config.kv_n_heads
        if min(d_model, n_heads, n_layers, vocab_size, max_seq_len, self.num_key_value_heads) <= 0:
            raise ValueError("Converted DBRX dimensions must be positive.")
        if d_model % n_heads or n_heads % self.num_key_value_heads or (d_model // n_heads) % 2:
            raise ValueError("Converted DBRX requires divisible GQA heads and an even rotary head dimension.")
        if not 1 <= self.ffn_config.moe_top_k <= self.ffn_config.moe_num_experts:
            raise ValueError("moe_top_k must be between 1 and moe_num_experts.")
        if self.ffn_config.ffn_hidden_size <= 0:
            raise ValueError("ffn_hidden_size must be positive.")
