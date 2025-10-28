# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Configuration for the MiniMax M2 architecture."""

from __future__ import annotations

from typing import List, Optional, Union

from transformers.configuration_utils import PretrainedConfig


class MiniMaxM2Config(PretrainedConfig):
    model_type = "minimax"

    def __init__(
        self,
        vocab_size: int = 200_064,
        hidden_size: int = 3_072,
        intermediate_size: int = 1_536,
        mlp_intermediate_size: int = 8_192,
        num_hidden_layers: int = 62,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = 128,
        num_local_experts: int = 256,
        num_experts_per_tok: int = 8,
        attn_type_list: Optional[List[int]] = None,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 196_608,
        rope_theta: float = 5_000_000.0,
        rotary_dim: int = 64,
        rope_scaling: Optional[dict] = None,
        use_qk_norm: bool = True,
        qk_norm_type: str = "per_layer",
        use_routing_bias: bool = True,
        scoring_func: str = "sigmoid",
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        output_router_logits: bool = False,
        use_grouped_topk: bool = True,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: float = 1.0,
        layernorm_full_attention_beta: float = 1.0,
        layernorm_linear_attention_beta: float = 1.0,
        layernorm_mlp_beta: float = 1.0,
        shared_intermediate_size: int = 0,
        shared_moe_mode: str = "sigmoid",
        use_mtp: bool = True,
        num_mtp_modules: int = 3,
        mtp_transformer_layers: int = 1,
        attn_window_size: Optional[Union[int, List[int]]] = None,
        swa_rope_theta: float = -1.0,
        sliding_window: Optional[int] = None,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        max_model_len: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        quantization_config = kwargs.pop("quantization_config", None)
        transformers_version = kwargs.pop("transformers_version", None)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_intermediate_size = mlp_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.attn_type_list = attn_type_list or [1] * num_hidden_layers
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.use_routing_bias = use_routing_bias
        self.scoring_func = scoring_func
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.output_router_logits = output_router_logits
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.layernorm_full_attention_beta = layernorm_full_attention_beta
        self.layernorm_linear_attention_beta = layernorm_linear_attention_beta
        self.layernorm_mlp_beta = layernorm_mlp_beta
        self.shared_intermediate_size = shared_intermediate_size
        self.shared_moe_mode = shared_moe_mode
        self.use_mtp = use_mtp
        self.num_mtp_modules = num_mtp_modules
        self.mtp_transformer_layers = mtp_transformer_layers
        self.attn_window_size = attn_window_size
        self.swa_rope_theta = swa_rope_theta
        self.sliding_window = sliding_window
        self.initializer_range = initializer_range
        self.max_model_len = max_model_len
        self.use_cache = use_cache

        # Convenient accessor used by rotary embedding helper
        self.partial_rotary_factor = float(self.rotary_dim) / float(self.head_dim)
        if quantization_config is not None:
            self.quantization_config = quantization_config
        self.transformers_version = transformers_version


__all__ = ["MiniMaxM2Config"]
