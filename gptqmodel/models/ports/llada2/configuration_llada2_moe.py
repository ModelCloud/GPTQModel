# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for Transformers 5 by ModelCloud.ai, 2026. See NOTICE.
"""LLaDA2 MoE model configuration"""

from transformers.configuration_utils import PreTrainedConfig


LLADA_VERSIONS = ("2.0", "2.0-mini-preview", "2.0-flash-preview", "2.0-cap", "2.1", "2.2")


class LLaDA2MoeConfig(PreTrainedConfig):
    model_type = "llada2_moe"

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        hidden_act="silu",
        use_qkv_bias=False,  # llada2 only
        use_qk_norm=True,
        use_bias=True,  # llada2 only
        rms_norm_eps=1e-05,
        norm_head=False,  # llada2 only
        tie_word_embeddings=False,  # PreTrainedConfig key, here change default value.
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        rope_scaling=None,
        pad_token_id=126081,
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        llada_version=None,
        expert_capacity=None,
        block_size=None,
        delete_token_id=None,
        split_token_id=None,
        **kwargs,
    ):
        self.llada_version = llada_version
        self.expert_capacity = expert_capacity
        self.block_size = block_size
        self.delete_token_id = delete_token_id
        self.split_token_id = split_token_id
        signature = (expert_capacity, block_size, delete_token_id, split_token_id)
        if any(value is not None for value in signature):
            if not all(value is not None for value in signature):
                raise ValueError("LLaDA2.2 requires expert_capacity, block_size, delete_token_id and split_token_id")
            if llada_version not in (None, "2.2"):
                raise ValueError("Block-routing/edit metadata requires llada_version='2.2'")
            self.llada_version = "2.2"
        elif llada_version == "2.2":
            raise ValueError("LLaDA2.2 requires block-routing and edit-token metadata")
        if self.llada_version is not None and self.llada_version not in LLADA_VERSIONS:
            raise ValueError(
                "Unsupported or ambiguous llada_version; select 2.0, 2.0-mini-preview, "
                "2.0-flash-preview, 2.0-cap, 2.1 or 2.2"
            )
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_parameters = kwargs.pop("rope_parameters", None) or {
            "rope_type": (rope_scaling or {}).get("rope_type", (rope_scaling or {}).get("type", "default")),
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
            **{k: v for k, v in (rope_scaling or {}).items() if k != "type"},
        }
        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.partial_rotary_factor = partial_rotary_factor

        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

    def validate_local_architecture(self):
        if self.llada_version is None:
            raise ValueError(
                "Ambiguous llada2_moe metadata: 2.0 and 2.1 share the same config. "
                "Set config.llada_version (or pass llada_version when loading config) "
                "to '2.0', '2.0-mini-preview', '2.0-flash-preview', '2.0-cap' or '2.1'."
            )
        if self.llada_version not in LLADA_VERSIONS:
            raise ValueError(f"Unsupported llada_version; select one of {LLADA_VERSIONS}")
        signature = (self.expert_capacity, self.block_size, self.delete_token_id, self.split_token_id)
        if self.llada_version == "2.2":
            if any(value is None for value in signature):
                raise ValueError("LLaDA2.2 requires block-routing and edit-token metadata")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in signature):
                raise ValueError("LLaDA2.2 routing and edit-token metadata must be integers")
            if (
                not 0 <= self.delete_token_id < self.vocab_size
                or not 0 <= self.split_token_id < self.vocab_size
                or self.delete_token_id == self.split_token_id
            ):
                raise ValueError("LLaDA2.2 requires distinct in-vocabulary delete/split tokens")
        elif any(value is not None for value in signature):
            raise ValueError("Block-routing/edit metadata requires llada_version='2.2'")
        if getattr(self, "using_split_qkv_in_self_attention", False):
            raise ValueError("Pinned LLaDA2 sources implement packed query_key_value only, not split QKV")
        if self.num_key_value_heads < 1 or self.num_attention_heads < 1 or self.head_dim < 1:
            raise ValueError("Attention head counts and dimensions must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim < 2 or rotary_dim % 2 or rotary_dim > self.head_dim:
            raise ValueError("The partial rotary dimension must be even, positive and <= head_dim")
        if self.num_experts is not None:
            if not 1 <= self.num_experts_per_tok <= self.num_experts:
                raise ValueError("Invalid num_experts_per_tok")
            if self.llada_version == "2.2":
                if not 1 <= self.num_experts_per_tok <= self.expert_capacity <= self.num_experts:
                    raise ValueError("LLaDA2.2 expert_capacity must contain top-k and not exceed num_experts")
                if self.block_size < 1:
                    raise ValueError("LLaDA2.2 block_size must be positive")
            elif (
                self.n_group < 1
                or self.num_experts % self.n_group
                or self.num_experts // self.n_group < 2
                or not 1 <= self.topk_group <= self.n_group
                or self.num_experts_per_tok > self.topk_group * (self.num_experts // self.n_group)
            ):
                raise ValueError("Invalid group-limited routing geometry (each group needs two experts)")
