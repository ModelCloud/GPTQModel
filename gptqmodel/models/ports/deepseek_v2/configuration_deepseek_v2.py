# SPDX-FileCopyrightText: 2025 HuggingFace Inc. team
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Preserve the legacy DeepSeek V2 architecture switches (DeepSeek-Coder-V2-Lite-Instruct@e434a23f)."""

import math

from huggingface_hub.dataclasses import strict
from transformers import DeepseekV2Config as NativeDeepseekV2Config


@strict
class DeepseekV2Config(NativeDeepseekV2Config):
    n_routed_experts: int | None = 64
    n_shared_experts: int | None = 2
    num_experts_per_tok: int | None = 6
    moe_layer_freq: int = 1
    scoring_func: str = "softmax"
    ep_size: int = 1

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.moe_layer_freq < 1:
            raise ValueError("DeepSeek V2 moe_layer_freq must be a positive integer")
        if self.scoring_func != "softmax":
            raise ValueError("DeepSeek V2 supports only scoring_func='softmax'")
        if self.ep_size != 1:
            raise ValueError("DeepSeek V2 checkpoint expert-parallel ep_size must be 1; load all experts")
        if self.topk_method not in ("greedy", "group_limited_greedy"):
            raise ValueError("DeepSeek V2 topk_method must be greedy or group_limited_greedy")
        if type(self.first_k_dense_replace) is not int or self.first_k_dense_replace < 0:
            raise ValueError("DeepSeek V2 first_k_dense_replace must be a nonnegative integer")
        if self.n_routed_experts is not None:
            if type(self.n_routed_experts) is not int or self.n_routed_experts < 1:
                raise ValueError("DeepSeek V2 n_routed_experts must be positive or None for dense models")
            if type(self.num_experts_per_tok) is not int or not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
                raise ValueError("DeepSeek V2 num_experts_per_tok must be within the routed expert count")
            if self.topk_method == "group_limited_greedy":
                if (
                    type(self.n_group) is not int or self.n_group < 1
                    or self.n_routed_experts % self.n_group
                    or type(self.topk_group) is not int or not 1 <= self.topk_group <= self.n_group
                    or self.num_experts_per_tok > self.topk_group * (self.n_routed_experts // self.n_group)
                ):
                    raise ValueError("DeepSeek V2 group-limited routing requires divisible, sufficient expert groups")
        if self.n_shared_experts is not None and (
            type(self.n_shared_experts) is not int or self.n_shared_experts < 1
        ):
            raise ValueError("DeepSeek V2 n_shared_experts must be positive or None")
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("DeepSeek V2 routed_scaling_factor must be finite and positive")
