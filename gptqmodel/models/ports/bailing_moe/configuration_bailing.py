# Copyright 2023, 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
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
"""Ling 2.0 / 2.6 configuration, adapted from the sources pinned in README.md."""

from transformers.configuration_utils import PretrainedConfig


def is_native_hybrid(values):
    return (
        values.get("model_type") == "bailing_hybrid"
        and values.get("q_lora_rank") is None
        and not {"BailingMoeV2_5ForCausalLM", "BailingForCausalLM"}.intersection(values.get("architectures") or ())
    )


class BailingConfig(PretrainedConfig):
    model_type = "bailing_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=157184, hidden_size=2048, intermediate_size=5120,
        num_hidden_layers=20, num_attention_heads=16, num_key_value_heads=4,
        hidden_act="silu", use_qkv_bias=False, use_bias=False, rms_norm_eps=1e-6,
        tie_word_embeddings=False, embedding_dropout=0.0, attention_dropout=0.0,
        output_dropout=0.0, initializer_range=0.02, max_position_embeddings=32768,
        rope_theta=600000.0, use_cache=True, pad_token_id=156892,
        eos_token_id=156892, num_experts=256, num_shared_experts=1,
        num_experts_per_tok=8, n_group=8, topk_group=4, moe_intermediate_size=512,
        first_k_dense_replace=1, head_dim=None, output_router_logits=False,
        use_qk_norm=None, num_nextn_predict_layers=0, mtp_loss_scaling_factor=0.0,
        moe_router_enable_expert_bias=True, routed_scaling_factor=1.0,
        layer_group_size=5, group_norm_size=4, linear_silu=False,
        kv_lora_rank=512, q_lora_rank=None, qk_rope_head_dim=64, v_head_dim=128,
        qk_nope_head_dim=128, rope_interleave=True, partial_rotary_factor=None,
        model_type="bailing_moe", legacy_gqa=None, norm_head=False, norm_topk_prob=True, **kwargs,
    ):
        values = locals().copy()
        values.pop("self")
        values.pop("kwargs")
        values.pop("__class__", None)
        # Released checkpoints may carry a `rope_scaling` field; it stays in kwargs so
        # `PreTrainedConfig` converts it into the standardized `rope_parameters` at parse time.
        self.rope_parameters = kwargs.pop("rope_parameters", None)
        for key, value in values.items():
            setattr(self, key, value)
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.legacy_gqa = legacy_gqa if legacy_gqa is not None else (
            "BailingMoeForCausalLM" in (kwargs.get("architectures") or ())
        )
        self.use_qk_norm = not self.legacy_gqa if use_qk_norm is None else use_qk_norm
        self.partial_rotary_factor = (1.0 if self.legacy_gqa else 0.5) if partial_rotary_factor is None else partial_rotary_factor
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        super().__init__(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings, **kwargs,
        )
        self.rope_theta = self.rope_parameters["rope_theta"]
        if model_type not in ("bailing_moe", "bailing_hybrid"):
            raise ValueError(f"Unsupported Bailing model_type: {model_type}")
        if min(num_attention_heads, num_key_value_heads) < 1 or num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.legacy_gqa and num_experts is not None and not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("Invalid legacy expert routing configuration")
        if not self.legacy_gqa and num_experts is not None and (
            n_group < 1 or num_experts % n_group or num_experts // n_group < 2
            or not 1 <= topk_group <= n_group
            or not 1 <= num_experts_per_tok <= topk_group * (num_experts // n_group)
        ):
            raise ValueError("Invalid group-limited expert routing configuration")
        if model_type == "bailing_hybrid" and layer_group_size < 1:
            raise ValueError("layer_group_size must be positive")

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if is_native_hybrid(config_dict):
            from transformers.models.bailing_hybrid.configuration_bailing_hybrid import BailingHybridConfig

            return BailingHybridConfig.from_dict(config_dict, **kwargs)
        config_dict = dict(config_dict)
        for native, legacy in (
            ("num_experts_per_token", "num_experts_per_tok"), ("num_expert_group", "n_group"),
            ("model_max_length", "max_position_embeddings"), ("moe_renormalize", "norm_topk_prob"),
        ):
            if native in config_dict and legacy not in config_dict:
                config_dict[legacy] = config_dict[native]
        return super().from_dict(config_dict, **kwargs)

    def to_dict(self):
        result = super().to_dict()
        result["model_type"] = self.model_type
        return result
