# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2023-2025 XVERSE and 2026 ModelCloud.ai.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the pinned XVERSE dense, 256K and MoE implementations."""

import json

from transformers import PretrainedConfig
from transformers.utils.hub import cached_file


class XverseConfig(PretrainedConfig):
    model_type = "xverse"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=100278,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        max_tokenizer_truncation=None,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        pretraining_tp=1,
        num_experts=0,
        moe_top_k=6,
        num_shared_experts=None,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        xverse_rope_type=None,
        xverse_rope_persistent=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.max_tokenizer_truncation = max_tokenizer_truncation or max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.num_shared_experts = num_shared_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.pretraining_tp = pretraining_tp
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # The released 13B-256K config predates a serialized RoPE policy field.
        if xverse_rope_type is None:
            xverse_rope_type = (
                "dynamic_256k"
                if (not num_experts and max_position_embeddings == 32768 and max_tokenizer_truncation == 262144)
                else "default"
            )
        if xverse_rope_type not in {"default", "dynamic_256k"}:
            raise ValueError("Unsupported xverse_rope_type")
        if xverse_rope_type == "dynamic_256k" and num_experts:
            raise ValueError("XVERSE 256K is a dense decoder policy")
        self.xverse_rope_type = xverse_rope_type
        self.xverse_rope_persistent = (
            not bool(num_experts) if xverse_rope_persistent is None else xverse_rope_persistent
        )
        # Released config.json files spell RoPE as top-level ``rope_theta``/``rope_scaling``; canonicalize them
        # into ``rope_parameters`` here. Nothing downstream reads the released spelling.
        rope_theta = kwargs.pop("rope_theta", None)
        rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default"}
            if rope_scaling is not None:
                rope_parameters = {
                    "rope_type": rope_scaling.get("rope_type", rope_scaling.get("type")),
                    "factor": rope_scaling.get("factor", 0),
                }
        rope_parameters = dict(rope_parameters)
        rope_parameters.setdefault(
            "rope_theta",
            rope_theta if rope_theta is not None else (500000.0 if xverse_rope_type == "dynamic_256k" else 10000.0),
        )
        self.rope_parameters = rope_parameters
        if (
            num_attention_heads < 1
            or self.num_key_value_heads < 1
            or hidden_size % num_attention_heads
            or num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("XVERSE hidden/head and query/KV head counts must be divisible")
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("XVERSE rotary head dimension must be even")
        if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts < 0:
            raise ValueError("num_experts must be a non-negative integer")
        if num_experts and not 1 <= moe_top_k <= num_experts:
            raise ValueError("moe_top_k must be within the routed expert count")
        if num_shared_experts is not None and (not isinstance(num_shared_experts, int) or num_shared_experts < 1):
            raise ValueError("num_shared_experts must be positive or None")
        if pretraining_tp < 1 or any(
            width % pretraining_tp
            for width in (
                intermediate_size,
                hidden_size,
                vocab_size,
                self.num_key_value_heads * (hidden_size // num_attention_heads),
            )
        ):
            raise ValueError("Invalid pretraining_tp partition")
        if rope_parameters.get("rope_type", "default") != "default":
            if (
                not num_experts
                or rope_parameters["rope_type"] not in {"linear", "dynamic"}
                or rope_parameters.get("factor", 0) <= 1
            ):
                raise ValueError("XVERSE MoE rope_scaling requires linear/dynamic type and factor > 1")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        data, remaining = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        # This private HF4 execution preference is not an architecture field.
        # Explicit caller attn_implementation still reaches the TF5 validator.
        data.pop("_attn_implementation", None)
        if "xverse_rope_persistent" not in data:
            hub_kwargs = {
                k: kwargs[k]
                for k in ("cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder")
                if k in kwargs
            }
            for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
                index = cached_file(
                    pretrained_model_name_or_path,
                    name,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    **hub_kwargs,
                )
                if index is not None:
                    with open(index, encoding="utf-8") as stream:
                        keys = json.load(stream)["weight_map"]
                    data["xverse_rope_persistent"] = any(k.endswith(".self_attn.rotary_emb.inv_freq") for k in keys)
                    break
        return data, remaining
