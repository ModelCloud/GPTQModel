# Adapted from NVIDIA Hymba, revision bc2d2b43bc73d62ad8bfc9e6a3dcc26ab0795e77.
# NVIDIA Open Model License Agreement, June 2024. See LICENSE.pdf and NOTICE.
"""Data-only configuration for the checkpoint-compatible Hymba hybrid decoder."""

import math

from transformers import PreTrainedConfig


class HymbaConfig(PreTrainedConfig):
    model_type = "hymba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=65536, hidden_size=4096, intermediate_size=14336,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
        hidden_act="silu", mlp_hidden_act="silu", initializer_range=0.02,
        rms_norm_eps=1e-6, use_cache=True, calc_logits_for_entire_prompt=False,
        output_router_logits=False, router_aux_loss_coef=0.001, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, tie_word_embeddings=False, sliding_window=None,
        max_position_embeddings=262144, orig_max_position_embeddings=None,
        attention_dropout=0.0, num_experts_per_tok=2, num_experts=16,
        use_mamba_kernels=True, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_dt_rank="auto", mamba_conv_bias=True, mamba_proj_bias=False,
        mamba_inner_layernorms=True, kv_reuse_every_i_layer=-1, kv_reuse_group=None,
        kv_weight_reuse=False, global_attn_idx=None, num_mamba=1,
        attn_implementation_new="sdpa", rope_type=None, rope=False, rope_theta=10000.0,
        kq_norm="none", kq_head_dim=-1, v_head_dim=-1, attn_hidden_size=-1,
        num_memory_tokens=0, memory_tokens_interspersed_every=0, **kwargs,
    ):
        values = locals().copy()
        values.pop("self")
        values.pop("kwargs")
        values.pop("__class__", None)
        for name, value in values.items():
            setattr(self, name, value)
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.mamba_dt_rank = math.ceil(hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.kv_reuse_group = None if kv_reuse_group is None else [list(group) for group in kv_reuse_group]
        self.global_attn_idx = list(global_attn_idx or [])
        # Legacy JSON keys are accepted as data. Dimensions are determined by
        # the model, never patched by mutating a caller's dictionary at runtime.
        conv_dim = kwargs.pop("conv_dim", None)
        self.conv_dim = {str(i): int(mamba_expand * hidden_size) for i in range(num_hidden_layers)}
        if conv_dim is not None and any(
            str(i) not in self.conv_dim or dim != self.conv_dim[str(i)] for i, dim in conv_dim.items()
        ):
            raise ValueError("Hymba conv_dim must match mamba_expand * hidden_size")
        layer_type = kwargs.pop("layer_type", ["h"] * num_hidden_layers)
        if layer_type != ["h"] * num_hidden_layers or num_mamba != 1:
            raise ValueError("The NVIDIA Hymba implementation defines one hybrid head per layer")
        self.layer_type = list(layer_type)
        if min(hidden_size, num_hidden_layers, num_attention_heads, self.num_key_value_heads,
               mamba_d_state, mamba_d_conv, self.mamba_dt_rank, num_experts) <= 0:
            raise ValueError("Hymba dimensions must be positive")
        if hidden_size % num_attention_heads or num_attention_heads % self.num_key_value_heads:
            raise ValueError("Hymba requires divisible attention and KV heads")
        inner = int(mamba_expand * hidden_size)
        if inner <= 0 or inner % num_attention_heads:
            raise ValueError("Hymba expanded width must be divisible by attention heads")
        if attn_hidden_size not in (-1, hidden_size) or kq_head_dim not in (-1, hidden_size // num_attention_heads):
            raise ValueError("Hymba fused in_proj requires the checkpoint's hidden-size Q/K layout")
        if v_head_dim not in (-1, inner // num_attention_heads):
            raise ValueError("Hymba value width must match the expanded SSM width")
        if kq_norm not in (None, "none", "rms", "perhead-rms"):
            raise ValueError("Unsupported Hymba kq_norm")
        if mlp_hidden_act not in ("silu", "relu2"):
            raise ValueError("Hymba MLP supports silu and relu2")
        if not 1 <= num_experts_per_tok <= num_experts and num_experts > 1:
            raise ValueError("Invalid Hymba expert top-k")
        if num_memory_tokens < 0 or memory_tokens_interspersed_every < 0:
            raise ValueError("Hymba memory-token counts must be nonnegative")
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("Hymba sliding_window must be positive")
        if rope and (hidden_size // num_attention_heads) % 2:
            raise ValueError("Hymba rotary head dimension must be even")
        if rope_type not in (None, "default", "ntk", "dynamic_ntk"):
            raise ValueError("Unsupported Hymba rope_type")
        if rope and rope_type in ("ntk", "dynamic_ntk"):
            if not orig_max_position_embeddings or hidden_size // num_attention_heads <= 2:
                raise ValueError("NTK RoPE needs original context length and head dimension > 2")
        self.kv_sources  # Validate before allocating a model or a cache.
        # Original checkpoints advertise a Transformers-4/Flex execution hint.
        # This package implements the same masks with portable Torch SDPA; do
        # not let the serialized hint select an unavailable remote/kernel path.
        kwargs.pop("attn_implementation", None)
        kwargs["attn_implementation"] = "sdpa"
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def kv_sources(self):
        sources = list(range(self.num_hidden_layers))
        if self.kv_reuse_group is not None:
            assigned = set()
            for group in self.kv_reuse_group:
                if len(group) < 2 or len(set(group)) != len(group):
                    raise ValueError("Hymba KV groups need one producer and distinct consumers")
                producer, *consumers = group
                if any(not isinstance(i, int) or not 0 <= i < self.num_hidden_layers for i in group):
                    raise ValueError("Hymba KV group layer index is out of range")
                if assigned.intersection(group) or any(i <= producer for i in consumers):
                    raise ValueError("Hymba KV groups must be disjoint and follow their producer")
                assigned.update(group)
                for consumer in consumers:
                    sources[consumer] = producer
        elif self.kv_reuse_every_i_layer > 0:
            sources = [i - i % self.kv_reuse_every_i_layer for i in sources]
        return sources
