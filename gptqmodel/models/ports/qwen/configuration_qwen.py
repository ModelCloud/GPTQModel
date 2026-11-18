# Copyright (c) Alibaba Cloud.
# Modified by ModelCloud.ai for Transformers 5; see NOTICE and LICENSE.
"""Original Qwen checkpoint configuration (Qwen/Qwen-7B@ef3c5c9c)."""

from transformers import PreTrainedConfig


class QWenConfig(PreTrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        emb_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        max_position_embeddings=8192,
        scale_attn_weights=True,
        use_cache=True,
        bf16=False,
        fp16=False,
        fp32=False,
        kv_channels=128,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        use_flash_attn="auto",
        intermediate_size=22016,
        no_bias=True,
        tie_word_embeddings=False,
        use_cache_quantization=False,
        use_cache_kernel=False,
        softmax_in_fp32=False,
        seq_length=2048,
        **kwargs,
    ):
        values = locals().copy()
        for key in ("self", "kwargs", "__class__", "tie_word_embeddings"):
            values.pop(key, None)
        for key, value in values.items():
            setattr(self, key, value)
        if hidden_size != num_attention_heads * kv_channels:
            raise ValueError("Original Qwen requires hidden_size = heads * kv_channels")
        if kwargs.get("num_key_value_heads", num_attention_heads) != num_attention_heads:
            raise ValueError("Original Qwen uses fused equal-width Q/K/V (MHA)")
        rotary_dim = int(kv_channels * rotary_pct)
        if not 0 <= rotary_pct <= 1 or rotary_dim == 2 or rotary_dim % 2:
            raise ValueError("Original Qwen rotary dimension must be zero or even and greater than 2")
        if intermediate_size % 2 or seq_length <= 1 or num_hidden_layers < 1:
            raise ValueError("Invalid original Qwen MLP, context or layer geometry")
        if use_cache_kernel:
            raise ValueError("Legacy Qwen CUDA cache kernels are not ported; use_cache_kernel=False")
        if use_flash_attn is True:
            raise ValueError("Qwen FlashAttention is unqualified; use auto/False and eager or SDPA")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
