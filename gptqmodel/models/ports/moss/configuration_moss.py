# Copyright (c) OpenMOSS and the HuggingFace teams.
# SPDX-License-Identifier: Apache-2.0
"""MOSS Moon-003 configuration, adapted for Transformers 5."""

from transformers import PreTrainedConfig


class MossConfig(PreTrainedConfig):
    model_type = "moss"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions", "hidden_size": "n_embd",
        "num_attention_heads": "n_head", "num_hidden_layers": "n_layer",
    }

    def __init__(self, vocab_size=107008, n_positions=2048, n_ctx=2048,
                 n_embd=4096, n_layer=28, n_head=16, rotary_dim=64, n_inner=None,
                 activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0,
                 attn_pdrop=0.0, layer_norm_epsilon=1e-5, initializer_range=0.02,
                 use_cache=True, bos_token_id=106028, eos_token_id=106068,
                 tie_word_embeddings=False, **kwargs):
        # Accept the standard HF spellings through the source's attribute map.
        n_positions = kwargs.pop("max_position_embeddings", n_positions)
        n_embd = kwargs.pop("hidden_size", n_embd)
        n_layer = kwargs.pop("num_hidden_layers", n_layer)
        n_head = kwargs.pop("num_attention_heads", n_head)
        if n_head <= 0 or n_head % 4 or n_embd <= 0 or n_embd % n_head:
            raise ValueError("MOSS requires n_head divisible by four and n_embd divisible by n_head")
        head_dim = n_embd // n_head
        if rotary_dim is None:
            # The source's full-rotary branch incorrectly sizes the table to
            # n_embd. Its intended full-head operation uses head_dim instead.
            rotary_dim = head_dim
        if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
            raise ValueError("MOSS rotary_dim must be positive, even and no greater than head_dim")
        if n_layer <= 0 or n_positions <= 0 or vocab_size <= 0 or (n_inner is not None and n_inner <= 0):
            raise ValueError("MOSS layer, position, vocabulary and inner dimensions must be positive")
        if kwargs.pop("num_key_value_heads", n_head) != n_head:
            raise ValueError("MOSS Moon-003 uses MHA, not GQA or MQA")
        if kwargs.get("wbits", 32) != 32:
            raise ValueError("Upstream MOSS wbits checkpoints require their original converter; use GPT-QModel quantization_config")
        self.vocab_size, self.n_positions, self.n_ctx = vocab_size, n_positions, n_ctx
        self.n_embd, self.n_layer, self.n_head = n_embd, n_layer, n_head
        self.num_key_value_heads, self.head_dim = n_head, head_dim
        self.rotary_dim, self.n_inner = rotary_dim, n_inner
        self.activation_function = activation_function
        self.resid_pdrop, self.embd_pdrop, self.attn_pdrop = resid_pdrop, embd_pdrop, attn_pdrop
        self.layer_norm_epsilon, self.initializer_range = layer_norm_epsilon, initializer_range
        self.use_cache = use_cache
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)
