# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/instella/SPEC.md; see NOTICE.
"""Configuration for the Instella decoder-only causal LM."""

from transformers import PreTrainedConfig


_ROPE_ERROR = "Instella rope_scaling requires type linear/dynamic and factor > 1"


def _rope_parameters(scaling, rope_theta):
    """Normalize a legacy ``rope_scaling`` / TF5 ``rope_parameters`` dict."""
    if scaling is None:
        return {"rope_type": "default", "rope_theta": rope_theta}
    if not isinstance(scaling, dict):
        raise ValueError(_ROPE_ERROR)
    kind = scaling.get("rope_type", scaling.get("type"))
    if kind == "default":
        return {"rope_type": "default", "rope_theta": rope_theta}
    factor = scaling.get("factor")
    valid_factor = isinstance(factor, (int, float)) and not isinstance(factor, bool) and factor > 1
    if kind not in ("linear", "dynamic") or not valid_factor:
        raise ValueError(_ROPE_ERROR)
    return {"rope_type": kind, "rope_theta": rope_theta, "factor": factor}


class InstellaConfig(PreTrainedConfig):
    model_type = "instella"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        **kwargs,
    ):
        rope_parameters = kwargs.pop("rope_parameters", None)
        rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_parameters is not None:
            scaling = rope_parameters
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
        else:
            scaling = rope_scaling

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = _rope_parameters(scaling, rope_theta)

        if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 2:
            raise ValueError(
                "Instella requires an even head dimension and hidden_size divisible by num_attention_heads"
            )
        if self.num_key_value_heads <= 0 or num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if self.rope_parameters["rope_type"] == "dynamic" and hidden_size // num_attention_heads <= 2:
            raise ValueError("Dynamic NTK scaling requires head_dim > 2")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["InstellaConfig"]
