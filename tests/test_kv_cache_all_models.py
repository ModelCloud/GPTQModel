# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Every supported model type must resolve, and the arithmetic must be exact.

The estimator picks a cache layout per layer, so its correctness depends on how
each architecture declares that layout. Two things are asserted here:

1. the registry sweep: for all supported model types whose config class is
   installed, `estimate_kv_cache` either returns an estimate or a *documented*
   None (every layer recurrent), and never raises. Model types whose config class
   is not installed are skipped with the reason, so the coverage number in the
   summary is honest.
2. the arithmetic: for a model that can be built, the predicted bytes equal the
   bytes transformers actually allocates for the cache.
"""

import warnings

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.models.auto import SUPPORTED_MODELS
from gptqmodel.utils.kv_cache import estimate_kv_cache, kv_cache_unavailable_reason


warnings.filterwarnings("ignore")

_RECURRENT_TYPES = {
    "linear_attention", "gated_deltanet", "deltanet", "gla", "recurrent",
    "mamba", "mamba2", "ssm", "lightning_attention",
}
_ATTENTION_TYPES = {
    "full_attention", "attention", "self_attention", "indexed_attention",
    "deepseek_sparse_attention", "sparse_attention", "mla", "multi_latent_attention",
    "sliding_attention", "sliding_window_attention", "window_attention",
}


def _layer_types(config):
    """Declared layer types, unwrapping multimodal wrappers."""

    for field in ("layer_types", "attention_layer_types", "attn_layer_types"):
        try:
            declared = getattr(config, field, None)
        except Exception:
            declared = None
        if isinstance(declared, (list, tuple)) and declared:
            return [str(entry).lower() for entry in declared]
    for field in ("text_config", "thinker_config", "talker_config"):
        nested = getattr(config, field, None)
        if nested is not None:
            types = _layer_types(nested)
            if types:
                return types
    return None


def _has_layers(config) -> bool:
    for field in ("num_hidden_layers", "text_config", "thinker_config", "talker_config"):
        value = getattr(config, field, None)
        if field == "num_hidden_layers":
            if value:
                return True
        elif value is not None and _has_layers(value):
            return True
    return False


def _only_recurrent_layers(config) -> bool:
    declared = _layer_types(config)
    return bool(declared) and all(entry in _RECURRENT_TYPES for entry in declared)


@pytest.mark.parametrize("model_type", sorted(SUPPORTED_MODELS))
def test_supported_model_resolves_without_raising(model_type):
    try:
        config = AutoConfig.for_model(model_type)
    except Exception as exc:
        pytest.skip(f"{model_type}: config class not installed ({type(exc).__name__})")

    estimate = estimate_kv_cache(config)  # must not raise: that is the assertion

    if estimate is None:
        # Allowed only when the config does not expose a decoder stack at all
        # (an OCR wrapper, say) or every declared layer is recurrent.
        reason = kv_cache_unavailable_reason(config)
        assert reason != "unknown", (
            f"{model_type}: no estimate and no reason for a config with attention "
            f"layers (layer_types={_layer_types(config)})"
        )
        if _layer_types(config) and not _only_recurrent_layers(config):
            # A declared attention stack that still has no estimate means the
            # shapes are missing, not that the model has no cache.
            assert reason in ("attention head shapes not declared", "layer count not declared",
                              "unresolved layer layout"), f"{model_type}: {reason}"
        return

    assert estimate.growing_layers > 0, model_type
    assert estimate.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    assert any(entry in _ATTENTION_TYPES for entry in (estimate.unclassified or ())) is False
    # Growth is monotonic in the token count.
    assert estimate.kv_bytes_for(2048) >= estimate.kv_bytes_for(1024) >= 0


def _tiny_config(model_type, **overrides):
    config_class = AutoConfig.for_model(model_type).__class__
    config = config_class(**overrides)
    return config


# Every config is small enough to build and forward on CPU in a few seconds.
# The recurrent stacks are asked for bfloat16 so the conv state follows the
# activations while the SSM state stays FP32, the split the estimate must model.
_TINY_CASES = [
    ("llama", {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
               "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 128}, None),
    ("qwen2", {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
               "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128}, None),
    ("jamba", {"num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128,
               "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128,
               "attn_layer_period": 2, "attn_layer_offset": 1, "num_experts": 2,
               "num_experts_per_tok": 1, "expert_layer_period": 2, "expert_layer_offset": 1,
               "mamba_d_state": 8, "mamba_d_conv": 4, "mamba_expand": 2,
               "mamba_dt_rank": 4}, torch.bfloat16),
    ("falcon_h1", {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
                   "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128,
                   "mamba_d_state": 8, "mamba_d_conv": 4, "mamba_expand": 2,
                   "mamba_dt_rank": 4, "mamba_n_groups": 1}, torch.bfloat16),
    ("bamba", {"num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128,
               "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128,
               "attn_layer_indices": [0], "mamba_d_state": 8, "mamba_d_conv": 4,
               "mamba_expand": 2, "mamba_dt_rank": 4}, torch.bfloat16),
    ("qwen3_next", {"num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128,
                    "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 4, "num_experts": 2,
                    "num_experts_per_tok": 1, "moe_intermediate_size": 64,
                    "shared_expert_intermediate_size": 64}, torch.bfloat16),
]
_TINY_IDS = [case[0] for case in _TINY_CASES]
_RECURRENT_CASES = [case for case in _TINY_CASES if case[2] is not None]


def _cache_bytes(cache):
    """Every byte ``past_key_values`` retains, recurrent state included."""

    total = 0
    for layer in cache.layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        for name in ("conv_states", "recurrent_states"):
            states = getattr(layer, name, None)
            if isinstance(states, dict):
                for tensor in states.values():
                    if tensor is not None:
                        total += tensor.numel() * tensor.element_size()
    return total


@pytest.mark.parametrize("model_type,overrides,dtype", _TINY_CASES, ids=_TINY_IDS)
def test_prediction_matches_the_cache_transformers_allocates(model_type, overrides, dtype):
    """Ground truth: run the model and measure ``past_key_values``.

    ``dtype`` must be the weight dtype - a cache is only as small as the
    activations that fill it - so the estimate is asked for that dtype too.
    Hybrid and recurrent stacks (Jamba, Falcon-H1, Bamba, Qwen3-Next) keep a
    constant conv/SSM state next to the growing K/V; it is counted here too,
    and measured at two lengths so a per-token miscount cannot cancel out.
    """

    config = _tiny_config(model_type, **overrides)
    model = AutoModelForCausalLM.from_config(config).eval()
    if dtype is not None:
        model = model.to(dtype)
    dtype = next(model.parameters()).dtype
    estimate = estimate_kv_cache(config, dtype=dtype)
    assert estimate is not None
    assert estimate.is_exact is True

    for tokens in (4, 32):
        with torch.no_grad():
            output = model(torch.randint(3, 20, (1, tokens)), use_cache=True)
        measured = _cache_bytes(output.past_key_values)
        assert measured == estimate.kv_bytes_for(tokens), (
            model_type, tokens, measured, estimate.kv_bytes_for(tokens))


@pytest.mark.parametrize("model_type,overrides,dtype", _RECURRENT_CASES, ids=[case[0] for case in _RECURRENT_CASES])
def test_recurrent_state_is_counted_once_per_sequence(model_type, overrides, dtype):
    """The state is a per-sequence constant; it must not ride the token slope."""

    config = _tiny_config(model_type, **overrides)
    estimate = estimate_kv_cache(config, dtype=dtype)
    assert estimate is not None
    assert estimate.state_layers > 0
    assert estimate.state_bytes_per_sequence > 0
    # No live sequence, no state.
    assert estimate.kv_bytes_for(0) == 0
    one_token = estimate.kv_bytes_for(1)
    assert one_token == estimate.state_bytes_per_sequence + estimate.kv_bytes_per_token
    assert estimate.kv_bytes_for(64) - one_token == estimate.kv_bytes_per_token * 63
    assert "constant state" in estimate.describe()


def test_parallel_hybrid_reports_its_recurrent_half():
    """Falcon-H1 runs attention and Mamba in parallel in every layer.

    There is no ``linear_attention`` layer, so the state has to be attached to
    the attention layer itself or it is invisible.
    """

    overrides = dict(_TINY_CASES[3][1])
    estimate = estimate_kv_cache(_tiny_config("falcon_h1", **overrides), dtype=torch.bfloat16)
    assert estimate is not None
    assert estimate.linear_layers == 0
    assert estimate.state_layers == overrides["num_hidden_layers"]
    assert "hybrid layer(s) hold" in estimate.describe()


def test_unresolved_recurrent_state_is_partial_not_exact():
    """A stack that declares no state shape must not report an exact total."""

    config = _tiny_config("llama", num_hidden_layers=2, hidden_size=64,
                          intermediate_size=128, num_attention_heads=2,
                          num_key_value_heads=2, vocab_size=128)
    config.layer_types = ["linear_attention", "full_attention"]
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate is not None
    assert estimate.is_exact is False
    assert any("state layout not declared" in note for note in estimate.partial)
    assert "partial:" in estimate.describe()
    # The attention half is still counted; the unknown state is not guessed.
    assert estimate.kv_bytes_for(16) == 16 * estimate.kv_bytes_per_token
