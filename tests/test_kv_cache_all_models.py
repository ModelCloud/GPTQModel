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


def _tiny_llama_like(model_type, **overrides):
    config_class = AutoConfig.for_model(model_type).__class__
    config = config_class(**overrides)
    return config


@pytest.mark.parametrize(
    "model_type,overrides",
    [
        ("llama", {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
                   "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 128}),
        ("qwen2", {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
                   "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128}),
    ],
)
def test_prediction_matches_the_cache_transformers_allocates(model_type, overrides):
    """Ground truth: run the model and measure ``past_key_values``.

    ``dtype`` must be the weight dtype - a cache is only as small as the
    activations that fill it - so the estimate is asked for that dtype too.
    """

    config = _tiny_llama_like(model_type, **overrides)
    model = AutoModelForCausalLM.from_config(config).eval()
    dtype = next(model.parameters()).dtype
    estimate = estimate_kv_cache(config, dtype=dtype)
    assert estimate is not None

    tokens = 4
    with torch.no_grad():
        output = model(torch.randint(3, 20, (1, tokens)), use_cache=True)

    measured = 0
    for layer in output.past_key_values.layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                measured += tensor.numel() * tensor.element_size()

    assert measured == estimate.kv_bytes_for(tokens)
