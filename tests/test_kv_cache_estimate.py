# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""KV-cache estimates must describe the config, not a plausible number.

Weight quantization does not change the cache, so the question after a run is
"what does 8k of context cost me". These tests pin the arithmetic, the shapes it
refuses to guess at (MLA keeps a compressed latent, not per-head K/V), and the
text-config unwrapping for multimodal models.
"""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.kv_cache import KVCacheEstimate, estimate_kv_cache


def _llama(**overrides):
    base = {
        "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_grouped_query_attention_matches_the_arithmetic():
    # 32 layers * 2 (K and V) * 8 heads * 128 head_dim * 2 bytes
    estimate = estimate_kv_cache(_llama(), dtype=torch.float16)
    assert estimate.bytes_per_token == 32 * 2 * 8 * 128 * 2
    assert estimate.bytes_for(8192) == estimate.bytes_per_token * 8192
    assert estimate.bytes_for(8192, batch_size=4) == estimate.bytes_per_token * 8192 * 4


def test_multi_query_attention_uses_one_kv_head():
    estimate = estimate_kv_cache(_llama(num_key_value_heads=1), dtype=torch.bfloat16)
    assert estimate.kv_heads == 1
    assert estimate.bytes_per_token == 32 * 2 * 1 * 128 * 2


def test_head_dim_defaults_to_hidden_over_heads():
    estimate = estimate_kv_cache(_llama(hidden_size=2048, num_attention_heads=16), dtype=torch.float16)
    assert estimate.head_dim == 128


def test_explicit_head_dim_wins():
    estimate = estimate_kv_cache(_llama(head_dim=64), dtype=torch.float16)
    assert estimate.head_dim == 64


def test_fp32_doubles_the_footprint():
    fp16 = estimate_kv_cache(_llama(), dtype=torch.float16)
    fp32 = estimate_kv_cache(_llama(), dtype=torch.float32)
    assert fp32.bytes_per_token == 2 * fp16.bytes_per_token


@pytest.mark.parametrize("field", ("kv_lora_rank", "q_lora_rank"))
def test_mla_configs_are_refused(field):
    """A compressed latent is not one K/V pair per head per layer."""
    config = _llama()
    setattr(config, field, 512)
    assert estimate_kv_cache(config) is None


def test_multimodal_text_config_is_unwrapped():
    text = _llama()
    config = SimpleNamespace(text_config=text, vision_config=SimpleNamespace(hidden_size=1024))
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.bytes_per_token == 32 * 2 * 8 * 128 * 2


def test_missing_shapes_return_none():
    assert estimate_kv_cache(None) is None
    assert estimate_kv_cache(SimpleNamespace(hidden_size=4096)) is None
    assert estimate_kv_cache(SimpleNamespace(num_hidden_layers=4, num_attention_heads=8)) is None


def test_unsupported_dtype_returns_none():
    assert estimate_kv_cache(_llama(), dtype=torch.int8) is None


def test_config_dtype_is_used_when_none_is_passed():
    config = _llama(torch_dtype=torch.float32)
    assert estimate_kv_cache(config).dtype == torch.float32


def test_config_without_dtype_falls_back_to_float16():
    assert estimate_kv_cache(_llama()).dtype == torch.float16


def test_sliding_window_is_reported_and_kept():
    config = _llama()
    config.sliding_window = 4096
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.sliding_window == 4096
    assert "sliding window 4k caps growth" in estimate.format()


def test_zero_and_negative_inputs_are_clamped():
    estimate = KVCacheEstimate(layers=4, kv_heads=2, head_dim=64, dtype=torch.float16)
    assert estimate.bytes_for(0) == 0
    assert estimate.bytes_for(-8) == 0
    assert estimate.tokens_for(0) == 0


def test_tokens_for_is_the_inverse_of_bytes_for():
    estimate = KVCacheEstimate(layers=4, kv_heads=2, head_dim=64, dtype=torch.float16)
    budget = estimate.bytes_for(1000)
    assert estimate.tokens_for(budget) == 1000
    assert estimate.tokens_for(budget, batch_size=2) == 500


def test_format_names_the_shapes_and_contexts():
    estimate = KVCacheEstimate(layers=32, kv_heads=8, head_dim=128, dtype=torch.float16)
    line = estimate.format()
    assert "KV cache:" in line
    assert "32L x 8KV x 128d" in line
    assert "float16" in line
    assert "8k ctx" in line and "32k ctx" in line


def test_format_scales_units():
    small = KVCacheEstimate(layers=2, kv_heads=1, head_dim=8, dtype=torch.float16)
    assert "KiB" in small.format() or "B" in small.format()
    large = KVCacheEstimate(layers=80, kv_heads=8, head_dim=128, dtype=torch.float16)
    assert "GiB" in large.format()
