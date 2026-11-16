# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""KV-cache estimates must describe the config, not a plausible number.

Weight quantization does not change the cache, so the question after a run is
"what does 8k of context cost me". Recent model families no longer share one
layout: gqa keeps K and V per head, MLA keeps one compressed latent plus a
decoupled RoPE key shared across heads, linear-attention layers keep a
constant-size state, and sliding layers stop growing at their window. These
tests pin each layout, the mixed stacks that combine them, and the cases the
estimator refuses rather than guessing.
"""

from types import SimpleNamespace

import torch

from gptqmodel.utils.kv_cache import KVCacheEstimate, estimate_kv_cache


def _gqa(**overrides):
    base = {
        "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _mla(**overrides):
    base = {
        "hidden_size": 128, "num_hidden_layers": 4, "num_attention_heads": 8,
        "kv_lora_rank": 512, "qk_rope_head_dim": 64,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_grouped_query_attention_matches_the_arithmetic():
    estimate = estimate_kv_cache(_gqa(), dtype=torch.float16)
    # 32 layers * 2 (K and V) * 8 heads * 128 head_dim * 2 bytes
    assert estimate.layer_type == "gqa"
    assert estimate.growing_layers == 32
    assert estimate.dims_per_layer_token == 2 * 8 * 128
    assert estimate.bytes_per_token == 32 * 2 * 8 * 128 * 2
    assert estimate.bytes_for(8192) == estimate.bytes_per_token * 8192
    assert estimate.bytes_for(8192, batch_size=4) == estimate.bytes_per_token * 8192 * 4


def test_multi_query_attention_uses_one_kv_head():
    estimate = estimate_kv_cache(_gqa(num_key_value_heads=1), dtype=torch.bfloat16)
    assert estimate.dims_per_layer_token == 2 * 1 * 128


def test_head_dim_defaults_to_hidden_over_heads():
    estimate = estimate_kv_cache(_gqa(hidden_size=2048, num_attention_heads=16), dtype=torch.float16)
    assert estimate.dims_per_layer_token == 2 * 8 * 128


def test_explicit_head_dim_wins():
    estimate = estimate_kv_cache(_gqa(head_dim=64), dtype=torch.float16)
    assert estimate.dims_per_layer_token == 2 * 8 * 64


def test_fp32_doubles_the_footprint():
    fp16 = estimate_kv_cache(_gqa(), dtype=torch.float16)
    fp32 = estimate_kv_cache(_gqa(), dtype=torch.float32)
    assert fp32.bytes_per_token == 2 * fp16.bytes_per_token


def test_mla_caches_the_latent_and_rope_key_not_per_head_kv():
    """512 latent + 64 rope, shared across heads, and no factor of two."""
    estimate = estimate_kv_cache(_mla(), dtype=torch.float16)
    assert estimate.layer_type == "mla"
    assert estimate.dims_per_layer_token == 512 + 64
    assert estimate.bytes_per_token == 4 * (512 + 64) * 2
    # A per-head reading of the same config would be far larger.
    naive = 4 * 2 * 8 * 128 * 2
    assert estimate.bytes_per_token < naive


def test_mla_without_a_rope_key_still_counts_the_latent():
    estimate = estimate_kv_cache(_mla(qk_rope_head_dim=0), dtype=torch.float16)
    assert estimate.dims_per_layer_token == 512


def test_config_without_latent_rank_uses_the_gqa_path():
    estimate = estimate_kv_cache(_gqa(), dtype=torch.float16)
    assert estimate.layer_type == "gqa"


def test_linear_attention_layers_are_reported_but_not_counted_per_token():
    config = _mla(num_hidden_layers=3, layer_types=["linear_attention", "indexed_attention", "indexed_attention"])
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.growing_layers == 2
    assert estimate.linear_layers == 1
    assert estimate.bytes_per_token == 2 * (512 + 64) * 2
    assert "linear-attention layer(s) hold constant state" in estimate.format()


def test_mtp_layers_reuse_the_last_declared_layout():
    config = _gqa(num_hidden_layers=2, num_nextn_predict_layers=1,
                  layer_types=["full_attention", "full_attention"])
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.growing_layers == 3
    assert estimate.mtp_layers == 1
    assert "incl. 1 MTP layer(s)" in estimate.format()


def test_unknown_layer_types_are_reported_and_marked_inexact():
    config = _gqa(num_hidden_layers=2, layer_types=["full_attention", "quantum_attention"])
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.exact is False
    assert estimate.unclassified == ("quantum_attention",)
    assert "unknown layer type(s): quantum_attention" in estimate.format()


def test_sliding_layers_stop_growing_at_their_window():
    config = _gqa(num_hidden_layers=2, sliding_window=4096,
                  layer_types=["full_attention", "sliding_attention"])
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    per_layer = 2 * 8 * 128 * 2
    assert estimate.sliding_layers == 1
    assert estimate.full_layers == 1
    assert estimate.bytes_for(1024) == 2 * 1024 * per_layer
    assert estimate.bytes_for(4096) == 2 * 4096 * per_layer
    # Past the window only the full layer keeps growing.
    assert estimate.bytes_for(8192) == (8192 + 4096) * per_layer
    assert "capped at 4k" in estimate.format()


def test_tokens_for_inverts_bytes_for_without_a_window():
    estimate = KVCacheEstimate(layer_type="gqa", growing_layers=4,
                               dims_per_layer_token=2 * 2 * 64, dtype=torch.float16)
    budget = estimate.bytes_for(1000)
    assert estimate.tokens_for(budget) == 1000
    assert estimate.tokens_for(budget, batch_size=2) == 500


def test_tokens_for_handles_the_window_boundary():
    estimate = KVCacheEstimate(layer_type="gqa", growing_layers=2,
                               dims_per_layer_token=1024, dtype=torch.float16,
                               sliding_layers=1, sliding_window=4096)
    per_layer = 1024 * 2
    assert estimate.tokens_for(2 * 1024 * per_layer) == 1024
    assert estimate.tokens_for((4096 + 4096) * per_layer) == 4096
    assert estimate.tokens_for((4096 + 8192) * per_layer) == 8192


def test_all_recurrent_layers_have_no_per_token_cache():
    config = _mla(num_hidden_layers=2, layer_types=["linear_attention", "linear_attention"])
    assert estimate_kv_cache(config, dtype=torch.float16) is None


def test_multimodal_text_config_is_unwrapped():
    text = _gqa()
    config = SimpleNamespace(text_config=text, vision_config=SimpleNamespace(hidden_size=1024))
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.growing_layers == 32


def test_missing_shapes_return_none():
    assert estimate_kv_cache(None) is None
    assert estimate_kv_cache(SimpleNamespace(hidden_size=4096)) is None
    assert estimate_kv_cache(SimpleNamespace(num_hidden_layers=4, num_attention_heads=8)) is None


def test_unsupported_dtype_returns_none():
    assert estimate_kv_cache(_gqa(), dtype=torch.int8) is None


def test_config_dtype_is_used_when_none_is_passed():
    assert estimate_kv_cache(_gqa(torch_dtype=torch.float32)).dtype == torch.float32


def test_config_without_dtype_falls_back_to_float16():
    assert estimate_kv_cache(_gqa()).dtype == torch.float16


def test_zero_and_negative_inputs_are_clamped():
    estimate = KVCacheEstimate(layer_type="gqa", growing_layers=4,
                               dims_per_layer_token=2 * 2 * 64, dtype=torch.float16)
    assert estimate.bytes_for(0) == 0
    assert estimate.bytes_for(-8) == 0
    assert estimate.tokens_for(0) == 0


def test_format_names_the_layout_and_contexts():
    estimate = estimate_kv_cache(_mla(), dtype=torch.float16)
    line = estimate.format()
    assert "KV cache:" in line
    assert "mla" in line
    assert "float16" in line
    assert "8k ctx" in line and "32k ctx" in line


def test_format_scales_units():
    small = KVCacheEstimate(layer_type="gqa", growing_layers=2,
                            dims_per_layer_token=2 * 1 * 8, dtype=torch.float16)
    assert any(unit in small.format() for unit in ("B", "KiB"))
    large = KVCacheEstimate(layer_type="gqa", growing_layers=80,
                            dims_per_layer_token=2 * 8 * 128, dtype=torch.float16)
    assert "GiB" in large.format()
