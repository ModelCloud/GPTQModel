# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""KV-cache estimates must describe the config, not a plausible number.

Weight quantization does not change the cache, so the question after a run is
"what does 8k of context cost me". Recent model families no longer share one
layout: gqa keeps K and V per head, MLA keeps one compressed latent plus a
decoupled RoPE key shared across heads, linear-attention layers keep a
constant-size state, and sliding layers stop growing at their window. These
tests pin each layout, the per-layer arithmetic that mixes them, and the cases
the estimator refuses rather than guessing.
"""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.kv_cache import KVCacheEstimate, LayerCache, estimate_kv_cache


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


def _dims(estimate):
    return [layer.dims for layer in estimate.layers]


def test_grouped_query_attention_matches_the_arithmetic():
    estimate = estimate_kv_cache(_gqa(), dtype=torch.float16)
    # 32 layers * 2 (K and V) * 8 heads * 128 head_dim * 2 bytes
    assert estimate.layer_type == "gqa"
    assert estimate.growing_layers == 32
    assert set(_dims(estimate)) == {2 * 8 * 128}
    assert estimate.bytes_per_token == 32 * 2 * 8 * 128 * 2
    assert estimate.bytes_for(8192) == estimate.bytes_per_token * 8192
    assert estimate.bytes_for(8192, batch_size=4) == estimate.bytes_per_token * 8192 * 4


def test_multi_query_attention_uses_one_kv_head():
    estimate = estimate_kv_cache(_gqa(num_key_value_heads=1), dtype=torch.bfloat16)
    assert set(_dims(estimate)) == {2 * 1 * 128}


def test_head_dim_defaults_to_hidden_over_heads():
    estimate = estimate_kv_cache(_gqa(hidden_size=2048, num_attention_heads=16), dtype=torch.float16)
    assert set(_dims(estimate)) == {2 * 8 * 128}


def test_explicit_head_dim_wins():
    estimate = estimate_kv_cache(_gqa(head_dim=64), dtype=torch.float16)
    assert set(_dims(estimate)) == {2 * 8 * 64}


def test_fp32_doubles_the_footprint():
    fp16 = estimate_kv_cache(_gqa(), dtype=torch.float16)
    fp32 = estimate_kv_cache(_gqa(), dtype=torch.float32)
    assert fp32.bytes_per_token == 2 * fp16.bytes_per_token


def test_mla_caches_the_latent_and_rope_key_not_per_head_kv():
    """512 latent + 64 rope, shared across heads, and no factor of two."""
    estimate = estimate_kv_cache(_mla(), dtype=torch.float16)
    assert estimate.layer_type == "mla"
    assert set(_dims(estimate)) == {512 + 64}
    assert estimate.bytes_per_token == 4 * (512 + 64) * 2
    # A per-head reading of the same config would be far larger.
    naive = 4 * 2 * 8 * 128 * 2
    assert estimate.bytes_per_token < naive


def test_mla_without_a_rope_key_still_counts_the_latent():
    estimate = estimate_kv_cache(_mla(qk_rope_head_dim=0), dtype=torch.float16)
    assert set(_dims(estimate)) == {512}


def test_per_layer_head_dims_are_summed_not_assumed_uniform():
    """Heterogeneous stacks declare different head_dim per layer."""
    config = _gqa(num_hidden_layers=2)
    config.per_layer_config = [
        SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=64),
        SimpleNamespace(num_attention_heads=8, num_key_value_heads=4, head_dim=128),
    ]
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert _dims(estimate) == [2 * 2 * 64, 2 * 4 * 128]


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
    assert estimate.window == 4096
    assert estimate.bytes_for(1024) == 2 * 1024 * per_layer
    assert estimate.bytes_for(4096) == 2 * 4096 * per_layer
    # Past the window only the full layer keeps growing: that is the slope.
    assert estimate.bytes_for(8192) == (8192 + 4096) * per_layer
    assert estimate.bytes_per_token == per_layer  # only the full layer grows forever
    assert "capped at 4k" in estimate.format()


def test_tokens_for_inverts_bytes_for_without_a_window():
    estimate = KVCacheEstimate(
        layers=(LayerCache(kind="gqa", dims=2 * 2 * 64),) * 4,
        dtype=torch.float16,
    )
    budget = estimate.bytes_for(1000)
    assert estimate.tokens_for(budget) == 1000
    assert estimate.tokens_for(budget, batch_size=2) == 500


def test_tokens_for_handles_the_window_boundary():
    estimate = KVCacheEstimate(
        layers=(LayerCache(kind="gqa", dims=1024),
                LayerCache(kind="gqa", dims=1024, window=4096)),
        dtype=torch.float16,
    )
    per_layer = 1024 * 2
    assert estimate.tokens_for(2 * 1024 * per_layer) == 1024
    assert estimate.tokens_for((4096 + 4096) * per_layer) == 4096
    assert estimate.tokens_for((4096 + 8192) * per_layer) == 8192


def test_all_recurrent_layers_have_no_per_token_cache():
    config = _mla(num_hidden_layers=2, layer_types=["linear_attention", "linear_attention"])
    assert estimate_kv_cache(config, dtype=torch.float16) is None


def test_multimodal_stacks_are_summed():
    thinker = _gqa(num_hidden_layers=2)
    talker = _gqa(num_hidden_layers=3)
    config = SimpleNamespace(thinker_config=thinker, talker_config=talker)
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.growing_layers == 5


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
    estimate = KVCacheEstimate(layers=(LayerCache(kind="gqa", dims=2 * 2 * 64),) * 4,
                               dtype=torch.float16)
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
    small = KVCacheEstimate(layers=(LayerCache(kind="gqa", dims=2 * 1 * 8),) * 2,
                            dtype=torch.float16)
    assert any(unit in small.format() for unit in ("B", "KiB"))
    large = KVCacheEstimate(layers=(LayerCache(kind="gqa", dims=2 * 8 * 128),) * 80,
                            dtype=torch.float16)
    assert "GiB" in large.format()


def test_compressed_attention_counts_sliding_branch_plus_pooled_entries():
    """DeepSeek V4-style: bounded sliding K/V plus one entry per compress_rate."""
    config = SimpleNamespace(
        hidden_size=4096, num_hidden_layers=4, num_attention_heads=64,
        num_key_value_heads=1, head_dim=512, sliding_window=128,
        layer_types=["heavily_compressed_attention", "heavily_compressed_attention",
                     "compressed_sparse_attention", "compressed_sparse_attention"],
        compress_rates={"heavily_compressed_attention": 128, "compressed_sparse_attention": 4},
    )
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate is not None
    assert estimate.layer_type == "compressed"
    assert estimate.growing_layers == 4
    assert estimate.exact is True

    # Two HCA layers (rate 128) plus two CSA layers (rate 4), each storing one
    # shared K=V tensor of head_dim elements, capped at the 128-token window.
    def expected(tokens):
        hca = 2 * 512 * min(tokens, 128) + 2 * 512 * ((tokens + 127) // 128)
        csa = 2 * 512 * min(tokens, 128) + 2 * 512 * ((tokens + 3) // 4)
        return (hca + csa) * 2

    assert estimate.bytes_for(128) == expected(128)
    assert estimate.bytes_for(8192) == expected(8192)
    # The compressed slope dominates long contexts and is far below per-head K/V.
    assert estimate.bytes_for(32768) < 32768 * 2 * 64 * 512


def test_chunked_attention_is_capped_at_its_chunk_size():
    """Llama4 chunked layers are bounded, not full-context layers.

    Transformers builds a ``DynamicSlidingWindowLayer`` with
    ``sliding_window = attention_chunk_size`` for ``chunked_attention``. The
    previous expectation (full context) overestimated Llama4 by ~3.4x at 128k
    and made the estimate claim ``exact=True``.
    """

    config = _gqa(num_hidden_layers=48, attention_chunk_size=8192)
    config.layer_types = ["chunked_attention"] * 36 + ["full_attention"] * 12
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    per_layer = 2 * 8 * 128 * 2
    assert estimate.exact is True
    assert estimate.growing_layers == 48
    assert estimate.sliding_layers == 36
    assert estimate.window == 8192
    # Below the chunk size every layer grows with the context.
    assert estimate.bytes_for(4096) == 48 * 4096 * per_layer
    # Past the chunk size only the 12 full layers keep growing.
    assert estimate.bytes_for(32768) == (36 * 8192 + 12 * 32768) * per_layer
    assert estimate.bytes_per_token == 12 * per_layer
    assert "capped at 8k" in estimate.format()


def test_non_attention_blocks_contribute_nothing():
    """Nemotron-H dispatches moe/mlp blocks to non-attention mixers."""

    config = _gqa(num_hidden_layers=4)
    config.layer_types = ["linear_attention", "moe", "full_attention", "mlp"]
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate.exact is True
    assert estimate.growing_layers == 1
    assert estimate.linear_layers == 1
    assert estimate.block_layers == 2
    assert estimate.bytes_per_token == 2 * 8 * 128 * 2
    line = estimate.format()
    assert "2 non-attention block(s)" in line
    assert "linear-attention" in line


def test_fp8_kv_cache_is_one_byte_per_element():
    """vLLM/SGLang store the cache in fp8 to halve it; the estimate follows."""

    fp16 = estimate_kv_cache(_gqa(), dtype=torch.float16)
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("this torch build has no fp8 dtype")
    fp8 = estimate_kv_cache(_gqa(), dtype=fp8_dtype)
    assert fp8 is not None
    assert fp8.bytes_per_token * 2 == fp16.bytes_per_token


def _capped(**overrides):
    base = {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "max_position_embeddings": 8192,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_model_context_limit_is_carried_and_caps_the_answer():
    estimate = estimate_kv_cache(_capped(), dtype=torch.float16)
    assert estimate.context_limit == 8192
    # 20 GiB would buy far more than the model can take; the answer must not
    # claim 160k tokens for an 8k model.
    assert estimate.tokens_for(20 * 1024 ** 3) == 8192


def test_without_a_declared_limit_the_budget_decides():
    estimate = estimate_kv_cache(_gqa(), dtype=torch.float16)
    assert estimate.context_limit is None
    assert estimate.tokens_for(20 * 1024 ** 3) > 8192


def test_max_batch_is_the_budget_over_one_sequence():
    estimate = estimate_kv_cache(_capped(), dtype=torch.float16)
    budget = 4 * 1024 ** 3
    batch = estimate.max_batch(budget, 8192)
    assert batch > 0
    assert estimate.bytes_for(8192, batch_size=batch) <= budget
    assert estimate.bytes_for(8192, batch_size=batch + 1) > budget


def test_plan_contexts_never_grow_with_batch():
    estimate = estimate_kv_cache(_capped(), dtype=torch.float16)
    rows = estimate.plan(4 * 1024 ** 3)
    assert [batch for batch, _ in rows] == sorted(batch for batch, _ in rows)
    contexts = [context for _, context in rows]
    assert contexts == sorted(contexts, reverse=True)
    assert all(context <= estimate.context_limit for context in contexts)


def test_plan_drops_batches_that_fit_nothing():
    estimate = estimate_kv_cache(_capped(), dtype=torch.float16)
    rows = estimate.plan(4096)                      # a few tokens of budget
    assert rows == ()


def test_format_plan_collapses_equal_contexts():
    estimate = estimate_kv_cache(_capped(), dtype=torch.float16)
    line = estimate.format_plan(16 * 1024 ** 3)
    # 128 KiB per token: 16 GiB holds the 8k cap up to 16 sequences, 4k up to 32.
    assert "8k ctx up to 16 seq" in line
    assert "(model limit)" in line
    # One clause per distinct width, not one per batch size.
    assert line.count("ctx") == 2


def test_format_plan_marks_the_limit_only_when_the_cap_binds():
    estimate = estimate_kv_cache(_capped(), dtype=torch.float16)
    assert "(model limit)" not in estimate.format_plan(512 * 1024 ** 2)


def test_format_plan_reports_an_unusable_budget():
    estimate = estimate_kv_cache(_capped(), dtype=torch.float16)
    assert "fits no usable context" in estimate.format_plan(0)
