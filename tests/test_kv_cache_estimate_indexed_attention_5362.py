# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Regression test for #5362: indexed attention also retains indexer keys.

DeepSeek-V3.2 / GLM MoE DSA layers use transformers' ``DynamicIndexedLayer``,
which keeps the latent MLA K/V *and* a ``[batch, seq, index_head_dim]`` key
cache. The estimator counted only the latent branch and still reported
``is_exact=True``: on the tiny model below, 240 bytes at five tokens against
400 actually retained.
"""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.kv_cache import estimate_kv_cache


def _tiny_config():
    from transformers import DeepseekV32Config

    return DeepseekV32Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        max_position_embeddings=64,
        attn_implementation="eager",
    )


def test_indexed_attention_estimate_matches_the_native_cache():
    """The estimate equals the tensors a real DeepSeek-V3.2 forward retains."""

    pytest.importorskip("transformers.models.deepseek_v32")
    from transformers import DeepseekV32ForCausalLM

    config = _tiny_config()
    estimate = estimate_kv_cache(config, dtype=torch.float32)
    assert estimate is not None
    assert estimate.is_exact is True

    # Latent keys + RoPE values + one indexer vector per token, all fp32.
    per_token = (
        config.kv_lora_rank + config.qk_rope_head_dim + config.index_head_dim
    ) * 4
    assert estimate.kv_bytes_per_token == per_token

    torch.manual_seed(0)
    model = DeepseekV32ForCausalLM(config).eval()
    for tokens in (1, 2, 5, 16):
        with torch.no_grad():
            output = model(torch.randint(3, config.vocab_size, (1, tokens)), use_cache=True)
        layer = output.past_key_values.layers[0]
        measured = sum(
            tensor.nbytes for tensor in (layer.keys, layer.values, layer.indexer_keys)
        )
        estimated = estimate.kv_bytes_for(tokens)
        assert estimated == measured, (tokens, estimated, measured)
        assert estimate.kv_bytes_for(tokens, batch_size=2) == 2 * measured

    assert estimate.layer_type == "indexed"
    assert estimate.indexer_layers == 1


def test_indexed_attention_without_a_declared_indexer_width_is_partial():
    """An unknown indexer layout is reported instead of looking exact."""

    config = SimpleNamespace(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        layer_types=["indexed_attention", "indexed_attention"],
    )
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate is not None
    assert estimate.is_exact is False
    assert estimate.partial == ("indexed_attention indexer layout not declared",)
    assert "partial: indexed_attention indexer layout not declared" in estimate.describe()
    # The latent branch is still counted; only the indexer term is unknown.
    assert estimate.kv_bytes_per_token == 2 * (16 + 8) * 2
