# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Regression tests for bounded and KV-shared layers in the cache estimate.

Three reports share one root cause: the estimator resolved a layer's cache
layout from the global config instead of the layout transformers actually
builds. #4542 is bounded attention (Llama 4 chunked, implicit Mistral/Phi-3/
Starcoder2 sliding), #4543 is Gemma 4's heterogeneous per-layer head shapes,
and #4544 is the Gemma 3n/4 KV-shared tail. Each test asserts the reported
number and, where a model can be built, the bytes transformers allocates.
"""

from collections.abc import Sequence

import pytest
import torch
from transformers import (
    Gemma3nForCausalLM,
    Gemma3nTextConfig,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Llama4ForCausalLM,
    Llama4TextConfig,
    MistralConfig,
    MistralForCausalLM,
    Phi3Config,
    Starcoder2Config,
)

from gptqmodel.utils.kv_cache import estimate_kv_cache


_KIB = 1024


def _estimate(config):
    estimate = estimate_kv_cache(config, dtype=torch.bfloat16)
    assert estimate is not None
    return estimate


def _measured(config, model_cls, tokens):
    """Bytes and cache-layer count transformers actually allocates."""

    model = model_cls(config).eval().to(torch.bfloat16)
    with torch.no_grad():
        output = model(torch.randint(3, config.vocab_size, (1, tokens)), use_cache=True)
    layers = output.past_key_values.layers
    measured = sum(
        tensor.numel() * tensor.element_size()
        for layer in layers
        for tensor in (layer.keys, layer.values)
    )
    return measured, len(layers)


def _dynamic_cache_bytes(estimate, tokens):
    """``bytes_for`` adjusted for ``DynamicSlidingWindowLayer``'s window - 1."""

    elements = 0
    for layer in estimate.layers:
        if layer.dims <= 0:
            continue
        span = min(tokens, layer.window - 1) if layer.window else tokens
        elements += layer.dims * span
    return elements * estimate.kv_element_bytes


def _assert_matches_transformers(config, model_cls, tokens):
    estimate = _estimate(config)
    measured, cache_layers = _measured(config, model_cls, tokens)
    assert cache_layers == estimate.growing_layers
    assert measured == _dynamic_cache_bytes(estimate, tokens)
    return estimate


def test_llama4_scout_chunked_layers_are_capped_at_the_chunk_size():
    """#4542: 36 chunked layers cap at 8k; only the 12 full layers grow."""

    config = Llama4TextConfig()
    estimate = _estimate(config)
    per_layer = 2 * 8 * 128 * 2
    assert estimate.is_exact is True
    assert estimate.growing_layers == 48
    assert estimate.sliding_layers == 36
    assert estimate.window == 8192
    assert estimate.kv_bytes_per_token == 12 * per_layer
    assert estimate.kv_bytes_for(131072) == (36 * 8192 + 12 * 131072) * per_layer
    naive_full_context = 48 * 131072 * per_layer
    assert naive_full_context / estimate.kv_bytes_for(131072) >= 3


def test_llama4_chunked_attention_matches_the_allocated_cache():
    config = Llama4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        intermediate_size_mlp=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_chunk_size=16,
        num_local_experts=2,
        layer_types=["chunked_attention"] * 3 + ["full_attention"],
    )
    estimate = _assert_matches_transformers(config, Llama4ForCausalLM, 256)
    per_layer = 2 * 2 * 16 * 2
    assert estimate.sliding_layers == 3
    assert estimate.kv_bytes_per_token == per_layer
    assert estimate.kv_bytes_for(256) == (3 * 16 + 256) * per_layer
    assert (4 * 256 * per_layer) / estimate.kv_bytes_for(256) >= 3


@pytest.mark.parametrize(
    "config",
    [
        MistralConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, sliding_window=16,
        ),
        Starcoder2Config(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            sliding_window=16,
        ),
        Phi3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            sliding_window=16,
        ),
    ],
    ids=["mistral", "starcoder2", "phi3"],
)
def test_implicit_sliding_window_without_layer_types(config):
    """#4542: no ``layer_types`` plus a window means every layer is sliding."""

    estimate = _estimate(config)
    assert estimate.is_exact is True
    assert estimate.sliding_layers == 2
    assert estimate.window == 16
    assert estimate.kv_bytes_per_token == 0
    per_layer = estimate.layers[0].dims * estimate.kv_element_bytes
    assert (2 * 128 * per_layer) / estimate.kv_bytes_for(128) == 8


def test_mistral_implicit_sliding_matches_the_allocated_cache():
    config = MistralConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, sliding_window=16,
    )
    estimate = _assert_matches_transformers(config, MistralForCausalLM, 64)
    assert estimate.kv_bytes_per_token == 0
    assert estimate.kv_bytes_for(64) == 2 * 16 * estimate.layers[0].dims * estimate.kv_element_bytes


def test_mistral_full_config_is_bounded_at_32k_context():
    """#4542: the issue's table expects 512 MiB at 32k, not 16 GiB at 128k."""

    estimate = _estimate(MistralConfig(num_hidden_layers=32, sliding_window=4096))
    per_layer = estimate.layers[0].dims * estimate.kv_element_bytes
    assert estimate.kv_bytes_per_token == 0
    assert estimate.kv_bytes_for(32768) == 32 * 4096 * per_layer
    assert estimate.kv_bytes_for(32768) == 512 * 1024 * 1024
    assert (32 * 32768 * per_layer) / estimate.kv_bytes_for(32768) == 8


def test_gemma4_per_layer_view_supplies_distinct_global_head_shapes():
    """#4543: the view is a Sequence, and global layers are 4 x 512 heads."""

    config = Gemma4TextConfig(
        vocab_size=512,
        hidden_size=5376,
        intermediate_size=64,
        num_hidden_layers=60,
        num_attention_heads=32,
        num_key_value_heads=16,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=4,
        attention_k_eq_v=True,
        sliding_window=1024,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
    )
    assert isinstance(config.per_layer_config, Sequence)
    assert not isinstance(config.per_layer_config, (list, tuple))

    estimate = _estimate(config)
    assert estimate.is_exact is True
    assert estimate.growing_layers == 60
    assert estimate.sliding_layers == 50
    # 10 global layers * 2 (K and V) * 4 heads * 512 head_dim * 2 bytes.
    assert estimate.kv_bytes_per_token == 10 * 2 * 4 * 512 * 2 == 80 * _KIB
    assert estimate.kv_bytes_for(131072) == (
        50 * 1024 * (2 * 16 * 256) + 10 * 131072 * (2 * 4 * 512)
    ) * 2


def test_gemma4_measured_cache_uses_the_per_layer_head_shapes():
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        attention_k_eq_v=True,
        sliding_window=4,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
    )
    estimate = _assert_matches_transformers(config, Gemma4ForCausalLM, 8)
    assert estimate.layers[0].dims == 2 * 2 * 16
    assert estimate.layers[-1].dims == 2 * 1 * 32


def test_ambiguous_global_head_dim_without_a_layer_view_is_unknown():
    """#4543: do not fall back to ``hidden_size // heads`` when the read is ambiguous."""

    from transformers.integrations.heterogeneity.configuration_utils import (
        AmbiguousGlobalPerLayerAttributeError,
    )

    class _AmbiguousConfig:
        hidden_size = 4096
        num_hidden_layers = 2
        num_attention_heads = 32
        num_key_value_heads = 8

        @property
        def head_dim(self):
            raise AmbiguousGlobalPerLayerAttributeError("head_dim varies per layer")

    assert estimate_kv_cache(_AmbiguousConfig(), dtype=torch.bfloat16) is None


def _gemma_layer_types(layers):
    return [
        "full_attention" if (index + 1) % 5 == 0 else "sliding_attention"
        for index in range(layers)
    ]


def test_gemma3n_kv_shared_tail_is_not_counted():
    """#4544: 10 shared layers out of 30; 4 global layers at 2 KiB is 8 KiB/token."""

    layers = 30
    config = Gemma3nTextConfig(
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size=2048,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        sliding_window=512,
        layer_types=_gemma_layer_types(layers),
        num_kv_shared_layers=10,
        hidden_size_per_layer_input=0,
    )
    estimate = _estimate(config)
    assert estimate.is_exact is True
    assert estimate.shared_layers == 10
    assert estimate.growing_layers == 20
    assert estimate.sliding_layers == 16
    assert estimate.kv_bytes_per_token == 4 * 2 * 2 * 256 * 2 == 8 * _KIB
    assert "10 KV-shared layer(s)" in estimate.describe()


def test_gemma3n_measured_cache_skips_the_shared_tail():
    config = Gemma3nTextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=4,
        layer_types=_gemma_layer_types(10),
        num_kv_shared_layers=5,
        hidden_size_per_layer_input=0,
    )
    estimate = _assert_matches_transformers(config, Gemma3nForCausalLM, 8)
    assert estimate.shared_layers == 5
    assert estimate.growing_layers == 5


def test_gemma4_e2b_combines_per_layer_shapes_and_kv_sharing():
    """#4543 + #4544: only 15 of 35 layers cache, 3 of them global at 2 KiB."""

    config = Gemma4TextConfig(
        vocab_size=512,
        hidden_size=1536,
        intermediate_size=64,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=None,
        attention_k_eq_v=False,
        sliding_window=512,
        layer_types=_gemma_layer_types(35),
        num_kv_shared_layers=20,
        hidden_size_per_layer_input=0,
    )
    estimate = _estimate(config)
    assert estimate.shared_layers == 20
    assert estimate.growing_layers == 15
    assert estimate.sliding_layers == 12
    assert estimate.kv_bytes_per_token == 3 * 2 * 1 * 512 * 2 == 6 * _KIB
