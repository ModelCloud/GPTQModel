# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Regressions for the two KV-cache defects found reviewing #5623.

#5623 made recurrent state a counted, per-sequence cost. Two consequences were
missed:

* a shard that owns *only* recurrent layers holds memory but no growing cache,
  so the sharded planners skipped it and reported the other cards' room;
* ``Zamba2Config`` publishes its group count as ``mamba_ngroups``, so the
  Mamba-2 branch never matched and the estimate fell back to the Mamba-1
  layout - close enough to look right, wrong by the conv's group term.
"""

import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.utils.kv_cache import estimate_kv_cache
from gptqmodel.utils.resource_budget import DeviceBudget, ResourceBudget


def _hybrid_config(layer_types, **overrides):
    base = {
        "hidden_size": 64, "num_hidden_layers": len(layer_types),
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 128, "layer_types": list(layer_types),
        "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _budget(config, devices, layer_devices, *, reserve=0.0, dtype=torch.float16):
    estimate = estimate_kv_cache(config, dtype=dtype)
    assert estimate is not None and len(estimate.layers) == len(layer_devices)
    return ResourceBudget(
        devices=tuple(DeviceBudget(index=index, free_bytes=free, total_bytes=24 * 1024 ** 3,
                                   weights_bytes=weights)
                      for index, free, weights in devices),
        kv_cache=estimate,
        layer_devices=tuple(layer_devices),
        reserve_fraction=reserve,
    )


def _state_only_shard_budget():
    """Device 1 owns both recurrent layers and room for three sequences."""

    config = _hybrid_config(
        ["linear_attention", "linear_attention", "full_attention", "full_attention"])
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate is not None
    state = estimate.state_bytes_per_sequence
    assert state > 0
    # 3 sequences of state fit on the recurrent shard, four do not.
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, 3 * state, 8)],
        layer_devices=(1, 1, 0, 0),
    )
    return budget, state


def test_a_state_only_shard_binds_the_sequence_count():
    """Regression: the recurrent shard was skipped, so its state was free."""

    budget, state = _state_only_shard_budget()

    # Only the attention shard grows per token; the recurrent shard is a
    # constant per live sequence and is the binding one.
    assert budget.kv_cache.growing_layers == 2
    assert budget.kv_cache.state_layers == 2
    assert budget.max_sequences(1024) == 3
    # Before the fix the attention card alone answered, ~100x higher.
    assert budget.max_sequences(1024) < (20 * 1024 ** 3) // (1024 * budget.kv_cache.kv_bytes_per_token + state)


def test_a_state_only_shard_binds_the_batch_size():
    """Regression: four sequences of state do not fit, so the context is 0."""

    budget, state = _state_only_shard_budget()

    assert budget.max_context_tokens(batch_size=1) == 128
    assert budget.max_context_tokens(batch_size=2) == 128
    # 4 * state > 3 * state: no room left for a single token on that shard.
    assert budget.max_context_tokens(batch_size=4) == 0
    assert (4, 0) not in budget.context_batch_plan()


def test_a_state_only_shard_without_room_reports_zero_not_the_model_limit():
    """The model limit is not a licence to ignore a full recurrent shard."""

    config = _hybrid_config(
        ["linear_attention", "full_attention"])
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate is not None
    state = estimate.state_bytes_per_sequence
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, state // 2, 8)],
        layer_devices=(1, 0),
    )
    assert budget.max_context_tokens(batch_size=1) == 0
    assert budget.max_sequences(1024) == 0


def test_a_shard_that_caches_nothing_still_does_not_constrain():
    """Control: a device with no growing cache *and* no state is still moot."""

    config = _hybrid_config(["full_attention", "moe"])
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, None, 8)],
        layer_devices=(0, 1),
    )
    assert budget.kv_cache.block_layers == 1
    # Device 1 owns no cache at all, so its missing snapshot cannot bind.
    assert budget.max_context_tokens() is not None
    assert budget.max_sequences(1024) is not None


def test_an_unmeasured_state_only_shard_leaves_the_plan_unknown():
    """Control: the shard holds state now, so its missing snapshot matters."""

    config = _hybrid_config(
        ["linear_attention", "linear_attention", "full_attention", "full_attention"])
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, None, 8)],
        layer_devices=(1, 1, 0, 0),
    )
    assert budget.max_context_tokens() is None
    assert budget.max_sequences(1024) is None


def _tiny_zamba2_config():
    return AutoConfig.for_model(
        "zamba2", num_hidden_layers=2, hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, n_mamba_heads=4,
        mamba_ngroups=1, chunk_size=16, max_position_embeddings=128,
        layers_block_type=["mamba", "hybrid"], num_mem_blocks=1,
        use_shared_attention_adapter=False, use_mem_rope=False,
        use_mamba_kernels=False, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        tie_word_embeddings=True,
    )


def _cache_bytes(cache):
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


def test_zamba2_state_uses_its_declared_group_count():
    """Regression: ``mamba_ngroups`` missed, so Zamba2 used the Mamba-1 layout.

    The conv sees ``d_ssm + 2 * groups * d_state`` channels; the fallback
    dropped the ``2 * groups * d_state`` term and under-counted every layer.
    """

    config = _tiny_zamba2_config()
    assert config.mamba_ngroups == 1
    assert not hasattr(config, "mamba_n_groups")

    estimate = estimate_kv_cache(config, dtype=torch.bfloat16)
    assert estimate is not None
    assert estimate.is_exact is True

    model = AutoModelForCausalLM.from_config(config).eval().to(torch.bfloat16)
    for tokens in (4, 32):
        with torch.no_grad():
            output = model(torch.randint(3, 20, (1, tokens)), use_cache=True)
        measured = _cache_bytes(output.past_key_values)
        assert measured == estimate.kv_bytes_for(tokens), (tokens, measured, estimate.kv_bytes_for(tokens))
