# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Regressions for #5698: unknown context and unsized recurrent state.

The estimator used the ``1 << 20`` binary-search bound as if it were a declared
model context, so a cache with no per-token growth claimed a million tokens.
A recurrent state whose shape the config does not declare was counted as zero,
so a shard made only of those layers looked free. Both turned "unknown" into a
confident number: a plan line named a device budget that did not produce its
rows, and a batch count ignored the state every sequence keeps.
"""

import types

import torch

from gptqmodel.utils.kv_cache import KVCacheEstimate, LayerCache, estimate_kv_cache
from gptqmodel.utils.resource_budget import DeviceBudget, ResourceBudget


def _hybrid_config(layer_types, *, state=True, **overrides):
    base = {
        "hidden_size": 64, "num_hidden_layers": len(layer_types),
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 128, "layer_types": list(layer_types),
    }
    if state:
        base.update({"mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2})
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


def _linear_only_cache(*, context_limit=None):
    """A cache whose only layers are recurrent, so nothing grows per token."""

    return KVCacheEstimate(
        layers=(
            LayerCache(kind="linear", state_unsized=True),
            LayerCache(kind="linear", state_unsized=True),
        ),
        dtype=torch.float16,
        context_limit=context_limit,
    )


def test_a_cache_without_per_token_growth_has_an_unknown_context():
    """No slope and no declared limit: the answer is unknown, not 1 << 20."""

    estimate = _linear_only_cache()
    assert estimate.growing_layers == 0
    assert estimate.unsized_state_layers == 2
    assert estimate.max_context_tokens(16 * 1024 ** 3) is None
    assert estimate.context_batch_plan(16 * 1024 ** 3) is None


def test_a_cache_without_per_token_growth_returns_the_declared_limit():
    """The same cache with a declared limit answers it, not the search bound."""

    estimate = _linear_only_cache(context_limit=4096)
    assert estimate.max_context_tokens(16 * 1024 ** 3) == 4096
    assert estimate.max_context_tokens(16 * 1024 ** 3) != (1 << 20)
    assert estimate.context_batch_plan(16 * 1024 ** 3)[0] == (1, 4096)


def test_a_model_capped_plan_does_not_claim_a_device_budget():
    """The rows came from the model limit, so no device budget labels them."""

    config = _hybrid_config(
        ["linear_attention", "full_attention"], max_position_embeddings=128)
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, 20 * 1024 ** 3, 8)],
        layer_devices=(0, 1),
    )

    plan = budget.format_plan()
    assert plan is not None
    assert plan.startswith("KV plan: ")
    assert "limited by device" not in plan
    # A " -> " after "KV plan:" would print a budget no single card spent.
    assert " -> " not in plan


def test_a_device_bound_plan_still_names_the_binding_device():
    """Control: when a card binds, the plan keeps its budget and device label."""

    config = _hybrid_config(
        ["linear_attention", "full_attention"], max_position_embeddings=100_000)
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, 1024 ** 2, 8)],
        layer_devices=(0, 1),
    )

    plan = budget.format_plan()
    assert plan is not None
    assert "limited by device 1" in plan
    assert " -> " in plan


def test_an_unsized_state_only_shard_is_kept_but_does_not_cap_context():
    """A shard with undeclared state is memory, but only the batch is unknown."""

    config = _hybrid_config(
        ["linear_attention", "linear_attention", "full_attention", "full_attention"],
        state=False,
    )
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, 4 * 1024 ** 3, 8)],
        layer_devices=(1, 1, 0, 0),
    )

    subsets = budget._cache_device_subsets()
    assert subsets is not None
    assert [device.index for device, _ in subsets] == [0, 1]
    assert [subset.unsized_state_layers for _, subset in subsets] == [0, 2]
    assert budget.kv_cache.unsized_state_layers == 2
    # The unsized shard holds memory, but recurrent state is not a per-token
    # cost: only the growing shard constrains the context.
    assert budget._per_device_limits(batch_size=1, kv_budget_bytes=None) == {0: 128}
    assert budget.max_context_tokens() == 128
    assert budget.max_sequences(1024) is None


def test_unsized_hybrid_state_keeps_the_context_but_not_the_batch():
    """A hybrid layer grows K/V *and* holds an unsized recurrent branch.

    The per-token cache is still measurable, so the context answer stands; the
    per-sequence state is not, so the sequence count must stay unknown.
    """

    config = _hybrid_config(["hybrid", "full_attention"], state=False)
    budget = _budget(
        config,
        devices=[(0, 20 * 1024 ** 3, 16)],
        layer_devices=(0, 0),
    )

    assert budget.kv_cache.growing_layers == 2
    assert budget.kv_cache.unsized_state_layers == 1
    assert budget.max_context_tokens() == 128
    assert budget.max_sequences(1024) is None
