# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""A loaded model has to report what it costs, and say which numbers are measured.

Most users load a quantized checkpoint rather than quantize one. `model.budget()`
is the single entry point: measured weight storage and device memory sit next to
the estimated KV-cache footprint, and every accessor names its units so a caller
does not have to read the implementation to know what a value counts.
"""

import types

import torch

import gptqmodel.models.base as base
from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.resource_budget import ESTIMATED, MEASURED, ResourceBudget


def _stub(*, layers=2, heads=4, kv_heads=2, hidden=64, dtype=torch.bfloat16):
    config = types.SimpleNamespace(
        hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
    )
    parameters = [torch.zeros(4, dtype=torch.int64), torch.zeros(8, dtype=dtype)]
    inner = types.SimpleNamespace(config=config, parameters=lambda: iter(parameters),
                                  buffers=lambda: iter(()))
    return types.SimpleNamespace(model=inner)


class _FakeTensor:
    """Minimal tensor stand-in that reports a CUDA device."""

    is_meta = False

    def __init__(self, index, numel=8, size=2, device_type="cuda"):
        self._numel, self._size = numel, size
        self.device = types.SimpleNamespace(type=device_type, index=index if device_type == "cuda" else None)

    def numel(self):
        return self._numel

    def element_size(self):
        return self._size


def _cuda_stub(*, layers=2, heads=4, kv_heads=2, hidden=64, dtype=torch.bfloat16, device_index=0):
    config = types.SimpleNamespace(
        hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
    )
    params = [_FakeTensor(device_index, numel=8, size=2)]
    inner = types.SimpleNamespace(config=config, parameters=lambda: iter(params),
                                  buffers=lambda: iter(()))
    return types.SimpleNamespace(model=inner)


def test_budget_reports_measured_storage_and_an_estimated_cache():
    budget = BaseQModel.budget(_stub())
    assert isinstance(budget, ResourceBudget)
    # 4 int64 + 8 bf16 bytes, measured from the tensors themselves.
    assert budget.weights_bytes == 4 * 8 + 8 * 2
    assert budget.kv_cache is not None
    # 2 layers * 2 (K and V) * 2 kv_heads * 16 head_dim * 2 bytes
    assert budget.kv_cache.kv_bytes_per_token == 2 * 2 * 2 * 16 * 2


def test_every_accessor_states_whether_it_is_measured_or_estimated():
    assert ResourceBudget.ACCURACY["weights_bytes"] == MEASURED
    assert ResourceBudget.ACCURACY["devices"] == MEASURED
    assert ResourceBudget.ACCURACY["kv_cache"] == ESTIMATED
    assert ResourceBudget.ACCURACY["max_context_tokens"] == ESTIMATED


def test_meta_tensors_are_not_counted_as_storage():
    """A lazily loaded checkpoint carries shapes but no bytes."""

    class _Meta:
        is_meta = True
        def numel(self): return 1_000_000
        def element_size(self): return 4

    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2)
    inner = types.SimpleNamespace(config=config, parameters=lambda: iter([_Meta()]),
                                  buffers=lambda: iter(()))
    budget = BaseQModel.budget(types.SimpleNamespace(model=inner))
    assert budget.weights_bytes == 0


def test_kv_budget_holds_back_the_reserve(monkeypatch):
    monkeypatch.setattr(base, "device_memory_bytes", lambda index=None: (10 * 1024 ** 3, 24 * 1024 ** 3))
    budget = BaseQModel.budget(_cuda_stub(), reserve=0.1)
    assert budget.device_count == 1
    assert budget.devices[0].free_bytes == 10 * 1024 ** 3
    assert budget.devices[0].total_bytes == 24 * 1024 ** 3
    assert budget.kv_budget_bytes == int(9 * 1024 ** 3)


def test_planning_accessors_name_their_units(monkeypatch):
    monkeypatch.setattr(base, "device_memory_bytes", lambda index=None: (4 * 1024 ** 3, 24 * 1024 ** 3))
    budget = BaseQModel.budget(_cuda_stub(), reserve=0.0)
    rows = budget.context_batch_plan()
    assert rows and all(context > 0 for _batch, context in rows)
    assert budget.kv_cache_bytes_for(1024) == budget.kv_cache.kv_bytes_for(1024)
    assert budget.max_sequences(1024) >= 1
    assert budget.max_context_tokens(batch_size=1) >= 1024


def test_planning_returns_none_without_a_device(monkeypatch):
    monkeypatch.setattr(base, "device_memory_bytes", lambda index=None: (None, None))
    budget = BaseQModel.budget(_stub())
    assert budget.kv_budget_bytes is None
    assert budget.max_context_tokens() is None
    assert budget.context_batch_plan() == ()
    # An explicit budget still works without a device.
    assert budget.max_context_tokens(kv_budget_bytes=1024 ** 2) > 0


def test_describe_marks_measured_and_estimated(monkeypatch):
    monkeypatch.setattr(base, "device_memory_bytes", lambda index=None: (8 * 1024 ** 3, 24 * 1024 ** 3))
    lines = BaseQModel.budget(_cuda_stub()).describe()
    text = "\n".join(lines)
    assert "(measured)" in text
    assert "(estimated)" in text
    assert "weights" in text and "KV cache" in text


def test_without_a_kv_estimate_the_budget_says_so(monkeypatch):
    monkeypatch.setattr(base, "device_memory_bytes", lambda index=None: (8 * 1024 ** 3, 24 * 1024 ** 3))
    inner = types.SimpleNamespace(config=types.SimpleNamespace(), parameters=lambda: iter([]),
                                  buffers=lambda: iter(()))
    lines = BaseQModel.budget(types.SimpleNamespace(model=inner)).describe()
    assert any("KV cache unknown" in line for line in lines)


def test_log_budget_emits_the_described_lines(monkeypatch, capsys):
    monkeypatch.setattr(base, "device_memory_bytes", lambda index=None: (8 * 1024 ** 3, 24 * 1024 ** 3))
    BaseQModel.log_budget(_cuda_stub())
    out = capsys.readouterr().out
    assert "Budget: weights" in out
    assert "KV cache" in out


def test_kv_cache_estimate_is_the_budget_slice():
    estimate = BaseQModel.kv_cache_estimate(_stub(), dtype=torch.float32)
    assert estimate is not None
    assert estimate.dtype == torch.float32


def test_two_devices_are_reported_separately(monkeypatch):
    """A sharded model has one budget per device, not one total."""
    monkeypatch.setattr(
        base, "device_memory_bytes",
        lambda index=None: {0: (10 * 1024 ** 3, 24 * 1024 ** 3), 1: (6 * 1024 ** 3, 24 * 1024 ** 3)}[index],
    )
    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    params = [_FakeTensor(0, 8, 2), _FakeTensor(1, 4, 2)]
    inner = types.SimpleNamespace(config=config, parameters=lambda: iter(params), buffers=lambda: iter(()))
    budget = BaseQModel.budget(types.SimpleNamespace(model=inner), reserve=0.0)

    assert budget.device_count == 2
    assert budget.is_sharded is True
    assert [device.weights_bytes for device in budget.devices] == [16, 8]
    assert budget.weights_bytes == 24
    assert budget.devices[0].free_bytes == 10 * 1024 ** 3
    assert budget.devices[1].free_bytes == 6 * 1024 ** 3


def test_sharded_planning_is_limited_by_the_tightest_device(monkeypatch):
    """Layer i lives on the device that holds it; the answer is the min over devices."""
    from dataclasses import replace

    from gptqmodel.utils.kv_cache import estimate_kv_cache
    from gptqmodel.utils.resource_budget import DeviceBudget, ResourceBudget

    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate is not None and len(estimate.layers) == 2

    devices = (
        DeviceBudget(index=0, free_bytes=20 * 1024 ** 3, total_bytes=24 * 1024 ** 3, weights_bytes=16),
        DeviceBudget(index=1, free_bytes=1024 ** 2, total_bytes=24 * 1024 ** 3, weights_bytes=8),
    )
    budget = ResourceBudget(devices=devices, kv_cache=estimate, layer_devices=(0, 1), reserve_fraction=0.0)

    # The second device has almost no memory and owns one of two layers, so it binds.
    per_layer = estimate.layers[0].dims * 2
    assert budget.max_context_tokens() == (1024 ** 2) // per_layer
    assert budget.max_context_tokens() < estimate.max_context_tokens(20 * 1024 ** 3)

    # Without a layer map the sharded answer is unknown, not a guess.
    unmapped = replace(budget, layer_devices=None)
    assert unmapped.max_context_tokens() is None
    assert any("needs its layer-to-device map" in line for line in unmapped.describe())


class _FakeLayer:
    """One decoder layer holding a single tensor on ``device_index``."""

    def __init__(self, device_index):
        self._tensor = _FakeTensor(device_index, numel=4, size=2)

    def parameters(self):
        return iter([self._tensor])

    def buffers(self):
        return iter(())


class _FakeInner:
    """Stands in for the HF model: config, weights, and layer paths."""

    def __init__(self, config, layer_devices, weights=None):
        self.config = config
        self.model = types.SimpleNamespace()
        self._layer_devices = tuple(layer_devices)
        self._weights = list(weights or [_FakeTensor(0, numel=8, size=2)])

    def parameters(self):
        return iter(self._weights)

    def buffers(self):
        return iter(())

    def get_submodule(self, path):
        assert path == "model.layers"
        return [_FakeLayer(index) for index in self._layer_devices]


def _wrapper_stub(inner):
    """Wrapper that carries the layer paths, like BaseQModel does."""

    return types.SimpleNamespace(model=inner, extract_layers_node=lambda: ["model.layers"])


def test_layer_devices_resolve_through_the_wrapper(monkeypatch):
    """Regression: `extract_layers_node` lives on the wrapper, not the HF model."""

    monkeypatch.setattr(base, "device_memory_bytes",
                        lambda index=None: (8 * 1024 ** 3, 24 * 1024 ** 3))
    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    # A real split puts weights on both devices, not just the first.
    weights = [_FakeTensor(0, numel=8, size=2), _FakeTensor(1, numel=8, size=2)]
    inner = _FakeInner(config, layer_devices=(0, 1), weights=weights)
    wrapper = _wrapper_stub(inner)

    budget = BaseQModel.budget(wrapper)

    assert budget.layer_devices == (0, 1)
    assert budget.is_sharded is True
    assert budget.max_context_tokens() is not None


def test_a_host_resident_buffer_does_not_make_a_model_sharded(monkeypatch):
    """Regression: transformers keeps `rotary_emb.inv_freq` on the CPU."""

    monkeypatch.setattr(base, "device_memory_bytes",
                        lambda index=None: (8 * 1024 ** 3, 24 * 1024 ** 3))
    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    host_weights = [_FakeTensor(0, numel=8, size=2), _FakeTensor(-1, numel=16, size=1, device_type="cpu")]
    inner = _FakeInner(config, layer_devices=(0, 0), weights=host_weights)
    wrapper = _wrapper_stub(inner)

    budget = BaseQModel.budget(wrapper)

    assert budget.device_count == 2            # host bucket still reported
    assert budget.is_sharded is False          # but it is not a shard
    assert budget.host_weights_bytes == 16 * 1
    assert budget.max_context_tokens() is not None
    lines = budget.describe()
    assert any("host" in line for line in lines)
    assert not any("device -1" in line for line in lines)


def _budget_for_config(config, devices, layer_devices, *, reserve=0.0):
    from gptqmodel.utils.kv_cache import estimate_kv_cache
    from gptqmodel.utils.resource_budget import DeviceBudget, ResourceBudget

    estimate = estimate_kv_cache(config, dtype=torch.float16)
    assert estimate is not None and len(estimate.layers) == len(layer_devices)
    return ResourceBudget(
        devices=tuple(DeviceBudget(index=index, free_bytes=free, total_bytes=24 * 1024 ** 3,
                                   weights_bytes=weights)
                      for index, free, weights in devices),
        kv_cache=estimate,
        layer_devices=tuple(layer_devices),
        reserve_fraction=reserve,
    )


def test_sharded_plan_is_the_minimum_over_devices_not_the_sum_of_budgets():
    """Regression: the printed plan spent every device's budget on every layer.

    `max_context_tokens` was per device, but `describe`/the plan line formatted
    the whole layer stack against `kv_budget_bytes`, i.e. the sum of all cards.
    A model split one layer per card, with the second card nearly full, printed
    the context the *aggregate* would buy - ~100x what the tight card holds.
    """

    from gptqmodel.utils.kv_cache import human_tokens

    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=1_000_000)
    budget = _budget_for_config(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, 1024 ** 2, 8)],
        layer_devices=(0, 1),
    )
    per_layer = budget.kv_cache.layers[0].dims * budget.kv_cache.kv_element_bytes
    tightest = (1024 ** 2) // per_layer

    assert budget.max_context_tokens() == tightest
    # describe() must agree with the plan it reports. On the unfixed code this
    # line held the aggregate-budget answer (1000000 ctx at 1 seq).
    text = "\n".join(budget.describe())
    assert f"{human_tokens(tightest)} ctx at 1 seq" in text
    assert human_tokens(1_000_000) not in text
    # The aggregate budget would buy ~100x more; no single card can spend it.
    assert budget.kv_cache.max_context_tokens(20 * 1024 ** 3) > tightest * 100
    plan = budget.format_plan()
    assert plan is not None
    assert f"{human_tokens(tightest)} ctx at 1 seq" in plan
    assert "limited by device 1" in plan


def test_a_device_with_only_constant_state_layers_does_not_zero_the_plan():
    """Regression: a mamba-only shard capped a granite-4.0-h plan at 0 tokens.

    A device whose layers are all linear/recurrent holds constant state, not a
    per-token cache; its own `max_context_tokens` is 0, but that 0 says nothing
    about the context the model can serve.
    """

    from gptqmodel.utils.kv_cache import human_tokens

    config = types.SimpleNamespace(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=100_000,
        layer_types=["linear_attention", "linear_attention", "full_attention", "full_attention"],
    )
    budget = _budget_for_config(
        config,
        devices=[(0, 20 * 1024 ** 3, 16), (1, 4 * 1024 ** 3, 8)],
        layer_devices=(0, 0, 1, 1),
    )

    assert budget.kv_cache.linear_layers == 2
    assert budget.kv_cache.growing_layers == 2
    # Device 0 holds only recurrent layers, so only device 1 constrains context.
    assert budget._per_device_limits(batch_size=1, kv_budget_bytes=None) == {1: 100_000}
    assert budget.max_context_tokens() == 100_000
    assert budget.context_batch_plan()[0] == (1, 100_000)
    assert budget.max_sequences(1024) > 0
    plan = budget.format_plan()
    assert plan is not None and f"{human_tokens(100_000)} ctx" in plan


class _AliasedInner(_FakeInner):
    """VL-style wrapper: two paths that name the same decoder stack."""

    def __init__(self, config, layer_devices, weights=None, alias=True):
        super().__init__(config, layer_devices, weights)
        self._alias = alias
        self._layers = [_FakeLayer(index) for index in self._layer_devices]

    def get_submodule(self, path):
        assert path in ("model.language_model.layers", "language_model.layers")
        if self._alias:
            return self._layers
        return [_FakeLayer(index) for index in self._layer_devices]


def _alias_wrapper(inner):
    return types.SimpleNamespace(model=inner, extract_layers_node=lambda: [
        "model.language_model.layers", "language_model.layers"])


def test_aliased_layer_paths_resolve_to_the_same_stack(monkeypatch):
    """Regression: Qwen2.5-VL names its layer list twice; planning went unavailable."""

    monkeypatch.setattr(base, "device_memory_bytes",
                        lambda index=None: (8 * 1024 ** 3, 24 * 1024 ** 3))
    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    weights = [_FakeTensor(0, numel=8, size=2), _FakeTensor(1, numel=8, size=2)]
    inner = _AliasedInner(config, layer_devices=(0, 1), weights=weights)
    budget = BaseQModel.budget(_alias_wrapper(inner))

    assert budget.layer_devices == (0, 1)
    assert budget.max_context_tokens() is not None


def test_distinct_layer_paths_stay_unresolved(monkeypatch):
    """Two genuinely different stacks must not be mapped onto one config list."""

    monkeypatch.setattr(base, "device_memory_bytes",
                        lambda index=None: (8 * 1024 ** 3, 24 * 1024 ** 3))
    config = types.SimpleNamespace(hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    weights = [_FakeTensor(0, numel=8, size=2), _FakeTensor(1, numel=8, size=2)]
    inner = _AliasedInner(config, layer_devices=(0, 1), weights=weights, alias=False)
    budget = BaseQModel.budget(_alias_wrapper(inner))

    assert budget.layer_devices is None
    assert budget.max_context_tokens() is None
