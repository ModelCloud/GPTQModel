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

    def __init__(self, index, numel=8, size=2):
        self._numel, self._size = numel, size
        self.device = types.SimpleNamespace(type="cuda", index=index)

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
