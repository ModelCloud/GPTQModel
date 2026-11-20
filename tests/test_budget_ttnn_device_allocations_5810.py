# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""``model.budget()`` has to count native device allocations (#5810).

A TTNN layer uploads its packed operands into the mesh allocator, where they
are ``ttnn.Tensor`` objects, not Torch parameters or buffers. Before this fix a
768 x 768 ``TTNNLinear.post_init()`` added 524,288 bytes to the synchronized
DRAM allocator while ``_parameter_devices()`` still reported only the host
checkpoint bytes.

The fix gives modules a second hook, ``iter_runtime_device_allocations()``,
that names its backend and that backend's own device ordinal. ``DeviceBudget``
carries the backend too, so a TTNN chip 1 is never read as CUDA device 1 or as
the host (-1). A backend that owns storage but cannot measure it is reported as
unmeasured coverage, never as zero.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from gptqmodel.models.base import BaseQModel, _parameter_devices, _runtime_device_allocations
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.ttnn import TTNNLinear
from gptqmodel.nn_modules.qlinear.ttnn_blockfloat import TTNNBlockFloat8Linear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.resource_budget import (
    MEASURED,
    PARTIAL,
    DeviceAllocation,
    DeviceBudget,
    ResourceBudget,
    RuntimeAllocationUnavailable,
)


def _budget_bytes(module) -> int:
    return BaseQModel.budget(types.SimpleNamespace(model=module)).weights_bytes


# ---------------------------------------------------------------------------
# Device identity: backend namespaces never collide.
# ---------------------------------------------------------------------------

def test_device_budget_infers_the_torch_backends():
    assert DeviceBudget(index=0).backend == "cuda"
    assert DeviceBudget(index=3).backend == "cuda"
    assert DeviceBudget(index=-1).backend == "host"


def test_device_budget_keeps_an_explicit_native_backend():
    device = DeviceBudget(index=1, weights_bytes=4096, backend="ttnn")

    assert device.backend == "ttnn"
    # The TTNN ordinal is not a CUDA index: the two sets stay separate.
    assert DeviceBudget(index=1, backend="cuda") != device


def _mixed_budget(**kwargs) -> ResourceBudget:
    return ResourceBudget(
        devices=(
            DeviceBudget(index=-1, weights_bytes=50),
            DeviceBudget(index=0, free_bytes=100, total_bytes=200, weights_bytes=100),
            DeviceBudget(index=0, weights_bytes=1000, backend="ttnn"),
            DeviceBudget(index=1, weights_bytes=1000, backend="ttnn"),
        ),
        **kwargs,
    )


def test_cuda_devices_do_not_include_native_chips_or_the_host():
    budget = _mixed_budget()

    assert [device.index for device in budget.cuda_devices] == [0]
    assert budget.cuda_devices[0].backend == "cuda"
    assert budget.host_weights_bytes == 50
    assert budget.is_sharded is False


def test_accelerator_devices_are_reported_separately():
    budget = _mixed_budget()

    assert [device.index for device in budget.accelerator_devices] == [0, 1]
    assert {device.backend for device in budget.accelerator_devices} == {"ttnn"}
    # The total is every resident weight, including the native chips.
    assert budget.weights_bytes == 50 + 100 + 1000 + 1000


def test_a_native_chip_does_not_make_a_single_cuda_device_look_sharded():
    budget = _mixed_budget()

    assert budget.is_sharded is False
    # Only the CUDA device contributes to the KV budget; the 10% reserve is
    # held back from its 100 measured free bytes.
    assert budget.kv_budget_bytes == 90


def test_describe_names_the_native_backend_and_its_chip():
    lines = _mixed_budget().describe()
    weights = next(line for line in lines if line.startswith("Budget: weights"))

    assert "ttnn device 0" in weights
    assert "ttnn device 1" in weights
    assert "host 50" in weights
    assert "(measured)" in weights


def test_unmeasured_backend_is_partial_coverage_not_zero():
    budget = _mixed_budget(unmeasured_backends=("ttnn",))

    assert budget.weights_coverage == PARTIAL
    assert budget.weights_coverage != MEASURED
    assert budget.unmeasured_backends == ("ttnn",)
    weights = next(line for line in budget.describe() if line.startswith("Budget: weights"))
    assert "partial: ttnn allocations unmeasured" in weights


def test_a_complete_report_says_measured():
    assert _mixed_budget().weights_coverage == MEASURED
    assert _mixed_budget().unmeasured_backends == ()


# ---------------------------------------------------------------------------
# The module hook merges with the registered tensors.
# ---------------------------------------------------------------------------

class _BackendModule(nn.Module):
    def __init__(self, allocations=(), error=None):
        super().__init__()
        self.register_buffer("packed", torch.zeros(8, dtype=torch.int32))
        self._allocations = tuple(allocations)
        self._error = error

    def iter_runtime_device_allocations(self):
        if self._error is not None:
            raise self._error
        yield from self._allocations


def test_a_native_allocation_is_added_to_the_measured_total():
    module = _BackendModule([DeviceAllocation(backend="ttnn", index=1, weights_bytes=4096)])

    # Registered storage is still host bytes; the chip entry is its own device.
    assert _parameter_devices(module) == {-1: 32}
    budget = BaseQModel.budget(types.SimpleNamespace(model=module))

    assert budget.host_weights_bytes == 32
    assert budget.accelerator_devices == (
        DeviceBudget(index=1, weights_bytes=4096, backend="ttnn"),
    )
    assert budget.weights_bytes == 32 + 4096
    assert budget.weights_coverage == MEASURED


def test_two_layers_on_one_chip_are_summed():
    parent = nn.Module()
    parent.first = _BackendModule([DeviceAllocation("ttnn", 0, 100)])
    parent.second = _BackendModule([DeviceAllocation("ttnn", 0, 250)])

    budget = BaseQModel.budget(types.SimpleNamespace(model=parent))

    assert len(budget.accelerator_devices) == 1
    assert budget.accelerator_devices[0].weights_bytes == 350
    assert budget.weights_bytes == 64 + 350


def test_two_chips_stay_two_entries():
    module = _BackendModule(
        [DeviceAllocation("ttnn", 0, 100), DeviceAllocation("ttnn", 3, 100)]
    )

    budget = BaseQModel.budget(types.SimpleNamespace(model=module))

    assert [device.index for device in budget.accelerator_devices] == [0, 3]
    assert budget.weights_bytes == 32 + 200


def test_an_unmeasurable_backend_is_unknown_coverage():
    module = _BackendModule(error=RuntimeAllocationUnavailable("ttnn"))

    budget = BaseQModel.budget(types.SimpleNamespace(model=module))

    # The registered bytes are still measured; the unreadable allocator is not
    # silently counted as empty.
    assert budget.weights_bytes == 32
    assert budget.unmeasured_backends == ("ttnn",)
    assert budget.weights_coverage == PARTIAL


def test_an_unexpected_hook_failure_is_unknown_coverage():
    module = _BackendModule(error=RuntimeError("allocator handle closed"))

    budget = BaseQModel.budget(types.SimpleNamespace(model=module))

    assert budget.unmeasured_backends == ("unknown",)
    assert budget.weights_coverage == PARTIAL


def test_a_malformed_entry_is_unknown_coverage():
    module = _BackendModule(
        [
            DeviceAllocation("ttnn", 0, 100),
            types.SimpleNamespace(backend="ttnn", index=-1, weights_bytes=10),
        ]
    )

    budget = BaseQModel.budget(types.SimpleNamespace(model=module))

    assert budget.weights_bytes == 32 + 100
    assert budget.unmeasured_backends == ("ttnn",)


def test_the_allocations_are_grouped_by_backend_and_index():
    parent = nn.Module()
    parent.a = _BackendModule([DeviceAllocation("ttnn", 0, 10)])
    parent.b = _BackendModule([DeviceAllocation("ttnn", 0, 5)])

    allocations, unmeasured = _runtime_device_allocations(parent)

    assert allocations == {("ttnn", 0): 15}
    assert unmeasured == ()


def test_the_default_hook_publishes_nothing():
    assert tuple(BaseQuantLinear.iter_runtime_device_allocations(object())) == ()


# ---------------------------------------------------------------------------
# TTNN kernels publish their per-chip device storage.
# ---------------------------------------------------------------------------

def _ttnn_layer() -> TTNNLinear:
    return TTNNLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=768, out_features=768, bias=True, format=FORMAT.GPTQ_V2,
    )


def test_an_unbuilt_ttnn_layer_owns_no_device_storage():
    layer = _ttnn_layer()

    assert tuple(layer.iter_runtime_device_allocations()) == ()
    # The checkpoint staging buffers are host bytes, not device bytes.
    assert set(_parameter_devices(layer)) == {-1}


def test_a_built_ttnn_layer_reports_every_chip():
    layer = _ttnn_layer()
    layer._ttnn_layer = object()
    layer._ttnn_device_ids = (1, 3)
    layer._ttnn_device_bytes = 524_288

    published = tuple(layer.iter_runtime_device_allocations())

    assert published == (
        DeviceAllocation("ttnn", 1, 524_288),
        DeviceAllocation("ttnn", 3, 524_288),
    )


def test_a_built_ttnn_layer_without_a_measurement_is_unmeasured():
    layer = _ttnn_layer()
    layer._ttnn_layer = object()
    layer._ttnn_device_ids = (0,)
    layer._ttnn_device_bytes = None

    with pytest.raises(RuntimeAllocationUnavailable, match="ttnn"):
        tuple(layer.iter_runtime_device_allocations())


def test_clearing_the_cache_drops_the_device_storage(monkeypatch):
    from gptqmodel.nn_modules.qlinear import ttnn as ttnn_module

    monkeypatch.setattr(ttnn_module, "import_ttnn", lambda: types.SimpleNamespace(
        deallocate=lambda tensor: None,
    ))
    layer = _ttnn_layer()
    layer._ttnn_layer = types.SimpleNamespace(
        weight_words=None, zero_words=None, scales=None, g_idx=None, bias=None,
    )
    layer._ttnn_device_ids = (0,)
    layer._ttnn_device_bytes = 4096

    layer.clear_weight_cache()

    assert tuple(layer.iter_runtime_device_allocations()) == ()
    assert layer._ttnn_device_ids == ()
    assert layer._ttnn_device_bytes is None


def _bf8_layer() -> TTNNBlockFloat8Linear:
    return TTNNBlockFloat8Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False, format=FORMAT.GPTQ_V2,
    )


def test_an_unbuilt_block_float_layer_owns_no_device_storage():
    assert tuple(_bf8_layer().iter_runtime_device_allocations()) == ()


def test_a_built_block_float_layer_reports_every_chip():
    layer = _bf8_layer()
    layer._ttnn_weight = object()
    layer._ttnn_device_ids = (2,)
    layer._ttnn_device_bytes = 8192

    assert tuple(layer.iter_runtime_device_allocations()) == (
        DeviceAllocation("ttnn", 2, 8192),
    )


def test_clearing_the_block_float_cache_drops_the_device_storage():
    layer = _bf8_layer()
    layer._ttnn_weight = object()
    layer._ttnn_device_ids = (2,)
    layer._ttnn_device_bytes = 8192

    layer.clear_weight_cache()

    assert tuple(layer.iter_runtime_device_allocations()) == ()
    assert layer._ttnn_device_bytes is None


# ---------------------------------------------------------------------------
# Runtime helpers: chip identity and allocator measurement.
# ---------------------------------------------------------------------------

def test_mesh_device_ids_reads_the_requested_single_chip():
    from gptqmodel.utils.ttnn_runtime import mesh_device_ids

    mesh = types.SimpleNamespace(num_devices=1, device_id=3, device=None)

    assert mesh_device_ids(mesh) == (3,)


def test_mesh_device_ids_reads_the_physical_mesh_chips():
    from gptqmodel.utils.ttnn_runtime import mesh_device_ids

    device = types.SimpleNamespace(get_device_ids=lambda: [0, 3])
    mesh = types.SimpleNamespace(num_devices=2, device_id=0, device=device)

    assert mesh_device_ids(mesh) == (0, 3)


def test_dram_allocated_bytes_uses_the_per_chip_view(monkeypatch):
    from gptqmodel.utils import ttnn_runtime

    calls = []

    class _View:
        num_banks = 8
        total_bytes_allocated_per_bank = 65_536

    class _Ttnn:
        BufferType = types.SimpleNamespace(DRAM="dram")

        @staticmethod
        def synchronize_device(device):
            calls.append(device)

        @staticmethod
        def get_memory_view(device, buffer_type):
            assert buffer_type == "dram"
            return _View()

    monkeypatch.setattr(ttnn_runtime, "import_ttnn", lambda: _Ttnn())

    assert ttnn_runtime.dram_allocated_bytes(object()) == 8 * 65_536
    assert len(calls) == 1
