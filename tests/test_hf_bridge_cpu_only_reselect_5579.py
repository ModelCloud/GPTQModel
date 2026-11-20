# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5579: a CPU-only resolved map must not keep a CUDA-only kernel.

`SUPPORTS_CPU_OFFLOAD` means "safe when Accelerate runs the layer on a GPU
execution device", not "safe to execute on CPU". On ROCm, `device_map="auto"`
plus a `max_memory` that leaves the model on CPU resolves to an all-CPU map;
AUTO had picked `AwqGEMMTritonLinear`, the reselect filter skipped it because of
that flag, and the first forward handed CPU tensors to Triton. These tests pin
the filter: with no GPU entry in the resolved map, a kernel that does not
declare CPU support is rebuilt.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models._const import DEVICE
from gptqmodel.quantization import FORMAT, AWQConfig
from gptqmodel.utils import hf_load, importer
from gptqmodel.utils import model as model_utils


class _TritonLike(nn.Module):
    """`AwqGEMMTritonLinear` / `TritonV2Linear`: GPU-only, CPU-offload safe."""

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_CPU_OFFLOAD = True


class _CpuCapable(nn.Module):
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_CPU_OFFLOAD = True


class _AllDevices(nn.Module):
    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_CPU_OFFLOAD = True


class _MarlinLike(nn.Module):
    """Needs resident CUDA tensors; the pre-#5579 filter already caught it."""

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_CPU_OFFLOAD = False


@pytest.fixture()
def context():
    return SimpleNamespace(
        backend=None,
        quantize_config=AWQConfig(bits=4, group_size=128, format=FORMAT.GEMM),
        dtype=torch.float16,
        lm_head_name="lm_head",
    )


def _reselect(monkeypatch, context, modules, device_map):
    rebuilt = {}

    def _find_modules(model, layers=None):
        return dict(modules)

    def _create_quant_layer(**kwargs):
        rebuilt.update(kwargs["quant_result"])

    monkeypatch.setattr(model_utils, "find_modules", _find_modules)
    monkeypatch.setattr(model_utils, "create_quant_layer", _create_quant_layer)
    monkeypatch.setattr(importer, "select_quant_linear", lambda **kwargs: [nn.Linear])
    return (
        hf_load.hf_gptqmodel_reselect_offloaded_kernels(
            nn.Linear(2, 2), context, device_map
        ),
        rebuilt,
    )


def test_execution_device_predicate():
    assert not hf_load._device_map_has_execution_device({})
    assert not hf_load._device_map_has_execution_device({"": "cpu"})
    assert not hf_load._device_map_has_execution_device(
        {"a": "cpu", "b": "disk", "c": torch.device("cpu")}
    )
    assert hf_load._device_map_has_execution_device({"": 0})
    assert hf_load._device_map_has_execution_device({"a": 0, "b": "cpu"})
    assert hf_load._device_map_has_execution_device({"": "cuda:0"})
    assert hf_load._device_map_has_execution_device({"": torch.device("cuda", 0)})


def test_cpu_only_map_rebuilds_a_kernel_that_cannot_run_on_cpu(monkeypatch, context):
    name = "model.layers.0.self_attn.q_proj"
    changed, rebuilt = _reselect(
        monkeypatch, context, {name: _TritonLike()}, {"": "cpu"}
    )
    assert changed
    assert rebuilt == {name: True}


def test_torch_device_cpu_map_is_also_cpu_only(monkeypatch, context):
    name = "model.layers.0.self_attn.q_proj"
    changed, rebuilt = _reselect(
        monkeypatch, context, {name: _TritonLike()}, {"": torch.device("cpu")}
    )
    assert changed
    assert rebuilt == {name: True}


def test_disk_only_map_rebuilds_a_kernel_that_cannot_run_on_cpu(monkeypatch, context):
    name = "model.layers.0.self_attn.q_proj"
    changed, rebuilt = _reselect(monkeypatch, context, {name: _TritonLike()}, {"": "disk"})
    assert changed
    assert rebuilt == {name: True}


def test_gpu_execution_device_keeps_the_cpu_offload_kernel(monkeypatch, context):
    """A spilled layer with a GPU elsewhere still runs through the Triton hook."""

    name = "model.layers.1.self_attn.q_proj"
    changed, rebuilt = _reselect(
        monkeypatch,
        context,
        {name: _TritonLike()},
        {"model.layers.0": 0, "model.layers.1": "cpu"},
    )
    assert not changed
    assert rebuilt == {}


def test_cpu_only_map_keeps_kernels_that_declare_cpu(monkeypatch, context):
    modules = {
        "model.layers.0.self_attn.q_proj": _CpuCapable(),
        "model.layers.0.self_attn.k_proj": _AllDevices(),
    }
    changed, rebuilt = _reselect(monkeypatch, context, modules, {"": "cpu"})
    assert not changed
    assert rebuilt == {}


def test_cpu_only_map_still_rebuilds_a_kernel_without_the_declaration(
    monkeypatch, context
):
    name = "model.layers.0.self_attn.q_proj"
    changed, rebuilt = _reselect(
        monkeypatch, context, {name: _MarlinLike()}, {"": "cpu"}
    )
    assert changed
    assert rebuilt == {name: True}


def test_resident_map_rebuilds_nothing(monkeypatch, context):
    name = "model.layers.0.self_attn.q_proj"
    changed, rebuilt = _reselect(monkeypatch, context, {name: _TritonLike()}, {"": 0})
    assert not changed
    assert rebuilt == {}
