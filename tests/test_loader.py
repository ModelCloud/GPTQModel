# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.models import loader
from gptqmodel.models._const import DEVICE
from gptqmodel.models.loader import _should_print_module_tree


def test_loader_module_tree_print_is_opt_in(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_PRINT_MODULE_TREE", raising=False)
    assert _should_print_module_tree() is False

    monkeypatch.setenv("GPTQMODEL_PRINT_MODULE_TREE", "1")
    assert _should_print_module_tree() is True

    monkeypatch.setenv("GPTQMODEL_PRINT_MODULE_TREE", "off")
    assert _should_print_module_tree() is False


def test_layerwise_rocm_uses_cuda_runtime_for_all_visible_devices(monkeypatch):
    monkeypatch.setattr(loader.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: True)

    assert loader._layerwise_device_count(DEVICE.ROCM) == 3
    assert loader._layerwise_device_strings(DEVICE.ROCM, 3) == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]


def test_layerwise_rocm_mock_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr("gptqmodel.utils.importer.IS_ROCM", True)
    monkeypatch.setattr(loader.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: True)

    num_gpus = loader._layerwise_device_count(DEVICE.ROCM)
    assert loader._layerwise_device_strings(torch.device("cuda"), num_gpus) == [
        "cuda:0",
        "cuda:1",
    ]


def test_device_map_string_device_is_hf_shorthand():
    assert loader._normalize_requested_device_map(None, "cpu") == {"": "cpu"}
    assert loader._normalize_requested_device_map(None, "cuda:1") == {"": "cuda:1"}
    assert loader._normalize_requested_device_map(None, "auto") == "auto"
    assert loader._normalize_requested_device_map(None, "balanced_low_0") == "balanced_low_0"


@pytest.mark.parametrize("device_map", ("not-a-device", "disk", "meta"))
def test_device_map_string_rejects_invalid_or_offload_targets(device_map):
    with pytest.raises(ValueError, match="device_map"):
        loader._normalize_requested_device_map(None, device_map)


@pytest.mark.parametrize(
    ("device", "device_map"),
    (
        ("cpu", {"": "cuda:0"}),
        ("cuda:0", {"": "cpu"}),
    ),
)
def test_device_and_device_map_conflict_raises(device, device_map):
    with pytest.raises(ValueError, match="conflicts"):
        loader._normalize_requested_device_map(device, device_map)


@pytest.mark.parametrize(
    ("device", "device_map"),
    (
        ("cpu", {"": "cpu"}),
        ("cuda:0", {"": "cuda:0"}),
    ),
)
def test_device_and_device_map_agreement_is_accepted(device, device_map):
    assert loader._normalize_requested_device_map(device, device_map) == device_map
