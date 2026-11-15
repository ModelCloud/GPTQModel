# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Placement contract for the default layer-wise device map.

``device=("cuda:1", "cuda:2")`` means *those two GPUs*. The map builder used to
ignore the sequence and span every visible accelerator, so asking for a subset
silently used all of them. These tests pin the helper contract without needing
accelerators: the visible-device count is monkeypatched.
"""

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.models.loader import (
    _assign_greedy_device_ids,
    _device_map_uses_cpu,
    _explicit_layerwise_devices,
    _explicit_requested_devices,
    _indexed_device_request,
    _layerwise_device_count,
    _layerwise_device_strings,
    _module_tensor_bytes,
    _normalize_explicit_placement,
    _normalize_max_memory,
    _resolve_default_layerwise_device,
)
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND


def _fake_visible(monkeypatch, n):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_explicit_devices_accepts_concrete_ordinals():
    assert _explicit_layerwise_devices(("cuda:1", "cuda:2")) == ["cuda:1", "cuda:2"]
    assert _explicit_layerwise_devices(["cuda:0", "cuda:3"]) == ["cuda:0", "cuda:3"]
    assert _explicit_layerwise_devices(
        (torch.device("cuda:2"), torch.device("cuda:1"))
    ) == ["cuda:2", "cuda:1"]


def test_duplicate_explicit_devices_do_not_expand_to_every_gpu(monkeypatch):
    _fake_visible(monkeypatch, 4)
    requested = ("cuda:1", "cuda:1")

    assert _explicit_requested_devices(requested) == [torch.device("cuda:1")]
    assert _layerwise_device_count(requested) == 1
    assert _layerwise_device_strings(requested, 1) == ["cuda:1"]


def test_single_explicit_device_sequence_does_not_expand(monkeypatch):
    _fake_visible(monkeypatch, 4)
    requested = ("cuda:1",)

    assert _explicit_requested_devices(requested) == [torch.device("cuda:1")]
    assert _layerwise_device_count(requested) == 1
    assert _layerwise_device_strings(requested, 1) == ["cuda:1"]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ((0, 1), ["cuda:0", "cuda:1"]),
        (("cuda:1", 0), ["cuda:1", "cuda:0"]),
        ((0,), ["cuda:0"]),
    ],
)
def test_integer_ordinals_keep_the_requested_subset(monkeypatch, requested, expected):
    _fake_visible(monkeypatch, 4)
    monkeypatch.setattr(importer, "_current_accelerator_type", lambda: "cuda")
    monkeypatch.setattr(importer, "_device_capability", lambda _target: (8, 9))

    assert _explicit_requested_devices(requested) == [torch.device(target) for target in expected]
    assert _layerwise_device_count(requested) == len(expected)
    assert _layerwise_device_strings(requested, len(expected)) == expected


def test_selected_devices_validate_each_gpu_capability(monkeypatch):
    monkeypatch.setattr(
        importer,
        "_device_capability",
        lambda target: (8, 9) if target.index == 0 else (9, 0),
    )
    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        _explicit_requested_devices(("cuda:0", "cuda:1"))


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (("cuda:1",), ["cuda:1"]),
        (("cuda:1", "cuda:1"), ["cuda:1"]),
        ((0, 1), ["cuda:0", "cuda:1"]),
    ],
)
def test_default_loader_resolution_preserves_explicit_sequences(monkeypatch, requested, expected):
    _fake_visible(monkeypatch, 4)
    monkeypatch.setattr(importer, "_current_accelerator_type", lambda: "cuda")
    resolved = _resolve_default_layerwise_device(requested, BACKEND.AUTO)

    assert resolved == tuple(torch.device(target) for target in expected)
    assert _layerwise_device_count(resolved) == len(expected)
    assert _layerwise_device_strings(resolved, len(expected)) == expected


def test_explicit_devices_rejects_non_ordinal_selectors():
    """Abstract families and bare names keep the family-wide path."""
    assert _explicit_layerwise_devices("cuda") is None
    assert _explicit_layerwise_devices("cuda:1") is None
    assert _explicit_layerwise_devices(("cuda", "cuda")) is None
    assert _explicit_layerwise_devices(("cuda",)) is None


@pytest.mark.parametrize("requested", [("cuda:1", "cuda"), ("cuda:1", "cpu")])
def test_mixed_concrete_and_abstract_devices_fail_before_broadening(monkeypatch, requested):
    _fake_visible(monkeypatch, 4)
    with pytest.raises(ValueError, match="cannot mix concrete accelerator ordinals"):
        _explicit_requested_devices(requested)
    with pytest.raises(ValueError, match="cannot mix concrete accelerator ordinals"):
        _layerwise_device_count(requested)
    with pytest.raises(ValueError, match="cannot mix concrete accelerator ordinals"):
        _layerwise_device_strings(requested, 4)
    with pytest.raises(ValueError, match="cannot mix concrete accelerator ordinals"):
        _resolve_default_layerwise_device(requested, BACKEND.AUTO)


def test_normalize_max_memory_maps_ordinals_and_cpu():
    from accelerate.utils import convert_file_size_to_int

    budgets = _normalize_max_memory(
        {1: "4GiB", "cpu": "8GiB"}, DEVICE.CUDA
    )

    assert budgets == [
        ("cuda:1", convert_file_size_to_int("4GiB")),
        ("cpu", convert_file_size_to_int("8GiB")),
    ]


def test_normalize_max_memory_accepts_concrete_device_strings():
    assert _normalize_max_memory({"cuda:0": 1024}, DEVICE.CUDA) == [("cuda:0", 1024)]


def test_normalize_max_memory_rejects_an_unknown_device_key():
    with pytest.raises(ValueError, match="unsupported device key"):
        _normalize_max_memory({"tpu": "1GiB"}, DEVICE.CUDA)


def test_normalize_max_memory_rejects_non_positive_budgets():
    with pytest.raises(ValueError, match="greater than zero"):
        _normalize_max_memory({0: "0GiB"}, DEVICE.CUDA)


@pytest.mark.parametrize("device", ["cuda:0", torch.device("cuda:0")])
def test_indexed_device_is_a_full_model_map_without_a_memory_budget(device):
    assert _indexed_device_request(device) == "cuda:0"
    requested, explicit = _normalize_explicit_placement(device, None, None)

    assert requested == {"": "cuda:0"}
    assert explicit == {"": "cuda:0"}


@pytest.mark.parametrize("device", ["cuda:0", torch.device("cuda:0")])
def test_max_memory_keeps_the_indexed_device_for_the_budgeted_map(device):
    """#4068: the cap must survive when the caller also passes device="cuda:0"."""

    requested, explicit = _normalize_explicit_placement(
        device, None, {0: "300MiB", "cpu": "20GiB"}
    )

    assert requested is None
    assert explicit is None


def test_max_memory_and_an_explicit_device_map_still_conflict():
    with pytest.raises(ValueError, match="not both"):
        _normalize_explicit_placement(
            "cuda:0", {"": "cuda:0"}, {0: "300MiB", "cpu": "20GiB"}
        )


def test_accelerate_map_string_still_pairs_with_max_memory():
    requested, explicit = _normalize_explicit_placement(
        None, "auto", {0: "300MiB", "cpu": "20GiB"}
    )

    assert requested == "auto"
    assert explicit is None


def test_assign_greedy_device_ids_fills_devices_in_order():
    budgets = [10, 100]

    assert _assign_greedy_device_ids([4, 4, 4], budgets) == [0, 0, 1]
    assert budgets == [2, 96]


def test_assign_greedy_device_ids_reports_a_unit_that_does_not_fit():
    with pytest.raises(ValueError, match="cannot hold `model.layers.0`"):
        _assign_greedy_device_ids([11], [10], ["model.layers.0"])


def test_device_map_uses_cpu_detects_only_real_cpu_entries():
    assert _device_map_uses_cpu({"a": "cuda:0", "b": "cpu"})
    assert _device_map_uses_cpu({"a": torch.device("cpu")})
    assert not _device_map_uses_cpu({"a": "cuda:0", "b": 1})
    assert not _device_map_uses_cpu({"a": "disk"})
    assert not _device_map_uses_cpu(None)


def test_module_tensor_bytes_counts_parameters_and_buffers():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.zeros((4, 4), dtype=torch.float32))
    module.register_buffer("packed", torch.zeros((2,), dtype=torch.int32))

    assert _module_tensor_bytes(module) == 4 * 4 * 4 + 2 * 4


def test_requested_devices_are_concrete_torch_devices():
    requested = _explicit_requested_devices(("cuda:1", "cuda:2"))
    assert requested == [torch.device("cuda:1"), torch.device("cuda:2")]
    assert all(isinstance(target, torch.device) for target in requested)


def test_count_honours_requested_subset(monkeypatch):
    _fake_visible(monkeypatch, 4)
    assert _layerwise_device_count(("cuda:1", "cuda:2")) == 2
    assert _layerwise_device_count(["cuda:0", "cuda:1", "cuda:2", "cuda:3"]) == 4
    # abstract family still spans every visible device
    assert _layerwise_device_count("cuda") == 4


def test_strings_render_requested_subset(monkeypatch):
    _fake_visible(monkeypatch, 4)
    assert _layerwise_device_strings(("cuda:1", "cuda:2"), 2) == ["cuda:1", "cuda:2"]
    assert _layerwise_device_strings(["cuda:2", "cuda:3"], 2) == ["cuda:2", "cuda:3"]
    # family selector keeps the 0..n-1 rendering
    assert _layerwise_device_strings("cuda", 4) == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    ]
