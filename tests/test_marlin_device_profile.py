# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The Marlin device profile has to describe the card it is running on.

Kernel selection reads this instead of naming an architecture, so a wrong probe
silently sends a shape down a path tuned for different hardware.
"""

import pytest
import torch

from gptqmodel.utils.marlin import MARLIN_DEVICE_PROFILE_FIELDS, marlin_device_profile


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="marlin needs a CUDA device",
)


@requires_cuda
def test_profile_matches_torch_device_properties():
    properties = torch.cuda.get_device_properties(0)
    profile = marlin_device_profile(0)

    assert set(profile) == set(MARLIN_DEVICE_PROFILE_FIELDS)
    assert profile["sms"] == properties.multi_processor_count
    assert profile["max_shared_mem"] > 0
    assert profile["max_shared_mem"] <= properties.shared_memory_per_block_optin
    assert (profile["capability_major"], profile["capability_minor"]) == (
        properties.major,
        properties.minor,
    )
    assert profile["l2_cache_size"] > 0
    assert profile["max_threads_per_sm"] > 0


@requires_cuda
def test_tensor_memory_flag_follows_the_capability():
    """tcgen05 tensor memory is datacenter Blackwell only (sm_100/103/110)."""
    profile = marlin_device_profile(0)
    expected = 1 if profile["capability_major"] in (10, 11) else 0
    assert profile["has_tensor_memory"] == expected


@requires_cuda
def test_profile_is_cached():
    first = marlin_device_profile(0)
    second = marlin_device_profile(0)
    assert first == second


def test_field_list_is_stable():
    assert MARLIN_DEVICE_PROFILE_FIELDS == (
        "sms",
        "max_shared_mem",
        "capability_major",
        "capability_minor",
        "l2_cache_size",
        "max_threads_per_sm",
        "has_tensor_memory",
    )

def test_explicit_device_zero_is_not_treated_as_unset(monkeypatch):
    """`0` is falsy, so an explicit index 0 must not fall back to the current device."""
    from gptqmodel.utils import marlin as marlin_mod

    calls = []
    monkeypatch.setattr(marlin_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(marlin_mod.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(marlin_mod.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        marlin_mod,
        "_marlin_resolve_op",
        lambda **_kwargs: lambda index: calls.append(index)
        or [0] * len(marlin_mod.MARLIN_DEVICE_PROFILE_FIELDS),
    )

    marlin_mod.marlin_device_profile(0)
    assert calls == [0], f"device 0 was resolved to {calls}"


def test_index_less_cuda_means_the_current_device(monkeypatch):
    from gptqmodel.utils import marlin as marlin_mod

    calls = []
    monkeypatch.setattr(marlin_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(marlin_mod.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        marlin_mod,
        "_marlin_resolve_op",
        lambda **_kwargs: lambda index: calls.append(index)
        or [0] * len(marlin_mod.MARLIN_DEVICE_PROFILE_FIELDS),
    )

    marlin_mod.marlin_device_profile("cuda")
    assert calls == [2]


def test_non_cuda_device_is_rejected(monkeypatch):
    from gptqmodel.utils import marlin as marlin_mod

    monkeypatch.setattr(marlin_mod.torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        marlin_mod.marlin_device_profile("cpu")


@pytest.mark.parametrize("device", ["cuda:256", "cuda:999", torch.device("cuda:1")])
def test_explicit_device_index_is_validated_before_profile(monkeypatch, device):
    from gptqmodel.utils import marlin as marlin_mod

    monkeypatch.setattr(marlin_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(marlin_mod.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        marlin_mod,
        "_marlin_resolve_op",
        lambda **_kwargs: pytest.fail("invalid device index reached the extension"),
    )

    with pytest.raises(ValueError, match="out of range"):
        marlin_mod.marlin_device_profile(device)


@requires_cuda
def test_out_of_range_index_raises_with_the_number_the_caller_passed():
    """999 would wrap to -25 inside torch's int8 device index; report 999."""
    count = torch.cuda.device_count()
    with pytest.raises(ValueError, match=rf"device index {count + 32} is out of range"):
        marlin_device_profile(count + 32)


@requires_cuda
def test_negative_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        marlin_device_profile(-1)


@requires_cuda
def test_extension_rejects_an_out_of_range_index_directly():
    """Defence in depth: the C++ side must not index the cache unchecked."""
    op = torch.ops.gptqmodel_marlin_fp16.marlin_device_profile
    with pytest.raises(RuntimeError, match="out of range"):
        op(torch.cuda.device_count() + 4096)
    with pytest.raises(RuntimeError, match="out of range"):
        op(1 << 32)
