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
