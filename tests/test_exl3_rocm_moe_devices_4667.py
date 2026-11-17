# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EXL3's routed-expert owner devices must accept ``DEVICE.ROCM`` (#4667).

``DEVICE.ROCM`` has the value ``"rocm"`` even though PyTorch exposes ROCm
through the ``cuda`` namespace, so ``torch.device(DEVICE.ROCM)`` raises
``RuntimeError: Expected one of cpu, cuda, ... device type at start of device
string: rocm``. EXL3's routed-expert path built its owner-device list with a
bare ``torch.device(...)``, which crashed EXL3 on every MoE model on ROCm before
calibration. Dense models never reach that branch, which is why they worked.

This is a CPU-only test: it resolves the device values without touching a
device, which is exactly where the crash happened.
"""

import pytest
import torch

from gptqmodel.looper.exllamav3_processor import routed_expert_owner_devices
from gptqmodel.models._const import DEVICE


pytestmark = [pytest.mark.cpu]


def _qcfg(**kwargs):
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)


def test_rocm_device_enum_resolves_to_the_runtime_device():
    assert routed_expert_owner_devices(
        _qcfg(device=DEVICE.ROCM, moe_vram_strategy_devices=None)
    ) == [torch.device("cuda")]


def test_rocm_pool_entries_resolve_without_raising():
    devices = routed_expert_owner_devices(
        _qcfg(device=DEVICE.ROCM, moe_vram_strategy_devices=[DEVICE.ROCM, "cpu"])
    )

    assert devices == [torch.device("cuda"), torch.device("cpu")]


def test_no_configured_device_yields_an_empty_pool():
    assert routed_expert_owner_devices(
        _qcfg(device=None, moe_vram_strategy_devices=None)
    ) == []


def test_the_raw_enum_is_still_rejected_by_torch_device():
    """Pin the reason the helper exists, so a revert cannot pass silently."""

    with pytest.raises(RuntimeError, match="device type at start of device string"):
        torch.device(DEVICE.ROCM)
