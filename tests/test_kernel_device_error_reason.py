# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A kernel that rejects a device must keep its own explanation.

Several kernels raise from ``validate_device`` for reasons that have nothing to
do with the device family: Machete is Hopper-only, Swordfish only has images for
sm100/sm103/sm110, Marlin needs compute capability 7.5, the ROCm variants refuse
outright. The shared validator used to replace all of that with
"does not support device: cuda:0", which sends the user looking at the wrong
thing.
"""

import sys

import pytest
import torch

import gptqmodel.nn_modules.qlinear.machete as machete_module
from gptqmodel.models._const import DEVICE, PLATFORM
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.utils.rocm import IS_ROCM


def _validate_kwargs():
    return {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "in_features": 128,
        "out_features": 128,
        "pack_dtype": torch.int32,
        "device": torch.device("cuda:0"),
    }


def _reject(monkeypatch, reason):
    """Make validate_device reject, and make the platform gate accept anything.

    Machete declares Linux-only support, so on macOS and Windows the platform
    check returns before validate_device is ever consulted. These tests are
    about the device-message path, not about which OS the kernel ships for.
    """

    def reject(cls, device):
        raise NotImplementedError(reason)

    monkeypatch.setattr(MacheteLinear, "SUPPORTS_PLATFORM", [PLATFORM.ALL])
    monkeypatch.setattr(MacheteLinear, "validate_device", classmethod(reject))


def test_validate_device_reason_survives(monkeypatch):
    reason = "Machete kernel is Hopper-only (SM90); found compute capability 12.0"
    _reject(monkeypatch, reason)
    ok, err = MacheteLinear.validate(**_validate_kwargs())
    assert ok is False
    assert isinstance(err, NotImplementedError)
    message = str(err)
    assert "does not support device" in message
    assert reason in message, message


def test_empty_reason_keeps_the_plain_message(monkeypatch):
    # Assert the plain message; a HIP host appends a ROCm hint to it.
    monkeypatch.setattr(torch.version, "hip", None)
    _reject(monkeypatch, "")
    ok, err = MacheteLinear.validate(**_validate_kwargs())
    assert ok is False
    assert str(err).strip().endswith("cuda:0")


def test_device_family_reason_is_not_repeated(monkeypatch):
    """The base reason already names the mismatch; do not append it verbatim.

    With only the stock ``validate_device`` in place, an unsupported family
    raises `only supports [...]`, which the wrapping sentence restates. Keeping
    the outer message alone avoids:
    `does not support device: cpu (only supports ... actual device = cpu)`.
    """
    monkeypatch.setattr(MacheteLinear, "SUPPORTS_PLATFORM", [PLATFORM.ALL])
    monkeypatch.setattr(
        MacheteLinear,
        "validate_device",
        classmethod(BaseQuantLinear.validate_device.__func__),
    )
    ok, err = MacheteLinear._validate_shared(
        pack_dtype=torch.int32, device=torch.device("cpu"),
    )
    assert ok is False
    message = str(err)
    assert message.count("does not support device") == 1
    assert "actual device" not in message, message
    assert message.endswith("cpu"), message


def test_rocm_rejection_reason_survives(monkeypatch):
    """The ROCm refusal is the same code path with a different reason.

    Patched rather than detected so this covers the branch on any platform,
    including the CUDA machines that never enter it.  The support-family gate
    has to accept both families too: on a HIP host ``cuda:0`` normalizes to
    ``DEVICE.ROCM`` and the CUDA-only family declaration rejects before
    ``MacheteLinear.validate_device`` can raise its own reason.
    """
    monkeypatch.setattr(MacheteLinear, "SUPPORTS_PLATFORM", [PLATFORM.ALL])
    monkeypatch.setattr(MacheteLinear, "SUPPORTS_DEVICES", [DEVICE.CUDA, DEVICE.ROCM])
    monkeypatch.setattr(machete_module, "IS_ROCM", True)
    ok, err = MacheteLinear.validate(**_validate_kwargs())
    assert ok is False
    assert "not supported on ROCm" in str(err), str(err)


def test_real_hopper_only_message_reaches_the_caller():
    """On a non-Hopper CUDA GPU the concrete reason has to be visible end to end."""
    if sys.platform != "linux":
        pytest.skip("Machete only supports Linux")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if IS_ROCM:
        pytest.skip("Machete rejects ROCm before the Hopper check; see the mocked test")
    capability = torch.cuda.get_device_capability()
    if capability[0] == 9:
        pytest.skip("Hopper supports Machete; this checks the rejection path")
    ok, err = MacheteLinear.validate(**_validate_kwargs())
    assert ok is False
    assert "Hopper" in str(err) or "compute capability" in str(err), str(err)
