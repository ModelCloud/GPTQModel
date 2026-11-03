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

import pytest
import torch

from gptqmodel.nn_modules.qlinear.machete import MacheteLinear


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


def test_validate_device_reason_survives(monkeypatch):
    reason = "Machete kernel is Hopper-only (SM90); found compute capability 12.0"

    def reject(cls, device):
        raise NotImplementedError(reason)

    monkeypatch.setattr(MacheteLinear, "validate_device", classmethod(reject))
    ok, err = MacheteLinear.validate(**_validate_kwargs())
    assert ok is False
    assert isinstance(err, NotImplementedError)
    message = str(err)
    assert "does not support device" in message
    assert reason in message, message


def test_empty_reason_keeps_the_plain_message(monkeypatch):
    def reject(cls, device):
        raise NotImplementedError()

    monkeypatch.setattr(MacheteLinear, "validate_device", classmethod(reject))
    ok, err = MacheteLinear.validate(**_validate_kwargs())
    assert ok is False
    assert str(err).strip().endswith("cuda:0")


def test_real_hopper_only_message_reaches_the_caller():
    """On a non-Hopper GPU the concrete reason has to be visible end to end."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    capability = torch.cuda.get_device_capability()
    if capability[0] == 9:
        pytest.skip("Hopper supports Machete; this checks the rejection path")
    ok, err = MacheteLinear.validate(**_validate_kwargs())
    assert ok is False
    assert "Hopper" in str(err) or "compute capability" in str(err), str(err)
