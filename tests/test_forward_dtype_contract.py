# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Forward dtype contract: contracts/kernel/forward-dtype.md."""

import copy

import pytest
import torch

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


class _Echo(BaseQuantLinear):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.name = "echo"

    def forward(self, x):
        return x


class _EchoChild(_Echo):
    def forward(self, x):
        return super().forward(x)


def test_every_kernel_forward_is_locked():
    assert TorchLinear.forward._forward_dtype_locked
    assert _EchoChild.forward._forward_dtype_locked


@pytest.mark.parametrize("cls", [_Echo, _EchoChild])
def test_same_dtype_repeats(cls):
    module = cls()
    for _ in range(3):
        assert module(torch.zeros(2, dtype=torch.float16)).dtype == torch.float16
    assert module._forward_dtype_lock == torch.float16


@pytest.mark.parametrize("cls", [_Echo, _EchoChild])
def test_dtype_switch_raises(cls):
    module = cls()
    module(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="forward-dtype.md"):
        module(torch.zeros(2, dtype=torch.bfloat16))


def test_separate_instances_may_differ():
    _Echo()(torch.zeros(2, dtype=torch.float16))
    assert _Echo()(torch.zeros(2, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_lock_survives_deepcopy():
    module = _Echo()
    module(torch.zeros(2, dtype=torch.bfloat16))
    clone = copy.deepcopy(module)
    with pytest.raises(RuntimeError):
        clone(torch.zeros(2, dtype=torch.float16))


def test_compiled_dtype_switch_raises():
    compiled = torch.compile(_Echo(), backend="eager", fullgraph=True)
    compiled(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(Exception, match="forward-dtype.md"):
        compiled(torch.zeros(2, dtype=torch.bfloat16))
