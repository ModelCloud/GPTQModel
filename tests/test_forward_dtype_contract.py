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
    module = _Echo()
    module(torch.zeros(2, dtype=torch.float16))  # eager call records the lock
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    compiled(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(Exception, match="forward-dtype.md"):
        compiled(torch.zeros(2, dtype=torch.bfloat16))


def test_compiled_forward_does_not_recompile_for_the_lock():
    torch._dynamo.reset()
    frames = []

    def backend(graph_module, example_inputs):
        frames.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(_Echo(), backend=backend, fullgraph=True)
    for _ in range(3):
        compiled(torch.zeros(2, dtype=torch.float16))
    assert len(frames) == 1


class _Weighted(_Echo):
    def __init__(self):
        super().__init__()
        self.register_buffer("scales", torch.ones(4, dtype=torch.float16))
        self.register_buffer("qweight", torch.zeros(4, dtype=torch.int32))


@pytest.mark.parametrize(
    "cast",
    [
        lambda m: m.to(dtype=torch.bfloat16),
        lambda m: m.bfloat16(),
        lambda m: m.float(),
        lambda m: m.half() and m.to(torch.float32),
    ],
)
def test_weight_dtype_change_after_forward_raises_before_mutation(cast):
    module = _Weighted()
    module(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="frozen after the first forward"):
        cast(module)
    assert module.scales.dtype == torch.float16
    assert module.qweight.dtype == torch.int32


def test_weight_dtype_change_before_forward_is_allowed():
    module = _Weighted().to(dtype=torch.bfloat16)
    assert module.scales.dtype == torch.bfloat16
    assert module(torch.zeros(2, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_same_dtype_apply_after_forward_is_allowed():
    module = _Weighted()
    module(torch.zeros(2, dtype=torch.float16))
    module.to(dtype=torch.float16)
    module.to("cpu")
    module.half()
    assert module.scales.dtype == torch.float16


def test_parent_model_cast_after_forward_raises():
    parent = torch.nn.Sequential(_Weighted())
    parent(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="frozen after the first forward"):
        parent.to(torch.bfloat16)
