# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A cold trace must not pin the CPU int4pack kernels to the dense path (#5515).

The one-time int4pack transform rewrites the packed buffers in place, so it
only ran in an eager forward. ``optimize()`` (and any ``torch.compile`` made
before the first eager forward) therefore traced ``linear_mode=None`` and kept
dequantizing every weight on every token for the life of the process: 29x
slower generation, with correct output, so nothing signalled it.

Both entry points now install the runtime layout before the graph is traced.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.linear_mode import LinearMode
from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear


KERNELS = [TorchFusedLinear, TorchAtenLinear, TorchFusedAwqLinear, TorchAtenAwqLinear]

BITS = 4
GROUP_SIZE = 32
IN_FEATURES = 32
OUT_FEATURES = 16
PACK_FACTOR = 32 // BITS


def _module(cls):
    awq = "Awq" in cls.__name__
    qweight_shape = (
        (IN_FEATURES, OUT_FEATURES // PACK_FACTOR)
        if awq
        else (IN_FEATURES // PACK_FACTOR, OUT_FEATURES)
    )
    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    module.qweight = torch.full(qweight_shape, 0x77777777, dtype=torch.int32)
    module.qzeros = torch.full(
        (IN_FEATURES // GROUP_SIZE, OUT_FEATURES // PACK_FACTOR), 0x33333333, dtype=torch.int32
    )
    module.scales = torch.full(
        (IN_FEATURES // GROUP_SIZE, OUT_FEATURES), 0.5, dtype=torch.bfloat16
    )
    module.post_init()
    module.eval()
    return module


def _activation():
    return torch.randn(2, IN_FEATURES, dtype=torch.bfloat16)


@pytest.mark.parametrize("cls", KERNELS, ids=lambda cls: cls.__name__)
def test_a_cold_compiled_call_installs_the_inference_layout(cls):
    """Tracing before any eager forward must still reach the fused path."""

    module = _module(cls)
    compiled = torch.compile(module, backend="eager")

    with torch.no_grad():
        compiled(_activation())

    assert module.linear_mode == LinearMode.INFERENCE


@pytest.mark.parametrize("cls", KERNELS, ids=lambda cls: cls.__name__)
def test_a_cold_compiled_call_matches_the_eager_result(cls):
    module = _module(cls)
    reference_module = _module(cls)
    activation = _activation()

    with torch.no_grad():
        expected = reference_module(activation).clone()
        actual = torch.compile(module, backend="eager")(activation)

    assert module.linear_mode == LinearMode.INFERENCE
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("cls", KERNELS, ids=lambda cls: cls.__name__)
def test_optimize_installs_the_layout_before_compiling(cls):
    """``optimize()`` promises a warm-up; it must happen before the trace."""

    module = _module(cls)
    # A quantized model keeps the model dtype on its unquantized parameters.
    holder = nn.Module()
    holder.add_module("layer", module)
    holder.add_module("norm", nn.LayerNorm(IN_FEATURES, dtype=torch.bfloat16))
    stub = SimpleNamespace(
        quantized=True,
        compiled=False,
        model=holder,
    )

    BaseQModel.optimize(stub, backend="eager")

    assert module.linear_mode == LinearMode.INFERENCE
    assert stub.compiled is True
