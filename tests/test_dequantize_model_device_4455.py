# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4455: ``dequantize_model`` keeps each layer on its device.

The in-memory dequantizers built the dense ``nn.Linear`` with
``.to("cpu", dtype)``, so a CUDA-loaded model got CPU dense weights and the
next forward failed with a device mismatch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear import torch_aten_kernel as aten_module
from gptqmodel.nn_modules.qlinear import torch_fused as fused_module
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, dequantize_model
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from tests.test_dequantize_model_compute_dtype_4370 import _aten_available, _Holder, _make_layer


def _run(kernel_cls, dequantize, device, *, bias):
    layer = _make_layer(kernel_cls)
    if bias:
        layer.bias = torch.full((layer.out_features,), 0.5, dtype=torch.float16)
    model = _Holder(layer, torch.float16).to(device)
    x = torch.randn(3, layer.in_features, dtype=torch.float16, device=device)
    expected = model(x)

    dequantize(model)

    assert type(model.proj) is nn.Linear
    assert model.proj.weight.device == torch.device(device)
    if bias:
        assert model.proj.bias.device == torch.device(device)
    else:
        assert model.proj.bias is None
    out = model(x)
    assert out.device == torch.device(device)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


_CPU_BACKENDS = [
    pytest.param(TorchLinear, dequantize_model, id="torch"),
    pytest.param(TorchFusedLinear, fused_module.dequantize_model, id="torch_fused"),
    pytest.param(
        TorchAtenLinear,
        aten_module.dequantize_model,
        id="torch_aten",
        marks=pytest.mark.skipif(not _aten_available(), reason="ATen CPU int4pack ops are unavailable"),
    ),
]


@pytest.mark.cpu
@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize(("kernel_cls", "dequantize"), _CPU_BACKENDS)
def test_dequantize_model_keeps_cpu_device(kernel_cls, dequantize, bias):
    _run(kernel_cls, dequantize, "cpu", bias=bias)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
def test_dequantize_model_keeps_cuda_device(bias):
    _run(TorchLinear, dequantize_model, "cuda:0", bias=bias)
