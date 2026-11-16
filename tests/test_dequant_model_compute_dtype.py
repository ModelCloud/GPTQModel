# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``dequantize_model`` must use the model compute dtype, not ``scales.dtype``.

GPTQ checkpoints always store FP16 scales.  Dequantizing to the stored scale
dtype produced FP16 ``nn.Linear`` layers inside a BF16 model, and the first
forward failed with ``BFloat16 != Half``.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear import torch_aten_kernel as aten_module
from gptqmodel.nn_modules.qlinear import torch_fused as fused_module
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, dequantize_model
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear


pytestmark = [pytest.mark.cpu]

IN_FEATURES = 64
# TorchAtenLinear's CPU int4pack op requires out_features % 16 == 0.
OUT_FEATURES = 32
GROUP_SIZE = 32


def _aten_available() -> bool:
    return _has_local_int4pack_cpu_ops() and TorchAtenLinear.cached_validate_once()[0]


def _make_layer(kernel_cls):
    layer = kernel_cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    layer.post_init()
    layer.qweight.zero_()
    layer.qzeros.zero_()
    layer.scales.fill_(0.01)
    layer.g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    return layer


class _Model(nn.Module):
    def __init__(self, layer, dtype: torch.dtype):
        super().__init__()
        self.config = types.SimpleNamespace(quantization_config={"quant_method": "gptq"})
        self.embed = nn.Embedding(11, IN_FEATURES, dtype=dtype)
        self.proj = layer

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))


_IMPLEMENTATIONS = [
    pytest.param(TorchLinear, dequantize_model, id="torch"),
    pytest.param(TorchFusedLinear, fused_module.dequantize_model, id="torch_fused"),
    pytest.param(
        TorchAtenLinear,
        aten_module.dequantize_model,
        id="torch_aten",
        marks=pytest.mark.skipif(
            not _aten_available(), reason="ATen CPU int4pack ops are unavailable"
        ),
    ),
]


@pytest.mark.parametrize(("kernel_cls", "dequantize"), _IMPLEMENTATIONS)
def test_dequantize_bf16_model_produces_bf16_weights(kernel_cls, dequantize):
    layer = _make_layer(kernel_cls)
    # GPTQ checkpoints store fp16 scales regardless of the compute dtype.
    layer.scales = layer.scales.to(torch.float16)
    model = _Model(layer, torch.bfloat16)

    dequantize(model)

    assert isinstance(model.proj, nn.Linear)
    assert model.proj.weight.dtype is torch.bfloat16
    output = model(torch.tensor([[1, 2, 3]]))
    assert output.dtype is torch.bfloat16


@pytest.mark.parametrize(("kernel_cls", "dequantize"), _IMPLEMENTATIONS)
def test_dequantize_fp16_model_stays_fp16(kernel_cls, dequantize):
    layer = _make_layer(kernel_cls)
    layer.scales = layer.scales.to(torch.float16)
    model = _Model(layer, torch.float16)

    dequantize(model)

    assert isinstance(model.proj, nn.Linear)
    assert model.proj.weight.dtype is torch.float16
    output = model(torch.tensor([[1, 2, 3]]))
    assert output.dtype is torch.float16
