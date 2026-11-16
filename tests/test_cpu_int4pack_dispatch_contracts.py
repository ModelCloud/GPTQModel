# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU int4pack kernels must declare the contract their fused op enforces.

The selector validates a kernel against the exact layer shape before the first
forward. Declaring a wider contract made AUTO choose a CPU int4pack kernel for
layers that the packed op rejects, so loading a model succeeded and the first
forward raised inside ATen.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import torch_aten_kernel as aten_module
from gptqmodel.nn_modules.qlinear import torch_fused as fused_module
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear, validate_quant_linear
from gptqmodel.utils.model import create_quant_layer


pytestmark = [pytest.mark.cpu]


def _aten_available() -> bool:
    return _has_local_int4pack_cpu_ops() and TorchAtenLinear.cached_validate_once()[0]


def _validate(cls, **kwargs):
    params = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "pack_dtype": torch.int32,
        "device": DEVICE.CPU,
    }
    params.update(kwargs)
    return validate_quant_linear(cls, **params)


def _make_layer(cls, *, in_features=512, out_features=64, group_size=128, bias=False):
    layer = cls(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    layer.post_init()
    layer.qweight.zero_()
    layer.qzeros.zero_()
    layer.scales.fill_(0.01)
    layer.g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    return layer


class _Holder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.config = types.SimpleNamespace(quantization_config={"quant_method": "gptq"})
        self.proj = layer


@pytest.mark.skipif(not _aten_available(), reason="ATen CPU int4pack ops are unavailable")
def test_aten_out_features_must_be_divisible_by_16():
    ok, err = _validate(TorchAtenLinear, in_features=512, out_features=100)
    assert not ok and isinstance(err, NotImplementedError)
    ok, err = _validate(TorchAtenLinear, in_features=512, out_features=24)
    assert not ok and isinstance(err, NotImplementedError)
    ok, err = _validate(TorchAtenLinear, in_features=512, out_features=16)
    assert ok, err


@pytest.mark.skipif(not _aten_available(), reason="ATen CPU int4pack ops are unavailable")
def test_aten_group_size_equal_to_in_features_is_rejected():
    # ``group_size == in_features`` bypasses the declared group-size set, but
    # the aten op still only accepts 32/64/128/256.
    ok, err = _validate(TorchAtenLinear, in_features=512, group_size=512, out_features=64)
    assert not ok and isinstance(err, NotImplementedError)
    ok, err = _validate(TorchAtenLinear, in_features=256, group_size=256, out_features=64)
    assert ok, err


def test_fused_cpu_out_features_must_be_divisible_by_16():
    ok, err = _validate(TorchFusedLinear, in_features=512, out_features=8)
    assert not ok and isinstance(err, NotImplementedError)
    ok, err = _validate(TorchFusedLinear, in_features=512, out_features=24)
    assert not ok and isinstance(err, NotImplementedError)
    ok, err = _validate(
        TorchFusedLinear, in_features=512, out_features=16, device=DEVICE.CPU
    )
    assert ok, err


def test_fused_group_size_equal_to_in_features_is_rejected():
    ok, err = _validate(
        TorchFusedLinear, in_features=512, group_size=512, out_features=64
    )
    assert not ok and isinstance(err, NotImplementedError)


def test_avx512_keeps_its_wider_group_size_contract():
    ok, _ = TorchCpuAvx512Linear.cached_validate_once()
    if not ok:
        pytest.skip("AVX512-BF16 is unavailable")
    # The AVX512 kernel owns a different packed op and supports gs=512.
    ok, err = _validate(
        TorchCpuAvx512Linear,
        in_features=512,
        group_size=512,
        out_features=32,
    )
    assert ok, err


@pytest.mark.parametrize("out_features", [8, 100])
@pytest.mark.parametrize("kernel_cls", [TorchAtenLinear, TorchFusedLinear])
def test_auto_falls_through_to_torch_for_odd_output_width(kernel_cls, out_features):
    if kernel_cls is TorchAtenLinear and not _aten_available():
        pytest.skip("ATen CPU int4pack ops are unavailable")

    module = nn.Sequential(nn.Linear(512, out_features, bias=False))
    candidates = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack=False,
        pack_dtype=torch.int32,
        multi_select=True,
    )
    assert kernel_cls in candidates
    selected = create_quant_layer(
        linear_candidates=candidates,
        bits=4,
        desc_act=False,
        dynamic=None,
        group_size=128,
        quant_result={"0": {}},
        module=module,
        sym=True,
        device=DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        backend=BACKEND.AUTO,
    )
    assert selected is TorchLinear


@pytest.mark.parametrize(
    ("module_name", "dequantize", "kernel_cls"),
    [
        ("torch_aten_kernel", aten_module.dequantize_model, TorchAtenLinear),
        ("torch_fused", fused_module.dequantize_model, TorchFusedLinear),
    ],
)
def test_dequantize_model_omits_bias_when_absent(module_name, dequantize, kernel_cls):
    if kernel_cls is TorchAtenLinear and not _aten_available():
        pytest.skip("ATen CPU int4pack ops are unavailable")

    layer = _make_layer(kernel_cls, bias=False)
    layer.scales = layer.scales.to(torch.bfloat16)
    model = _Holder(layer)

    dequantize(model)

    assert isinstance(model.proj, nn.Linear)
    assert model.proj.bias is None
    assert model.proj.weight.dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("dequantize", "kernel_cls"),
    [
        (aten_module.dequantize_model, TorchAtenLinear),
        (fused_module.dequantize_model, TorchFusedLinear),
    ],
)
def test_dequantize_model_preserves_bias_when_present(dequantize, kernel_cls):
    if kernel_cls is TorchAtenLinear and not _aten_available():
        pytest.skip("ATen CPU int4pack ops are unavailable")

    layer = _make_layer(kernel_cls, bias=True)
    model = _Holder(layer)

    dequantize(model)

    assert isinstance(model.proj, nn.Linear)
    assert model.proj.bias is not None
    assert tuple(model.proj.bias.shape) == (64,)
    assert model.proj.bias.dtype is torch.float16
