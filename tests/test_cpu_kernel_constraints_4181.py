# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU int4pack kernels must not be selected for shapes their op rejects.

These are the regression cases from issue #4181: an odd output width and a
``group_size`` that equals ``in_features`` both bypass the selector's declared
contract, so AUTO used to load a kernel that failed on the first forward.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import torch as torch_module
from gptqmodel.nn_modules.qlinear import torch_aten_kernel as aten_module
from gptqmodel.nn_modules.qlinear import torch_fused as fused_module
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import (
    get_auto_backend_kernel_mapping,
    select_quant_linear,
)
from gptqmodel.utils.model import create_quant_layer


pytestmark = [pytest.mark.cpu]


def _aten_available() -> bool:
    return _has_local_int4pack_cpu_ops() and TorchAtenLinear.cached_validate_once()[0]


def _make_layer(
    cls,
    *,
    in_features: int = 512,
    out_features: int = 64,
    group_size: int = 128,
    bias: bool = False,
):
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


def _auto_candidates(*, group_size: int):
    return select_quant_linear(
        bits=4,
        group_size=group_size,
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


def _select_for_module(module: nn.Module, *, group_size: int):
    candidates = _auto_candidates(group_size=group_size)
    selected = create_quant_layer(
        linear_candidates=candidates,
        bits=4,
        desc_act=False,
        dynamic=None,
        group_size=group_size,
        quant_result={"0": {}},
        module=module,
        sym=True,
        device=DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        backend=BACKEND.AUTO,
    )
    return candidates, selected


def _auto_kernel_classes():
    """The available AUTO registry entries for GPTQ, in selection order."""

    mapping = get_auto_backend_kernel_mapping()[METHOD.GPTQ][FORMAT.GPTQ]
    classes = []
    for cls in mapping.values():
        devices = getattr(cls, "SUPPORTS_DEVICES", [])
        if DEVICE.CPU not in devices and DEVICE.ALL not in devices:
            continue
        if cls not in classes and cls.cached_validate_once()[0]:
            classes.append(cls)
    return classes


class _Holder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.config = types.SimpleNamespace(
            quantization_config={"quant_method": "gptq"}
        )
        self.proj = layer


@pytest.mark.parametrize("out_features", [8, 100])
def test_auto_odd_output_width_selects_torch_and_forwards(out_features):
    """AUTO must not pick an int4pack kernel for N not divisible by 16."""

    module = nn.Sequential(nn.Linear(512, out_features, bias=False))
    # The model-wide candidate pass does not know the layer width yet; the
    # per-module selection below is what must reject the int4pack kernels.
    _, selected = _select_for_module(module, group_size=128)

    assert selected is TorchLinear

    layer = _make_layer(TorchLinear, out_features=out_features)
    out = layer(torch.randn(2, 512, dtype=torch.float16))

    assert tuple(out.shape) == (2, out_features)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("in_features", [512, 1024])
def test_auto_group_size_equal_in_features_avoids_aten_and_fused(in_features):
    """``group_size == in_features`` must not bypass the CPU op's gs contract."""

    module = nn.Sequential(nn.Linear(in_features, 64, bias=False))
    # The model-wide pass cannot resolve ``group_size == in_features`` because
    # it does not know the layer width; the per-module pass does. Use the full
    # AUTO registry so that escape is actually exercised.
    selected = create_quant_layer(
        linear_candidates=_auto_kernel_classes(),
        bits=4,
        desc_act=False,
        dynamic=None,
        group_size=in_features,
        quant_result={"0": {}},
        module=module,
        sym=True,
        device=DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        backend=BACKEND.AUTO,
    )

    assert selected not in (TorchAtenLinear, TorchFusedLinear)


@pytest.mark.parametrize(
    ("dequantize", "kernel_cls"),
    [
        (torch_module.dequantize_model, TorchLinear),
        (aten_module.dequantize_model, TorchAtenLinear),
        (fused_module.dequantize_model, TorchFusedLinear),
    ],
)
def test_dequantize_model_biasless_layer_has_no_bias_and_forwards(
    dequantize, kernel_cls
):
    if kernel_cls is TorchAtenLinear and not _aten_available():
        pytest.skip("ATen CPU int4pack ops are unavailable")

    layer = _make_layer(kernel_cls, bias=False)
    # A bare quantized layer falls back to the checkpoint scale dtype; keep it
    # BF16 so a hard-coded FP16 weight would fail this assertion.
    layer.scales = layer.scales.to(torch.bfloat16)
    model = _Holder(layer)

    dequantize(model)

    assert isinstance(model.proj, nn.Linear)
    assert model.proj.bias is None
    assert model.proj.weight.dtype is torch.bfloat16

    out = model.proj(torch.randn(2, layer.in_features, dtype=torch.bfloat16))
    assert tuple(out.shape) == (2, layer.out_features)
    assert torch.isfinite(out).all()
