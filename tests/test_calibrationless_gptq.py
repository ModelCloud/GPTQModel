# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.config import (
    FailSafe,
    FailSafeStrategy,
    FORMAT,
    METHOD,
    QuantizeConfig,
    RTNQuantizeConfig,
    SmoothMAD,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import find_modules
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module


class _TinyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = _TinyMLP(hidden_size)


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_TinyBlock(hidden_size) for _ in range(layers)])


class _TinyModel(nn.Module):
    def __init__(self, hidden_size: int = 32, layers: int = 2):
        super().__init__()
        self.model = _TinyBackbone(hidden_size=hidden_size, layers=layers)
        self.config = SimpleNamespace(
            use_cache=False,
            tie_word_embeddings=False,
            model_type="tiny_calibrationless_test",
        )


class _TinyQModel(BaseQModel):
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "mlp": ("up_proj:0", "down_proj:1"),
        },
    ]


def _reference_rtn_quantized_weight(weight: torch.Tensor, device: torch.device, smooth: SmoothMAD) -> tuple[torch.Tensor, torch.Tensor]:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
    linear.weight.data.copy_(weight)
    linear.to(device)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        failsafe=FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=smooth,
        ),
        offload_to_disk=False,
        device=str(device),
    )
    gptq = GPTQ(linear, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    qweight, _, _, g_idx, *_ = gptq.quantize()
    gptq.free()
    return qweight.detach().cpu(), g_idx.detach().cpu()


def test_baseqmodel_quantize_uses_calibrationless_gptq_pipeline():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    native = _TinyModel().to(device=device, dtype=torch.float16).eval()
    original_state = copy.deepcopy(native.state_dict())

    smooth = SmoothMAD(k=2.25)
    qcfg = RTNQuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        smooth=smooth,
        offload_to_disk=False,
        device=device_type,
    )

    model = _TinyQModel(
        model=native,
        quantized=False,
        quantize_config=qcfg,
        tokenizer=None,
    )

    result = model.quantize(calibration=None, backend=BACKEND.TORCH)

    assert "calibrationless_gptq" in result
    assert model.quantized is True

    qmodules = find_modules(model.model, [TorchQuantLinear])
    assert len(qmodules) == 4

    for name, qmodule in qmodules.items():
        original_weight = original_state[f"{name}.weight"].to(dtype=torch.float16)
        expected_qweight, expected_g_idx = _reference_rtn_quantized_weight(original_weight, device=device, smooth=smooth)

        assert qmodule.qzero_format() == 1
        assert qmodule.qweight.device.type == "cpu"
        assert qmodule.qzeros.device.type == "cpu"
        assert qmodule.scales.device.type == "cpu"
        assert qmodule.g_idx.device.type == "cpu"

        dequant_module = copy.deepcopy(qmodule)
        if dequant_module.qzero_format() == 1:
            convert_gptq_v1_to_v2_format_module(
                dequant_module,
                bits=dequant_module.bits,
                pack_dtype=dequant_module.pack_dtype,
            )
        if not hasattr(dequant_module, "wf_unsqueeze_zero"):
            dequant_module.post_init()

        actual_qweight = dequant_module.dequantize_weight().T.detach().cpu().to(dtype=expected_qweight.dtype)
        actual_error = (actual_qweight - original_weight.cpu()).abs().mean().item()
        expected_error = (expected_qweight - original_weight.cpu()).abs().mean().item()

        assert actual_error <= expected_error + 0.01
        assert actual_error < 0.05
        torch.testing.assert_close(qmodule.g_idx.detach().cpu(), expected_g_idx)


def test_baseqmodel_quantize_allows_rtn_awq_export():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    native = _TinyModel().to(device=device, dtype=torch.float16).eval()
    smooth = SmoothMAD(k=2.25)

    qcfg = RTNQuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        format=FORMAT.GEMM,
        smooth=smooth,
        offload_to_disk=False,
        device=device_type,
    )

    model = _TinyQModel(
        model=native,
        quantized=False,
        quantize_config=qcfg,
        tokenizer=None,
    )

    result = model.quantize(calibration=None, backend=BACKEND.AUTO)

    assert "calibrationless_gptq" in result
    assert model.quantized is True
    assert model.quantize_config.format == FORMAT.GEMM
    assert model.quantize_config.export_quant_method() == METHOD.AWQ
    assert getattr(model.qlinear_kernel, "__name__", "") == "AwqTorchQuantLinear"

    qmodules = find_modules(model.model, [model.qlinear_kernel])
    assert len(qmodules) == 4

    for qmodule in qmodules.values():
        assert qmodule.qweight.device.type == "cpu"
        assert qmodule.qzeros.device.type == "cpu"
        assert qmodule.scales.device.type == "cpu"
