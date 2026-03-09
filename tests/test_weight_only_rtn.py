# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    QuantizeConfig,
    RTNQuantizeConfig,
    SmoothMAD,
)
from gptqmodel.quantization.rtn import RTN
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import find_modules
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module
from gptqmodel.utils.model import convert_gptq_v2_to_v1_format_module


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
            model_type="tiny_weight_only_test",
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

    qcfg = RTNQuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        smooth=smooth,
        offload_to_disk=False,
        device=str(device),
    )
    rtn = RTN(linear, qcfg=qcfg)
    qweight, _, _, g_idx, *_ = rtn.quantize()
    return qweight.detach().cpu(), g_idx.detach().cpu()


def _microbench_device(dtype: torch.dtype) -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype == torch.float16:
        pytest.skip("float16 RTN microbench requires CUDA for stable matmul")
    return device


def _error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (candidate - reference).abs()
    return {
        "mae": diff.mean().item(),
        "max": diff.max().item(),
    }


def _build_rtn_microbench_case(dtype: torch.dtype) -> dict[str, torch.Tensor | int | torch.dtype | torch.device]:
    device = _microbench_device(dtype)

    torch.manual_seed(1234)

    in_features = 128
    out_features = 128
    group_size = 32
    batch_size = 16

    # Use LLM-like weight scales so the microbench measures RTN/export behavior,
    # not an unrealistically wide toy distribution.
    weight_master = torch.randn(out_features, in_features, dtype=torch.float32) * 0.01
    bias_master = torch.randn(out_features, dtype=torch.float32) * 0.001
    inputs_master = torch.randn(batch_size, in_features, dtype=torch.float32) * 0.1

    linear = nn.Linear(
        in_features,
        out_features,
        bias=True,
        dtype=dtype,
        device=device,
    ).eval()
    with torch.no_grad():
        linear.weight.copy_(weight_master.to(device=device, dtype=dtype))
        linear.bias.copy_(bias_master.to(device=device, dtype=dtype))

    inputs = inputs_master.to(device=device, dtype=dtype)
    native_output = linear(inputs)

    qcfg = RTNQuantizeConfig(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        smooth=SmoothMAD(k=2.25),
        offload_to_disk=False,
        device=device.type,
    )
    rtn_weight, scales, zeros, g_idx, *_ = RTN(linear, qcfg=qcfg).quantize()
    rtn_output = F.linear(inputs, rtn_weight.to(device=device, dtype=dtype), linear.bias)

    cpu_linear = nn.Linear(in_features, out_features, bias=True, dtype=torch.float16).cpu().eval()
    with torch.no_grad():
        cpu_linear.weight.copy_(rtn_weight.detach().cpu().to(torch.float16))
        cpu_linear.bias.copy_(linear.bias.detach().cpu().to(torch.float16))

    return {
        "device": device,
        "dtype": dtype,
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "inputs": inputs,
        "native_output": native_output,
        "rtn_output": rtn_output,
        "rtn_weight_cpu": rtn_weight.detach().cpu(),
        "scales_cpu": scales.detach().cpu(),
        "zeros_cpu": zeros.detach().cpu(),
        "g_idx_cpu": g_idx.detach().cpu(),
        "cpu_linear": cpu_linear,
    }


def _build_rtn_gptq_module(case: dict[str, torch.Tensor | int | torch.dtype | torch.device]) -> TorchQuantLinear:
    module = TorchQuantLinear(
        bits=4,
        group_size=case["group_size"],
        sym=True,
        desc_act=False,
        in_features=case["in_features"],
        out_features=case["out_features"],
        bias=True,
        register_buffers=False,
    )
    module.pack_block(
        linear=case["cpu_linear"],
        scales=case["scales_cpu"],
        zeros=case["zeros_cpu"],
        g_idx=case["g_idx_cpu"],
    )
    # `pack_block()` gives us the in-memory GPTQ runtime layout that
    # `TorchQuantLinear` executes with. That is not the same thing as the
    # serialized GPTQ checkpoint layout this project can export.
    #
    # The real checkpoint round-trip is:
    # 1. runtime/internal layout -> GPTQ v1 serialized layout at export time
    # 2. GPTQ v1 serialized layout -> runtime/internal layout at load time
    #
    # This helper intentionally performs that round-trip so the microbench
    # validates export + reload behavior, not just the raw in-memory pack path.
    convert_gptq_v2_to_v1_format_module(
        module,
        QuantizeConfig(bits=module.bits, quant_method=METHOD.GPTQ),
    )
    convert_gptq_v1_to_v2_format_module(
        module,
        bits=module.bits,
        pack_dtype=module.pack_dtype,
    )
    # Skip the compile-heavy TorchQuantLinear override; the microbench only
    # needs unpack buffers initialized for numerical comparisons.
    PackableQuantLinear.post_init(module)
    return module.to(case["device"]).eval()


def _build_rtn_awq_module(case: dict[str, torch.Tensor | int | torch.dtype | torch.device]) -> AwqTorchQuantLinear:
    module = AwqTorchQuantLinear(
        bits=4,
        group_size=case["group_size"],
        sym=True,
        desc_act=False,
        in_features=case["in_features"],
        out_features=case["out_features"],
        bias=True,
        register_buffers=False,
    )
    module.pack(
        linear=case["cpu_linear"],
        scales=case["scales_cpu"],
        zeros=case["zeros_cpu"],
    )
    module.post_init()
    return module.to(case["device"]).eval()


def test_baseqmodel_quantize_uses_weight_only_rtn_pipeline():
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

    assert "weight_only_rtn" in result
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

    assert "weight_only_rtn" in result
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rtn_microbench_quantized_output_stays_close_to_native(dtype: torch.dtype):
    case = _build_rtn_microbench_case(dtype)

    stats = _error_stats(case["native_output"], case["rtn_output"])

    assert stats["mae"] < 0.0010
    assert stats["max"] < 0.0060


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rtn_microbench_gptq_export_stays_close_to_rtn(dtype: torch.dtype):
    case = _build_rtn_microbench_case(dtype)
    module = _build_rtn_gptq_module(case)

    packed_weight = module.dequantize_weight().T.detach().cpu().to(case["rtn_weight_cpu"].dtype)
    weight_stats = _error_stats(case["rtn_weight_cpu"], packed_weight)

    exported_output = module(case["inputs"])
    output_stats = _error_stats(case["rtn_output"], exported_output)

    assert weight_stats["mae"] < 2e-5
    assert output_stats["mae"] < 2e-5
    assert output_stats["max"] < 3e-4


def test_rtn_microbench_awq_export_stays_close_to_rtn():
    case = _build_rtn_microbench_case(torch.float16)
    module = _build_rtn_awq_module(case)

    packed_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=module.bits,
        group_size=module.group_size,
    ).detach().cpu().to(case["rtn_weight_cpu"].dtype)
    weight_stats = _error_stats(case["rtn_weight_cpu"], packed_weight)

    exported_output = module(case["inputs"])
    output_stats = _error_stats(case["rtn_output"], exported_output)

    assert weight_stats["mae"] < 0.0120
    assert output_stats["mae"] < 1e-5
    assert output_stats["max"] < 1e-4
