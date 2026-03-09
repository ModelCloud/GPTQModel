# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.config import (
    FORMAT,
    GGUFBits,
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


def _build_rtn_microbench_case(
    dtype: torch.dtype,
    *,
    bits: int | str = 4,
) -> dict[str, torch.Tensor | int | str | GGUFBits | torch.dtype | torch.device | None]:
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

    format = FORMAT.GPTQ
    if isinstance(bits, GGUFBits):
        format = FORMAT.GGUF
    elif isinstance(bits, str) and not bits.strip().isdigit():
        format = FORMAT.GGUF

    qcfg = RTNQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        format=format,
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
        "bits": qcfg.bits,
        "bit_width": int(qcfg.bits),
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
        bits=case["bit_width"],
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
        bits=case["bit_width"],
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


def _build_rtn_gguf_module(
    case: dict[str, torch.Tensor | int | str | GGUFBits | torch.dtype | torch.device | None],
) -> GGUFTorchQuantLinear:
    module = GGUFTorchQuantLinear(
        bits=case["bits"],
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
        g_idx=case["g_idx_cpu"],
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


@pytest.mark.parametrize(
    ("bits", "tensor_qtype", "bit_width", "variant", "quality"),
    [
        ("q4_k_m", "Q4_K", 4, "k", "m"),
        ("q5_k_s", "Q5_K", 5, "k", "s"),
        ("q5_k_m", "Q5_K", 5, "k", "m"),
        ("q6_k", "Q6_K", 6, "k", None),
    ],
)
def test_baseqmodel_quantize_allows_rtn_gguf_export(
    bits: str,
    tensor_qtype: str,
    bit_width: int,
    variant: str,
    quality: str | None,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    native = _TinyModel().to(device=device, dtype=torch.float16).eval()
    smooth = SmoothMAD(k=2.25)

    qcfg = RTNQuantizeConfig(
        bits=bits,
        group_size=32,
        desc_act=False,
        sym=True,
        format=FORMAT.GGUF,
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
    assert model.quantize_config.format == FORMAT.GGUF
    assert model.quantize_config.export_quant_method() == METHOD.GPTQ
    assert getattr(model.qlinear_kernel, "__name__", "") == "GGUFTorchQuantLinear"

    qmodules = find_modules(model.model, [model.qlinear_kernel])
    assert len(qmodules) == 4

    for qmodule in qmodules.values():
        assert qmodule.qweight.device.type == "cpu"
        assert qmodule.qweight.dtype == torch.uint8
        assert isinstance(qmodule.bits, GGUFBits)
        assert qmodule.bits == bits
        assert qmodule.bits.bits == bit_width
        assert qmodule.bits.version == "q"
        assert qmodule.bits.variant == variant
        assert qmodule.bits.quality == quality
        assert qmodule.gguf_tensor_qtype == tensor_qtype


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


def test_rtn_microbench_gguf_export_matches_reference_bytes():
    gguf = pytest.importorskip("gguf")

    case = _build_rtn_microbench_case(torch.float16, bits="q4_0")
    module = _build_rtn_gguf_module(case)

    reference = gguf.quantize(
        case["rtn_weight_cpu"].numpy().astype(np.float32),
        gguf.GGMLQuantizationType.Q4_0,
    )

    np.testing.assert_array_equal(module.qweight.detach().cpu().numpy(), reference)


def test_rtn_microbench_gguf_export_accepts_structured_bits():
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    bits = GGUFBits(bits=5, version="q", variant="k", quality="m")

    case = _build_rtn_microbench_case(dtype, bits=bits)
    module = _build_rtn_gguf_module(case)

    assert isinstance(case["bits"], GGUFBits)
    assert case["bits"] == "q5_k_m"
    assert isinstance(module.bits, GGUFBits)
    assert module.bits == "q5_k_m"
    assert module.gguf_tensor_qtype == "Q5_K"

    output_stats = _error_stats(case["rtn_output"], module(case["inputs"]))

    assert output_stats["mae"] < 6e-4
    assert output_stats["max"] < 0.012


@pytest.mark.parametrize(
    ("bits", "tensor_qtype"),
    [
        ("q4_k_m", "Q4_K"),
        ("q5_k_s", "Q5_K"),
        ("q5_k_m", "Q5_K"),
        ("q6_k", "Q6_K"),
    ],
)
def test_rtn_microbench_gguf_export_layout_round_trips_with_reference_dequantizer(
    bits: str,
    tensor_qtype: str,
):
    gguf = pytest.importorskip("gguf")

    case = _build_rtn_microbench_case(torch.float16, bits=bits)
    module = _build_rtn_gguf_module(case)

    reference = gguf.dequantize(
        module.qweight.detach().cpu().numpy(),
        getattr(gguf.GGMLQuantizationType, tensor_qtype),
    )

    np.testing.assert_allclose(
        module.dequantize_weight().T.detach().cpu().numpy(),
        reference[:, : case["in_features"]],
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("bits", "weight_mae_max", "output_mae_max", "output_max_max"),
    [
        ("q4_k_m", 0.0015, 7e-4, 0.015),
        ("q5_k_s", 0.0010, 6e-4, 0.012),
        ("q5_k_m", 0.0010, 6e-4, 0.012),
        ("q6_k", 7e-4, 5e-4, 0.010),
    ],
)
def test_rtn_microbench_gguf_export_stays_close_to_rtn(
    dtype: torch.dtype,
    bits: str,
    weight_mae_max: float,
    output_mae_max: float,
    output_max_max: float,
):
    case = _build_rtn_microbench_case(dtype, bits=bits)
    module = _build_rtn_gguf_module(case)

    packed_weight = module.dequantize_weight().T.detach().cpu().to(case["rtn_weight_cpu"].dtype)
    weight_stats = _error_stats(case["rtn_weight_cpu"], packed_weight)

    exported_output = module(case["inputs"])
    output_stats = _error_stats(case["rtn_output"], exported_output)

    assert weight_stats["mae"] < weight_mae_max
    assert output_stats["mae"] < output_mae_max
    assert output_stats["max"] < output_max_max


@pytest.mark.parametrize(
    ("bits", "output_mae_max", "output_max_max"),
    [
        ("q4_k_m", 7e-4, 0.015),
        ("q5_k_m", 6e-4, 0.012),
        ("q6_k", 5e-4, 0.010),
    ],
)
def test_rtn_microbench_gguf_reload_from_state_dict_stays_close_to_rtn(
    bits: str,
    output_mae_max: float,
    output_max_max: float,
):
    case = _build_rtn_microbench_case(torch.float16, bits=bits)
    module = _build_rtn_gguf_module(case)

    reloaded = GGUFTorchQuantLinear(
        bits=bits,
        group_size=case["group_size"],
        sym=True,
        desc_act=False,
        in_features=case["in_features"],
        out_features=case["out_features"],
        bias=True,
        register_buffers=True,
    )
    reloaded.load_state_dict({k: v.detach().cpu() for k, v in module.state_dict().items()})
    reloaded.post_init()
    reloaded = reloaded.to(case["device"]).eval()

    output_stats = _error_stats(case["rtn_output"], reloaded(case["inputs"]))

    assert output_stats["mae"] < output_mae_max
    assert output_stats["max"] < output_max_max
