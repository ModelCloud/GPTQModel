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
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    AutoModuleDecoderConfig,
    GGUFBits,
    GGUFConfig,
    QuantizeConfig,
    RTNConfig,
    SmoothMAD,
    quant_bits_width,
)
from gptqmodel.quantization.rtn import RTN
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import (
    convert_gptq_v1_to_v2_format_module,
    convert_gptq_v2_to_v1_format_module,
    find_modules,
)


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

    qcfg = RTNConfig(
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

    requested_bits: int | GGUFBits = bits
    rtn_bits = bits
    if isinstance(bits, GGUFBits):
        requested_bits = bits
        rtn_bits = int(bits)
    elif isinstance(bits, str) and not bits.strip().isdigit():
        requested_bits = GGUFBits.from_string(bits)
        rtn_bits = quant_bits_width(bits)

    qcfg = RTNConfig(
        bits=rtn_bits,
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
        "bits": requested_bits,
        "bit_width": quant_bits_width(requested_bits),
        "inputs": inputs,
        "native_output": native_output,
        "rtn_output": rtn_output,
        "rtn_weight_cpu": rtn_weight.detach().cpu(),
        "scales_cpu": scales.detach().cpu(),
        "zeros_cpu": zeros.detach().cpu(),
        "g_idx_cpu": g_idx.detach().cpu(),
        "cpu_linear": cpu_linear,
    }


def _build_rtn_gptq_module(case: dict[str, torch.Tensor | int | torch.dtype | torch.device]) -> TorchLinear:
    module = TorchLinear(
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
    # `TorchLinear` executes with. That is not the same thing as the
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
    # Skip the compile-heavy TorchLinear override; the microbench only
    # needs unpack buffers initialized for numerical comparisons.
    PackableQuantLinear.post_init(module)
    return module.to(case["device"]).eval()


def _build_rtn_awq_module(case: dict[str, torch.Tensor | int | torch.dtype | torch.device]) -> AwqTorchLinear:
    module = AwqTorchLinear(
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
) -> GGUFTorchLinear:
    module = GGUFTorchLinear(
        bits=case["bits"],
        group_size=-1,
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
    qcfg = RTNConfig(
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

    qmodules = find_modules(model.model, [TorchLinear])
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

    qcfg = RTNConfig(
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
    assert getattr(model.qlinear_kernel, "__name__", "") == "AwqTorchLinear"

    qmodules = find_modules(model.model, [model.qlinear_kernel])
    assert len(qmodules) == 4

    for qmodule in qmodules.values():
        assert qmodule.qweight.device.type == "cpu"
        assert qmodule.qzeros.device.type == "cpu"
        assert qmodule.scales.device.type == "cpu"


@pytest.mark.parametrize(
    ("bits", "tensor_qtype", "bit_width", "variant", "quality"),
    [
        ("q4_k_s", "Q4_K", 4, "k", "s"),
        ("q4_k_m", "Q4_K", 4, "k", "m"),
        ("q5_k_s", "Q5_K", 5, "k", "s"),
        ("q5_k_m", "Q5_K", 5, "k", "m"),
        ("q6_k", "Q6_K", 6, "k", None),
    ],
)
def test_baseqmodel_quantize_allows_direct_gguf_export(
    bits: str,
    tensor_qtype: str,
    bit_width: int,
    variant: str,
    quality: str | None,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    public_format = GGUFBits.from_string(bits).to_public_format()

    native = _TinyModel().to(device=device, dtype=torch.float16).eval()
    qcfg = GGUFConfig(
        bits=bit_width,
        format=public_format,
        smoother=None,
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

    assert "weight_only_gguf" in result
    assert model.quantized is True
    assert model.quantize_config.format == public_format
    assert model.quantize_config.bits == bit_width
    assert model.quantize_config.quant_method == METHOD.GGUF
    assert model.quantize_config.export_quant_method() == METHOD.GGUF
    expected_kernel = GGUFTritonKernel if device_type == "cuda" else GGUFTorchLinear
    assert model.qlinear_kernel is expected_kernel

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
        expected_padded_in_features = (
            (qmodule.in_features + qmodule.gguf_block_size - 1) // qmodule.gguf_block_size
        ) * qmodule.gguf_block_size
        assert qmodule.padded_in_features == expected_padded_in_features
        assert qmodule.qweight.shape == (qmodule.out_features, qmodule._bytes_per_row())


def test_baseqmodel_quantize_gguf_weight_only_skips_rtn(monkeypatch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    native = _TinyModel().to(device=device, dtype=torch.float16).eval()

    qcfg = GGUFConfig(
        bits=4,
        format="q_k_m",
        smoother=SmoothMAD(k=2.25),
        offload_to_disk=False,
        device=device_type,
    )

    model = _TinyQModel(
        model=native,
        quantized=False,
        quantize_config=qcfg,
        tokenizer=None,
    )

    def _fail_quantize(*args, **kwargs):
        raise AssertionError("RTN.quantize should not be called for direct GGUF packing")

    monkeypatch.setattr(RTN, "quantize", _fail_quantize)

    result = model.quantize(calibration=None, backend=BACKEND.AUTO)

    assert "weight_only_gguf" in result
    assert model.quantized is True
    qmodules = find_modules(model.model, [model.qlinear_kernel])
    assert len(qmodules) == 4


def test_baseqmodel_quantize_gguf_weight_only_applies_auto_module_decoder(monkeypatch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    native = _TinyModel().to(device=device, dtype=torch.float16).eval()

    qcfg = GGUFConfig(
        bits=4,
        format="q_k_m",
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
        offload_to_disk=False,
        device=device_type,
    )

    model = _TinyQModel(
        model=native,
        quantized=False,
        quantize_config=qcfg,
        tokenizer=None,
    )

    materialize_calls = []
    original_shell_module_materialize = BaseQModel.shell_module_materialize

    def _spy_shell_module_materialize(self, *args, **kwargs):
        materialize_calls.append(kwargs.get("role"))
        return original_shell_module_materialize(self, *args, **kwargs)

    monkeypatch.setattr(BaseQModel, "shell_module_materialize", _spy_shell_module_materialize)

    result = model.quantize(calibration=None, backend=BACKEND.AUTO)

    assert "weight_only_gguf" in result
    assert materialize_calls.count("quant_source") == 4


@pytest.mark.parametrize("bits", ["q4_k_m", "q5_k_m", "q6_k"])
def test_gguf_pack_original_auto_pads_non_aligned_k_in_features(bits: str):
    torch.manual_seed(77)

    in_features = 130
    out_features = 48
    linear = nn.Linear(in_features, out_features, bias=True, dtype=torch.float16).cpu().eval()

    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.02)
        linear.bias.normal_(mean=0.0, std=0.01)

    module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    ).cpu().eval()
    module.pack_original(linear, scales=None, zeros=None)

    assert module.in_features == in_features
    assert module.gguf_block_size == 256
    assert module.padded_in_features == 256
    assert module.qweight.shape == (out_features, module._bytes_per_row())

    x = torch.randn(7, in_features, dtype=torch.float32)
    with torch.inference_mode():
        native_out = F.linear(x, linear.weight.detach().to(torch.float32), linear.bias.detach().to(torch.float32))
        gguf_out = module(x)

    stats = _error_stats(native_out, gguf_out.to(torch.float32))
    assert stats["mae"] < 0.02
    assert stats["max"] < 0.08


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


@pytest.mark.parametrize("bits", ["q4_0", "q4_k_m"])
def test_gguf_dequantize_weight_accepts_requested_dtype_and_device(bits: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    case = _build_rtn_microbench_case(dtype, bits=bits)
    module = _build_rtn_gguf_module(case)

    direct = module.dequantize_weight(device=case["device"], dtype=case["dtype"])
    baseline = module.dequantize_weight().to(device=case["device"], dtype=case["dtype"])

    assert direct.device == case["device"]
    assert direct.dtype == case["dtype"]
    torch.testing.assert_close(direct, baseline, atol=2e-3, rtol=0.0)


def test_gguf_forward_requests_dequantized_weight_in_input_dtype(monkeypatch):
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    case = _build_rtn_microbench_case(dtype, bits="q4_0")
    module = _build_rtn_gguf_module(case)

    observed: dict[str, torch.device | torch.dtype | None] = {"device": None, "dtype": None}
    original = GGUFTorchLinear.dequantize_weight

    def _wrapped(self, *, device=None, dtype=None):
        observed["device"] = None if device is None else torch.device(device)
        observed["dtype"] = dtype
        return original(self, device=device, dtype=dtype)

    monkeypatch.setattr(GGUFTorchLinear, "dequantize_weight", _wrapped)

    module(case["inputs"])

    assert observed["device"] == case["inputs"].device
    assert observed["dtype"] == case["inputs"].dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF K-path CUDA tests")
@pytest.mark.parametrize(
    ("bits", "tensor_qtype"),
    [
        ("q4_k_m", "Q4_K"),
        ("q5_k_m", "Q5_K"),
        ("q6_k", "Q6_K"),
    ],
)
def test_gguf_k_dequantize_weight_matches_reference_on_cuda(bits: str, tensor_qtype: str):
    gguf = pytest.importorskip("gguf")

    case = _build_rtn_microbench_case(torch.float16, bits=bits)
    module = _build_rtn_gguf_module(case).to(case["device"]).eval()

    actual = module.dequantize_weight(device=case["device"], dtype=torch.float32).T.detach().cpu().numpy()
    reference = gguf.dequantize(
        module.qweight.detach().cpu().numpy(),
        getattr(gguf.GGMLQuantizationType, tensor_qtype),
    )[:, : case["in_features"]]

    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF fused K forward tests")
@pytest.mark.parametrize("bits", ["q4_k_m", "q5_k_m", "q6_k"])
def test_gguf_k_fused_forward_matches_dense_baseline(bits: str):
    case = _build_rtn_microbench_case(torch.float16, bits=bits)
    module = _build_rtn_gguf_module(case).to(case["device"]).eval()
    module.gguf_fused_cuda_max_rows = case["inputs"].shape[0]
    module.gguf_fused_cuda_min_matrix_elements = 0
    module.clear_weight_cache()

    baseline = module._forward_dequant_matmul(case["inputs"])
    fused = module._forward_fused_k(case["inputs"])

    output_stats = _error_stats(baseline.to(torch.float32), fused.to(torch.float32))
    assert output_stats["mae"] < 2e-4
    assert output_stats["max"] < 3e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton fused K tests")
@pytest.mark.parametrize("bits", ["q4_k_m", "q5_k_m", "q6_k"])
def test_gguf_triton_fused_forward_matches_dense_baseline(bits: str):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear.gguf_triton import triton_available

    if not triton_available():
        pytest.skip("Triton GGUF fused kernel unavailable")

    case = _build_rtn_microbench_case(torch.float16, bits=bits)
    module = _build_rtn_gguf_module(case).to(case["device"]).eval()
    triton_module = GGUFTritonKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=case["in_features"],
        out_features=case["out_features"],
        bias=True,
        register_buffers=True,
    ).to(case["device"]).eval()
    triton_module.load_state_dict(module.state_dict(), strict=True)
    triton_module.clear_weight_cache()

    baseline = module._forward_dequant_matmul(case["inputs"])
    if module.bias is not None:
        baseline = baseline + module.bias.to(device=baseline.device, dtype=baseline.dtype)
    fused = triton_module(case["inputs"])

    output_stats = _error_stats(baseline.to(torch.float32), fused.to(torch.float32))
    assert output_stats["mae"] < 2e-4
    assert output_stats["max"] < 3e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton sign-only tests")
def test_gguf_triton_q1_0_g128_fused_forward_matches_dense_baseline():
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear.gguf_triton import triton_available

    if not triton_available():
        pytest.skip("Triton GGUF fused kernel unavailable")

    torch.manual_seed(1234)
    cpu_linear = nn.Linear(256, 192, bias=True, dtype=torch.float16).cpu().eval()
    with torch.no_grad():
        cpu_linear.weight.normal_(mean=0.0, std=0.01)
        cpu_linear.bias.normal_(mean=0.0, std=0.001)

    module = GGUFTorchLinear(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=192,
        bias=True,
        register_buffers=False,
    )
    module.pack(
        linear=cpu_linear,
        scales=torch.empty(0),
        zeros=torch.empty(0),
        g_idx=None,
    )
    module.post_init()
    module = module.to("cuda").eval()

    triton_module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=192,
        bias=True,
        register_buffers=True,
    ).to("cuda").eval()
    triton_module.load_state_dict(module.state_dict(), strict=True)
    triton_module.clear_weight_cache()

    inputs = torch.randn(16, 256, device="cuda", dtype=torch.float16) * 0.1
    baseline = module._forward_dequant_matmul(inputs)
    if module.bias is not None:
        baseline = baseline + module.bias.to(device=baseline.device, dtype=baseline.dtype)
    fused = triton_module(inputs)

    output_stats = _error_stats(baseline.to(torch.float32), fused.to(torch.float32))
    assert output_stats["mae"] < 2e-4
    assert output_stats["max"] < 3e-3


def test_gguf_triton_kernel_rejects_unsupported_formats():
    with pytest.raises(NotImplementedError, match="only supports fused GGUF Triton formats"):
        GGUFTritonKernel(
            bits="q4_0",
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=64,
            out_features=48,
            bias=False,
            register_buffers=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_selects_large_config_bank_for_large_k(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[object] = []
    small_kernel = object()
    large_kernel = object()

    def _fake_launch(kernel, x, output, *args):
        calls.append(kernel)
        return output

    monkeypatch.setattr(gguf_triton, "_gguf_q4_k_fused_matmul_kernel_small", small_kernel)
    monkeypatch.setattr(gguf_triton, "_gguf_q4_k_fused_matmul_kernel_large", large_kernel)
    monkeypatch.setattr(gguf_triton, "_launch", _fake_launch)

    x_small = torch.randn(2, 2048, device="cuda", dtype=torch.float16)
    qs_small = torch.empty((8, 144, 32), device="cuda", dtype=torch.uint8)
    scale_small = torch.empty((8, 8, 32), device="cuda", dtype=torch.float16)
    min_small = torch.empty((8, 8, 32), device="cuda", dtype=torch.float16)
    gguf_triton.fused_q4_k_matmul(x_small, qs_small, scale_small, min_small)
    assert calls[-1] is small_kernel

    x_large = torch.randn(2, 4096, device="cuda", dtype=torch.float16)
    qs_large = torch.empty((16, 144, 32), device="cuda", dtype=torch.uint8)
    scale_large = torch.empty((16, 8, 32), device="cuda", dtype=torch.float16)
    min_large = torch.empty((16, 8, 32), device="cuda", dtype=torch.float16)
    gguf_triton.fused_q4_k_matmul(x_large, qs_large, scale_large, min_large)
    assert calls[-1] is large_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_selects_large_config_bank_for_large_q1_0_g128(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[object] = []
    small_kernel = object()
    large_kernel = object()

    def _fake_launch(kernel, x, output, *args):
        calls.append(kernel)
        return output

    monkeypatch.setattr(gguf_triton, "_gguf_q1_0_g128_fused_matmul_kernel_small", small_kernel)
    monkeypatch.setattr(gguf_triton, "_gguf_q1_0_g128_fused_matmul_kernel_large", large_kernel)
    monkeypatch.setattr(gguf_triton, "_launch", _fake_launch)

    x_small = torch.randn(2, 1024, device="cuda", dtype=torch.float16)
    sign_small = torch.empty((8, 16, 32), device="cuda", dtype=torch.uint8)
    scale_small = torch.empty((8, 32), device="cuda", dtype=torch.float16)
    gguf_triton.fused_q1_0_g128_matmul(x_small, sign_small, scale_small)
    assert calls[-1] is small_kernel

    x_large = torch.randn(2, 2048, device="cuda", dtype=torch.float16)
    sign_large = torch.empty((16, 16, 32), device="cuda", dtype=torch.uint8)
    scale_large = torch.empty((16, 32), device="cuda", dtype=torch.float16)
    gguf_triton.fused_q1_0_g128_matmul(x_large, sign_large, scale_large)
    assert calls[-1] is large_kernel


def test_select_q1_0_g128_fixed_launch_config_targets_arch_decode_shapes():
    from gptqmodel.nn_modules.qlinear.gguf_triton import _select_q1_0_g128_fixed_launch_config

    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 0), rows=1, cols=2048) == {
        "BLOCK_SIZE_M": 4,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 0), rows=1, cols=6144) == {
        "BLOCK_SIZE_M": 2,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=1, cols=2048) == {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 32,
        "num_warps": 8,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=1, in_features=2048, cols=2048) == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=1, in_features=6144, cols=2048) == {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 32,
        "num_warps": 8,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=1, cols=1024) == {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 32,
        "num_warps": 8,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=1, cols=6144) == {
        "BLOCK_SIZE_M": 2,
        "BLOCK_SIZE_N": 32,
        "num_warps": 8,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=1, in_features=2048, cols=6144) == {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 64,
        "num_warps": 8,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 0), rows=64, cols=2048) == {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 16,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 0), rows=64, cols=6144) == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=64, cols=2048) == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 16,
        "num_warps": 8,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=64, cols=6144) == {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=1, cols=4096) is None
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 0), rows=2, cols=6144) is None
    assert _select_q1_0_g128_fixed_launch_config(capability=(8, 9), rows=256, cols=2048) is None


def test_select_q1_0_g128_u32_layout_targets_sm80_down_proj():
    from gptqmodel.nn_modules.qlinear.gguf_triton import _select_q1_0_g128_u32_layout

    assert _select_q1_0_g128_u32_layout(capability=(8, 0), in_features=6144, out_features=2048) is True
    assert _select_q1_0_g128_u32_layout(capability=(8, 0), in_features=2048, out_features=1024) is True
    assert _select_q1_0_g128_u32_layout(capability=(8, 0), in_features=2048, out_features=2048) is True
    assert _select_q1_0_g128_u32_layout(capability=(8, 0), in_features=2048, out_features=6144) is True
    assert _select_q1_0_g128_u32_layout(capability=(8, 0), in_features=1024, out_features=2048) is False
    assert _select_q1_0_g128_u32_layout(capability=(8, 9), in_features=6144, out_features=2048) is False
    assert _select_q1_0_g128_u32_layout(capability=None, in_features=6144, out_features=2048) is False


def test_select_q1_0_g128_u32_fixed_launch_config_targets_sm80_down_proj_ranges():
    from gptqmodel.nn_modules.qlinear.gguf_triton import _select_q1_0_g128_u32_fixed_launch_config

    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 0),
        rows=1,
        in_features=6144,
        out_features=2048,
    ) == {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 16,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 0),
        rows=64,
        in_features=6144,
        out_features=2048,
    ) == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 0),
        rows=1,
        in_features=2048,
        out_features=1024,
    ) == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "num_warps": 2,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 0),
        rows=1,
        in_features=2048,
        out_features=2048,
    ) == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 2,
    }
    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 0),
        rows=1,
        in_features=2048,
        out_features=6144,
    ) == {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 0),
        rows=4,
        in_features=6144,
        out_features=2048,
    ) is None
    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 0),
        rows=256,
        in_features=6144,
        out_features=2048,
    ) is None
    assert _select_q1_0_g128_u32_fixed_launch_config(
        capability=(8, 9),
        rows=64,
        in_features=6144,
        out_features=2048,
    ) is None


def test_use_q1_0_g128_k2048_decode_specialization_only_targets_decode_k2048():
    from gptqmodel.nn_modules.qlinear.gguf_triton import _use_q1_0_g128_k2048_decode_specialization

    assert _use_q1_0_g128_k2048_decode_specialization(rows=1, in_features=2048) is True
    assert _use_q1_0_g128_k2048_decode_specialization(rows=4, in_features=2048) is False
    assert _use_q1_0_g128_k2048_decode_specialization(rows=1, in_features=6144) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_uses_fixed_decode_launch_for_sm80_q1_0_g128(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[tuple[str, object, dict[str, int] | None]] = []
    fixed_config = {
        "BLOCK_SIZE_M": 2,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 4,
    }

    def _fake_fixed_matmul(x, sign_bytes, scale, *, fixed_config):
        calls.append(("fixed", gguf_triton._gguf_q1_0_g128_fused_matmul_kernel_impl, fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_fused_matmul(x, sign_bytes, scale):
        calls.append(("generic", None, None))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_fixed_matmul", _fake_fixed_matmul)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_matmul", _fake_fused_matmul)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=192,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "fixed_decode_config": fixed_config,
            "sign_bytes": torch.empty((2, 16, 192), device="cuda", dtype=torch.uint8),
            "scale": torch.empty((2, 192), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == [("fixed", gguf_triton._gguf_q1_0_g128_fused_matmul_kernel_impl, fixed_config)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton u32 tests")
def test_gguf_triton_q1_0_g128_u32_fused_matches_byte_fused():
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    torch.manual_seed(1234)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
    sign_bytes = torch.randint(0, 256, (2, 16, 192), device="cuda", dtype=torch.uint8)
    sign_words = (
        sign_bytes.to(torch.int32)[:, 0::4, :]
        | torch.bitwise_left_shift(sign_bytes.to(torch.int32)[:, 1::4, :], 8)
        | torch.bitwise_left_shift(sign_bytes.to(torch.int32)[:, 2::4, :], 16)
        | torch.bitwise_left_shift(sign_bytes.to(torch.int32)[:, 3::4, :], 24)
    ).contiguous()
    scale = (torch.rand((2, 192), device="cuda", dtype=torch.float16) + 0.05).contiguous()

    byte_out = gguf_triton.fused_q1_0_g128_matmul(x, sign_bytes, scale)
    u32_out = gguf_triton.fused_q1_0_g128_u32_matmul(x, sign_words, scale)

    output_stats = _error_stats(byte_out.to(torch.float32), u32_out.to(torch.float32))
    assert output_stats["mae"] < 2e-4
    assert output_stats["max"] < 3e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_sm80_down_proj_to_u32_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[tuple[str, dict[str, int] | None]] = []

    fixed_config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 4,
    }

    def _fake_u32_fixed(x, sign_words, scale, *, fixed_config):
        calls.append(("u32_fixed", fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_byte(x, sign_bytes, scale):
        calls.append(("byte", None))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_u32_fixed_matmul", _fake_u32_fixed)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_matmul", _fake_byte)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=6144,
        out_features=2048,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": True,
            "fixed_decode_config": None,
            "sign_bytes": torch.empty((48, 16, 2048), device="cuda", dtype=torch.uint8),
            "sign_words": torch.empty((48, 4, 2048), device="cuda", dtype=torch.int32),
            "scale": torch.empty((48, 2048), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(64, 6144, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == [("u32_fixed", fixed_config)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_sm80_decode_2048x2048_to_k2048_specialized_u32_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[tuple[str, dict[str, int] | None]] = []

    fixed_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 2,
    }

    def _fake_u32_k2048_fixed(x, sign_words, scale, *, fixed_config):
        calls.append(("u32_k2048_fixed", fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_u32_fixed(x, sign_words, scale, *, fixed_config):
        calls.append(("u32_fixed", fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_u32_k2048_fixed_matmul", _fake_u32_k2048_fixed)
    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_u32_fixed_matmul", _fake_u32_fixed)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=2048,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": True,
            "fixed_decode_config": None,
            "sign_bytes": torch.empty((16, 16, 2048), device="cuda", dtype=torch.uint8),
            "sign_words": torch.empty((16, 4, 2048), device="cuda", dtype=torch.int32),
            "scale": torch.empty((16, 2048), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == [("u32_k2048_fixed", fixed_config)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_sm80_decode_6144x2048_to_u32_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[tuple[str, dict[str, int] | None]] = []

    fixed_config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 16,
        "num_warps": 4,
        "num_stages": 4,
    }

    def _fake_u32_fixed(x, sign_words, scale, *, fixed_config):
        calls.append(("u32_fixed", fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_byte(x, sign_bytes, scale):
        calls.append(("byte", None))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_u32_fixed_matmul", _fake_u32_fixed)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_matmul", _fake_byte)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=6144,
        out_features=2048,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": True,
            "fixed_decode_config": None,
            "sign_bytes": torch.empty((48, 16, 2048), device="cuda", dtype=torch.uint8),
            "sign_words": torch.empty((48, 4, 2048), device="cuda", dtype=torch.int32),
            "scale": torch.empty((48, 2048), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(1, 6144, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == [("u32_fixed", fixed_config)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_decode_2048x1024_to_k2048_specialized_byte_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[str] = []

    def _fake_k2048(x, sign_bytes, scale):
        calls.append("k2048")
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_generic(x, sign_bytes, scale):
        calls.append("generic")
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_k2048_matmul", _fake_k2048)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_matmul", _fake_generic)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=1024,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": False,
            "fixed_decode_config": None,
            "sign_bytes": torch.empty((16, 16, 1024), device="cuda", dtype=torch.uint8),
            "scale": torch.empty((16, 1024), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == ["k2048"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_out_of_range_sm80_down_proj_to_byte_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[str] = []

    def _fake_byte(x, sign_bytes, scale):
        calls.append("byte")
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_matmul", _fake_byte)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=6144,
        out_features=2048,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": True,
            "fixed_decode_config": None,
            "sign_bytes": torch.empty((48, 16, 2048), device="cuda", dtype=torch.uint8),
            "sign_words": torch.empty((48, 4, 2048), device="cuda", dtype=torch.int32),
            "scale": torch.empty((48, 2048), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(4, 6144, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == ["byte"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_non_decode_2048x1024_to_generic_byte_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[str] = []

    def _fake_k2048(x, sign_bytes, scale):
        calls.append("k2048")
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_generic(x, sign_bytes, scale):
        calls.append("generic")
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_k2048_matmul", _fake_k2048)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_matmul", _fake_generic)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=1024,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": False,
            "fixed_decode_config": None,
            "sign_bytes": torch.empty((16, 16, 1024), device="cuda", dtype=torch.uint8),
            "scale": torch.empty((16, 1024), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == ["generic"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_sm89_decode_2048x1024_to_fixed_byte_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[tuple[str, dict[str, int] | None]] = []

    fixed_config = {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 32,
        "num_warps": 8,
        "num_stages": 2,
    }

    def _fake_fixed(x, sign_bytes, scale, *, fixed_config):
        calls.append(("fixed", fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_generic(x, sign_bytes, scale):
        calls.append(("generic", None))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_k2048_fixed_matmul", _fake_fixed)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_k2048_matmul", _fake_generic)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=1024,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": False,
            "fixed_decode_config": fixed_config,
            "sign_bytes": torch.empty((16, 16, 1024), device="cuda", dtype=torch.uint8),
            "scale": torch.empty((16, 1024), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == [("fixed", fixed_config)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_sm89_decode_2048x2048_to_exact_fixed_byte_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[tuple[str, dict[str, int] | None]] = []

    fixed_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "num_warps": 4,
        "num_stages": 2,
    }

    def _fake_fixed(x, sign_bytes, scale, *, fixed_config):
        calls.append(("fixed", fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_generic(x, sign_bytes, scale):
        calls.append(("generic", None))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_k2048_fixed_matmul", _fake_fixed)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_k2048_matmul", _fake_generic)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=2048,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": False,
            "fixed_decode_config": fixed_config,
            "sign_bytes": torch.empty((16, 16, 2048), device="cuda", dtype=torch.uint8),
            "scale": torch.empty((16, 2048), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == [("fixed", fixed_config)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_sm89_decode_2048x6144_to_exact_fixed_byte_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[tuple[str, dict[str, int] | None]] = []

    fixed_config = {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 64,
        "num_warps": 8,
        "num_stages": 2,
    }

    def _fake_fixed(x, sign_bytes, scale, *, fixed_config):
        calls.append(("fixed", fixed_config))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    def _fake_generic(x, sign_bytes, scale):
        calls.append(("generic", None))
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(gguf_triton, "_launch_q1_0_g128_k2048_fixed_matmul", _fake_fixed)
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_k2048_matmul", _fake_generic)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=6144,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": False,
            "fixed_decode_config": fixed_config,
            "sign_bytes": torch.empty((16, 16, 6144), device="cuda", dtype=torch.uint8),
            "scale": torch.empty((16, 6144), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == [("fixed", fixed_config)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF Triton routing test")
def test_gguf_triton_routes_sm80_decode_2048x2048_out_of_range_to_byte_path(monkeypatch):
    pytest.importorskip("triton")

    from gptqmodel.nn_modules.qlinear import gguf_triton

    calls: list[str] = []

    def _fake_byte(x, sign_bytes, scale):
        calls.append("byte")
        return torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(gguf_triton, "_cuda_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(gguf_triton, "fused_q1_0_g128_matmul", _fake_byte)

    module = GGUFTritonKernel(
        bits="q1_0_g128",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=2048,
        bias=False,
        register_buffers=True,
    ).to("cuda").eval()
    monkeypatch.setattr(
        module,
        "_get_triton_cache",
        lambda _device: {
            "use_u32": True,
            "fixed_decode_config": None,
            "sign_bytes": torch.empty((16, 16, 2048), device="cuda", dtype=torch.uint8),
            "sign_words": torch.empty((16, 4, 2048), device="cuda", dtype=torch.int32),
            "scale": torch.empty((16, 2048), device="cuda", dtype=torch.float16),
        },
    )

    x = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
    module._forward_triton(x)

    assert calls == ["byte"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GGUF fused K routing test")
def test_gguf_forward_uses_fused_k_path_for_small_cuda_batches():
    case = _build_rtn_microbench_case(torch.float16, bits="q4_k_m")
    module = _build_rtn_gguf_module(case).to(case["device"]).eval()
    calls = {"dense": 0, "fused": 0}

    module.gguf_fused_cuda_max_rows = case["inputs"].shape[0]
    module.gguf_fused_cuda_min_matrix_elements = 0
    module.autotune_enabled = False
    module.clear_autotune()
    module.clear_weight_cache()

    def _dense(self, x_flat):
        calls["dense"] += 1
        return torch.zeros((x_flat.shape[0], self.out_features), device=x_flat.device, dtype=x_flat.dtype)

    def _fused(self, x_flat):
        calls["fused"] += 1
        return torch.zeros((x_flat.shape[0], self.out_features), device=x_flat.device, dtype=x_flat.dtype)

    module._forward_dequant_matmul = _dense.__get__(module, GGUFTorchLinear)
    module._forward_fused_k = _fused.__get__(module, GGUFTorchLinear)

    module(case["inputs"])
    assert calls == {"dense": 0, "fused": 1}

    module.gguf_fused_cuda_max_rows = 1
    calls = {"dense": 0, "fused": 0}
    module(case["inputs"])
    assert calls == {"dense": 1, "fused": 0}


@pytest.mark.parametrize("bits", ["q4_k_m", "q5_k_m", "q6_k"])
def test_gguf_cpu_fused_forward_matches_dense_baseline(bits: str):
    case = _build_rtn_microbench_case(torch.bfloat16, bits=bits)
    module = _build_rtn_gguf_module(case).cpu().eval()
    inputs = case["inputs"].detach().cpu().to(torch.bfloat16)

    module.gguf_fused_cpu_max_rows = inputs.shape[0]
    module.gguf_fused_cpu_min_matrix_elements = 0
    module.clear_weight_cache()

    baseline = module._forward_dequant_matmul(inputs)
    fused = module._forward_fused_k(inputs)

    output_stats = _error_stats(baseline.to(torch.float32), fused.to(torch.float32))
    assert output_stats["mae"] < 2e-4
    assert output_stats["max"] < 3e-3


def test_gguf_forward_uses_fused_k_path_for_small_cpu_batches():
    case = _build_rtn_microbench_case(torch.bfloat16, bits="q4_k_m")
    module = _build_rtn_gguf_module(case).cpu().eval()
    inputs = case["inputs"].detach().cpu().to(torch.bfloat16)
    calls = {"dense": 0, "fused": 0}

    module.gguf_fused_cpu_max_rows = inputs.shape[0]
    module.gguf_fused_cpu_min_matrix_elements = 0
    module.autotune_enabled = False
    module.clear_autotune()
    module.clear_weight_cache()

    def _dense(self, x_flat):
        calls["dense"] += 1
        return torch.zeros((x_flat.shape[0], self.out_features), device=x_flat.device, dtype=x_flat.dtype)

    def _fused(self, x_flat):
        calls["fused"] += 1
        return torch.zeros((x_flat.shape[0], self.out_features), device=x_flat.device, dtype=x_flat.dtype)

    module._forward_dequant_matmul = _dense.__get__(module, GGUFTorchLinear)
    module._forward_fused_k = _fused.__get__(module, GGUFTorchLinear)

    module(inputs)
    assert calls == {"dense": 0, "fused": 1}

    module.gguf_fused_cpu_max_rows = 1
    calls = {"dense": 0, "fused": 0}
    module(inputs)
    assert calls == {"dense": 1, "fused": 0}


def test_gguf_forward_autotunes_once_per_instance_with_fused_plan_on_cpu(monkeypatch):
    case = _build_rtn_microbench_case(torch.bfloat16, bits="q4_k_m")
    module = _build_rtn_gguf_module(case).cpu().eval()
    inputs = case["inputs"].detach().cpu().to(torch.bfloat16)

    module.gguf_fused_cpu_max_rows = inputs.shape[0]
    module.gguf_fused_cpu_min_matrix_elements = 0
    module.autotune_enabled = True
    module.clear_autotune()
    module.clear_weight_cache()

    calls = {"dense": 0, "fused": 0}

    def _dense(self, x_flat):
        calls["dense"] += 1
        return 2.0

    def _fused(self, x_flat):
        calls["fused"] += 1
        return 1.0

    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_dense_forward", _dense)
    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_fused_forward", _fused)

    module(inputs)

    assert calls == {"dense": 1, "fused": 1}
    assert module.get_autotune_result() is True

    def _fail(self, x_flat):
        raise AssertionError("autotune benchmark should not rerun for cached fused plan")

    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_dense_forward", _fail)
    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_fused_forward", _fail)

    module(inputs)
    assert module.get_autotune_result() is True


def test_gguf_forward_autotunes_once_per_instance_with_dense_plan_on_cpu(monkeypatch):
    case = _build_rtn_microbench_case(torch.bfloat16, bits="q4_k_m")
    module = _build_rtn_gguf_module(case).cpu().eval()
    inputs = case["inputs"].detach().cpu().to(torch.bfloat16)

    module.gguf_fused_cpu_max_rows = inputs.shape[0]
    module.gguf_fused_cpu_min_matrix_elements = 0
    module.autotune_enabled = True
    module.clear_autotune()
    module.clear_weight_cache()

    calls = {"dense": 0, "fused": 0}

    def _dense(self, x_flat):
        calls["dense"] += 1
        return 1.0

    def _fused(self, x_flat):
        calls["fused"] += 1
        return 2.0

    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_dense_forward", _dense)
    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_fused_forward", _fused)

    module(inputs)

    assert calls == {"dense": 1, "fused": 1}
    assert module.get_autotune_result() is False

    def _fail(self, x_flat):
        raise AssertionError("autotune benchmark should not rerun for cached dense plan")

    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_dense_forward", _fail)
    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_fused_forward", _fail)

    module(inputs)
    assert module.get_autotune_result() is False


def test_gguf_forward_autotunes_once_per_module_instance(monkeypatch):
    case = _build_rtn_microbench_case(torch.bfloat16, bits="q4_k_m")
    inputs = case["inputs"].detach().cpu().to(torch.bfloat16)
    module_a = _build_rtn_gguf_module(case).cpu().eval()
    module_b = _build_rtn_gguf_module(case).cpu().eval()

    for module in (module_a, module_b):
        module.gguf_fused_cpu_max_rows = inputs.shape[0]
        module.gguf_fused_cpu_min_matrix_elements = 0
        module.autotune_enabled = True
        module.clear_autotune()
        module.clear_weight_cache()

    calls = {"dense": 0, "fused": 0}

    def _dense(self, x_flat):
        calls["dense"] += 1
        return 2.0

    def _fused(self, x_flat):
        calls["fused"] += 1
        return 1.0

    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_dense_forward", _dense)
    monkeypatch.setattr(GGUFTorchLinear, "_benchmark_fused_forward", _fused)

    module_a(inputs)

    assert calls == {"dense": 1, "fused": 1}
    assert module_a.get_autotune_result() is True

    module_b(inputs)

    assert calls == {"dense": 2, "fused": 2}
    assert module_b.get_autotune_result() is True


@pytest.mark.parametrize(
    ("bits", "tensor_qtype"),
    [
        ("q4_k_s", "Q4_K"),
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

    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    case = _build_rtn_microbench_case(dtype, bits=bits)
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
        ("q4_k_s", 0.0015, 7e-4, 0.015),
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
        ("q4_k_s", 7e-4, 0.015),
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

    reloaded = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
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


def test_q4_k_s_and_q4_k_m_export_identical_tensor_bytes():
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    case = _build_rtn_microbench_case(dtype, bits="q4_k_s")
    module_s = _build_rtn_gguf_module(case)

    case_m = dict(case)
    case_m["bits"] = GGUFBits.from_string("q4_k_m")
    module_m = _build_rtn_gguf_module(case_m)

    assert module_s.gguf_tensor_qtype == "Q4_K"
    assert module_m.gguf_tensor_qtype == "Q4_K"
    np.testing.assert_array_equal(
        module_s.qweight.detach().cpu().numpy(),
        module_m.qweight.detach().cpu().numpy(),
    )
    torch.testing.assert_close(
        module_s.dequantize_weight(),
        module_m.dequantize_weight(),
        atol=0.0,
        rtol=0.0,
    )
