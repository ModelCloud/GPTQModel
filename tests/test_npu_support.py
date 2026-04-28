import copy
import os
import warnings

import pytest
import torch
import torch.nn as nn

from gptqmodel.models._const import DEVICE, normalize_device
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch import _right_shift_unpack
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import unpack_awq
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import auto_select_device, get_kernel_for_backend, select_quant_linear
from gptqmodel.utils.torch import HAS_NPU


NPU_TEST_DEVICE = os.environ.get("GPTQMODEL_TEST_NPU_DEVICE", "npu:0")
NPU_CPU_FALLBACK_MARKERS = (
    "not currently supported on the NPU backend",
    "fall back to run on the CPU",
)


def _test_npu_device() -> torch.device:
    return torch.device(NPU_TEST_DEVICE)


def _assert_no_npu_cpu_fallback(caught: list[warnings.WarningMessage]) -> None:
    fallback_warnings = [
        str(warning.message)
        for warning in caught
        if any(marker in str(warning.message) for marker in NPU_CPU_FALLBACK_MARKERS)
    ]
    assert fallback_warnings == []


def _assert_npu_forward_matches_cpu(
    module: nn.Module,
    x_cpu: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    npu_module = copy.deepcopy(module).to(_test_npu_device()).eval()
    module = module.eval()

    with torch.inference_mode():
        y_cpu = module(x_cpu)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.inference_mode():
            y_npu = npu_module(x_cpu.to(_test_npu_device()))
            y_npu_cpu = y_npu.to("cpu", dtype=torch.float32)

    assert y_npu.device.type == "npu"
    _assert_no_npu_cpu_fallback(caught)
    torch.testing.assert_close(y_npu_cpu, y_cpu.to(torch.float32), atol=atol, rtol=rtol)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros((unpacked.shape[0], unpacked.shape[1] // pack_factor), dtype=torch.int32)
    for col in range(unpacked.shape[1] // pack_factor):
        for lane, order in enumerate(order_map):
            packed[:, col] |= unpacked[:, col * pack_factor + order].to(torch.int32) << (lane * bits)
    return packed


def _make_awq_like_module(cls, dtype: torch.dtype, *, seed: int = 300, **kwargs):
    in_features = kwargs.pop("in_features", 64)
    out_features = kwargs.pop("out_features", 64)
    group_size = kwargs.pop("group_size", 16)
    bias = kwargs.pop("bias", True)
    bits = 4

    torch.manual_seed(seed)
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = ((torch.rand(groups, out_features, dtype=torch.float32) * 2.0) + 0.25).to(dtype)

    module = cls(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype,
        register_buffers=True,
        **kwargs,
    )
    module.qweight.copy_(_pack_awq_tensor(int_weight, bits))
    module.qzeros.copy_(_pack_awq_tensor(zero_points, bits))
    module.scales.copy_(scales.to(module.scales.dtype))
    if bias:
        module.bias.copy_(torch.randn(out_features, dtype=dtype).to(module.bias.dtype))
    return module


def _make_gptq_module(bits: int, dtype: torch.dtype) -> TorchLinear:
    in_features = 64
    out_features = 64
    group_size = 16

    torch.manual_seed(100 + bits)
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight.data.normal_(0, 0.12)
    linear.bias.data.normal_(0, 0.03)

    maxq = (1 << bits) - 1
    groups = in_features // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float32) * 0.04 + 0.01
    zeros = torch.randint(0, maxq + 1, (out_features, groups), dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )
    module.pack_block(linear, scales, zeros, g_idx)
    module.optimized = True
    module.post_init()
    return module.to(dtype=dtype).eval()


def _make_awq_module(dtype: torch.dtype) -> AwqTorchLinear:
    module = _make_awq_like_module(AwqTorchLinear, dtype)
    module.post_init()
    return module.eval()


def _make_paro_module(dtype: torch.dtype) -> ParoLinear:
    module = _make_awq_like_module(ParoLinear, dtype, seed=350, krot=1)
    theta = torch.linspace(-0.15, 0.15, module.in_features // 2, dtype=module.theta.dtype).view_as(module.theta)
    channel_scales = torch.linspace(0.95, 1.05, module.in_features, dtype=module.channel_scales.dtype).view_as(
        module.channel_scales
    )
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    return module.eval()


def _make_gguf_module(bits: str, dtype: torch.dtype) -> GGUFTorchLinear:
    in_features = 256
    out_features = 32

    torch.manual_seed(500 + sum(ord(ch) for ch in bits))
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight.data.normal_(0, 0.12)
    linear.bias.data.normal_(0, 0.03)

    module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )
    module.pack_original(linear, scales=None, zeros=None, g_idx=None)
    module.post_init()
    return module.to(dtype=dtype).eval()


def _make_qqq_module(dtype: torch.dtype, group_size: int) -> QQQTorchLinear:
    in_features = 256
    out_features = 128

    torch.manual_seed(600 + group_size)
    linear = nn.Linear(in_features, out_features, bias=True).to(dtype=torch.float16)
    linear.weight.data.normal_(0, 0.08)
    linear.bias.data.normal_(0, 0.02)

    module = QQQTorchLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )

    if group_size == -1:
        scales = torch.rand(out_features, 1, dtype=torch.float16) * 0.04 + 0.01
        s_extra = None
    else:
        groups = in_features // group_size
        scales = torch.rand(out_features, groups, dtype=torch.float16) * 0.04 + 0.01
        s_extra = torch.rand(out_features, dtype=torch.float32) * 0.5 + 0.75

    module.pack(linear, scales, s_extra)
    module.post_init()
    return module.eval()


def _make_exllamav3_torch_module(*, device: torch.device | str = "cpu") -> ExllamaV3TorchLinear:
    in_features = 128
    out_features = 128
    bits = 2
    target = torch.device(device)

    generator = torch.Generator(device="cpu").manual_seed(700)
    tensors = {
        "trellis": torch.randint(
            -32768,
            32767,
            (in_features // 16, out_features // 16, bits * 16),
            dtype=torch.int16,
            generator=generator,
        ),
        "suh": torch.randint(0, 2, (in_features,), dtype=torch.int8, generator=generator)
        .to(torch.float16)
        .mul_(2)
        .sub_(1),
        "svh": torch.randint(0, 2, (out_features,), dtype=torch.int8, generator=generator)
        .to(torch.float16)
        .mul_(2)
        .sub_(1),
        "bias": torch.randn(out_features, dtype=torch.float16, generator=generator) * 0.01,
    }
    tensors = {
        key: value.to(target) if target.type != "cpu" else value
        for key, value in tensors.items()
    }
    return ExllamaV3TorchLinear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name="npu_exl3_torch",
        tensors=tensors,
    ).eval()


def test_npu_device_normalization():
    assert normalize_device("npu") is DEVICE.NPU
    assert normalize_device("npu:3") is DEVICE.NPU
    assert DEVICE.NPU.type == "npu"
    try:
        expected = torch.device("npu:0")
    except (RuntimeError, ValueError):
        pytest.skip("This PyTorch build does not register the npu device type")
    assert DEVICE.NPU.to_torch_device() == expected


def test_auto_select_device_uses_npu_when_available(monkeypatch):
    monkeypatch.setattr(importer, "HAS_CUDA", False)
    monkeypatch.setattr(importer, "HAS_XPU", False)
    monkeypatch.setattr(importer, "HAS_NPU", True)
    monkeypatch.setattr(importer, "HAS_MPS", False)

    assert auto_select_device(None, BACKEND.AUTO) is DEVICE.NPU


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_npu_auto_selects_torch_gptq(fmt):
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is TorchLinear


def test_npu_auto_selects_torch_awq_for_gemm():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is AwqTorchLinear


def test_npu_auto_selects_paroquant_torch_dense_fallback():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.PAROQUANT,
        quant_method=METHOD.PARO,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is ParoLinear


def test_npu_auto_selects_gguf_torch():
    qlinear_cls = select_quant_linear(
        bits="q4_k_m",
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFTorchLinear


def test_npu_auto_selects_qqq_torch():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.QQQ,
        quant_method=METHOD.QQQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is QQQTorchLinear


def test_qqq_torch_backend_selects_torch_kernel():
    assert get_kernel_for_backend(BACKEND.QQQ_TORCH, METHOD.QQQ, FORMAT.QQQ) is QQQTorchLinear


def test_npu_does_not_advertise_fp8_torch_until_cann_supports_float8():
    assert DEVICE.ALL not in TorchFP8Linear.SUPPORTS_DEVICES
    assert DEVICE.NPU not in TorchFP8Linear.SUPPORTS_DEVICES


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
def test_npu_awq_unpack_preserves_pack_dimension():
    qweight_cpu = torch.tensor(
        [[0, 1, -1], [-2147483648, 2147483647, -123456789]],
        dtype=torch.int32,
    )
    qzeros_cpu = torch.tensor(
        [[-1, 0, 123456789], [2147483647, -2147483648, 7]],
        dtype=torch.int32,
    )
    qweight = qweight_cpu.to("npu:0")
    qzeros = qzeros_cpu.to("npu:0")

    iweight, izeros = unpack_awq(qweight, qzeros, bits=4)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    expected_iweight = (qweight_cpu[:, :, None] >> shifts[None, None, :]).to(torch.int8).view(2, 24)
    expected_izeros = (qzeros_cpu[:, :, None] >> shifts[None, None, :]).to(torch.int8).view(2, 24)

    assert iweight.shape == (2, 24)
    assert izeros.shape == (2, 24)
    assert iweight.device.type == "npu"
    assert izeros.device.type == "npu"
    assert torch.equal(iweight.cpu(), expected_iweight)
    assert torch.equal(izeros.cpu(), expected_izeros)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
def test_npu_torch_gptq_unpack_preserves_pack_dimension():
    qweight_cpu = torch.tensor(
        [
            [0, 1, -1],
            [-2147483648, 2147483647, -123456789],
            [12345, -98765, 42],
            [-42, 98765, -12345],
        ],
        dtype=torch.int32,
    )
    qweight = qweight_cpu.to("npu:0")
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device="npu:0").view(1, 8, 1)

    unpacked = _right_shift_unpack(
        qweight.unsqueeze(1).expand(-1, 8, -1),
        shifts,
        torch.int8,
    )

    assert unpacked.shape == (4, 8, 3)
    assert unpacked.device.type == "npu"
    expected = (
        qweight_cpu.unsqueeze(1).expand(-1, 8, -1)
        >> torch.arange(0, 32, 4, dtype=torch.int32).view(1, 8, 1)
    ).to(torch.int8)
    assert torch.equal(unpacked.cpu(), expected)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_gptq_forward_matches_cpu(bits, dtype):
    module = _make_gptq_module(bits, dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_awq_forward_matches_cpu(dtype):
    module = _make_awq_module(dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_paro_forward_matches_cpu(dtype):
    module = _make_paro_module(dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=6e-3, rtol=6e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("bits", ["q1_0", "q4_0", "q4_k_m", "q5_k_m", "q6_k", "q8_0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_gguf_forward_matches_cpu_without_fallback(bits, dtype):
    module = _make_gguf_module(bits, dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=8e-3, rtol=8e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_qqq_forward_matches_cpu_without_fallback(group_size, dtype, capfd):
    module_cpu = _make_qqq_module(dtype, group_size)
    module_npu = copy.deepcopy(module_cpu).to(_test_npu_device()).eval()
    x_cpu = torch.randn(2, 3, module_cpu.in_features, dtype=dtype)

    with torch.inference_mode():
        y_cpu = module_cpu(x_cpu)

    capfd.readouterr()
    with torch.inference_mode():
        y_npu = module_npu(x_cpu.to(_test_npu_device()))
        torch.npu.synchronize()

    captured = capfd.readouterr()
    combined_output = captured.out + captured.err
    assert "AiCpu" not in combined_output
    assert not any(marker in combined_output for marker in NPU_CPU_FALLBACK_MARKERS)
    assert y_npu.device.type == "npu"
    torch.testing.assert_close(y_npu.cpu(), y_cpu, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
def test_npu_exllamav3_torch_forward_matches_cpu_without_aicpu_sort(capfd):
    module_cpu = _make_exllamav3_torch_module()
    module_npu = _make_exllamav3_torch_module(device=_test_npu_device())
    x_cpu = torch.randn(2, 3, module_cpu.in_features, dtype=torch.float16)

    with torch.inference_mode():
        y_cpu = module_cpu(x_cpu)

    capfd.readouterr()
    with torch.inference_mode():
        y_npu = module_npu(x_cpu.to(_test_npu_device()))
        torch.npu.synchronize()

    captured = capfd.readouterr()
    combined_output = captured.out + captured.err
    assert "ArgSort" not in combined_output
    assert "AiCpu" not in combined_output
    assert not any(marker in combined_output for marker in NPU_CPU_FALLBACK_MARKERS)
    torch.testing.assert_close(y_npu.cpu(), y_cpu, atol=5e-2, rtol=5e-2)
