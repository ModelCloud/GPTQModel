import time

import pytest
import torch
from tabulate import tabulate

from gptqmodel.quantization.dtype import (
    dequantize_f4_e2m1,
    dequantize_f8_e4m3,
    device_supports_native_fp8,
)


def _print_accuracy(title: str, rows, headers) -> None:
    table = tabulate(rows, headers=headers, floatfmt=".6f")
    print(f"\n{title}\n{table}\n")


try:  # pragma: no cover - optional dependency
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, nvfp4_quantize
except Exception:  # pragma: no cover
    NVFP4Tensor = None
    nvfp4_quantize = None


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_basic_conversion():
    values = torch.linspace(-1, 1, steps=8, dtype=torch.float32)
    tensor = values.to(torch.float8_e4m3fn)

    result = dequantize_f8_e4m3(tensor)

    assert result.dtype is torch.bfloat16
    assert torch.equal(result, tensor.to(torch.bfloat16))


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_with_scale_inv():
    src = torch.randn(4, 6, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale_inv = torch.arange(1, 5, dtype=torch.float32)

    got = dequantize_f8_e4m3(fp8, scale_inv=scale_inv, axis=0)

    expected = (fp8.to(torch.bfloat16) / scale_inv.view(-1, 1).to(torch.bfloat16)).to(torch.bfloat16)
    diff = torch.max(torch.abs(got - expected)).item()
    _print_accuracy(
        "dequantize_f8_e4m3_with_scale_inv",
        [
            ["baseline", str(expected.dtype), float(expected.abs().max().item()), 0.0],
            ["candidate", str(got.dtype), float(got.abs().max().item()), diff],
        ],
        ["variant", "dtype", "max|value|", "max|diff vs baseline|"],
    )
    assert torch.equal(got, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_with_scale_axis_one():
    src = torch.randn(3, 5, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale = torch.linspace(0.5, 1.5, steps=5, dtype=torch.float32)

    got = dequantize_f8_e4m3(fp8, scale=scale, axis=1)

    expected = (fp8.to(torch.bfloat16) * scale.view(1, -1)).to(torch.bfloat16)
    diff = torch.max(torch.abs(got - expected)).item()
    _print_accuracy(
        "dequantize_f8_e4m3_with_scale_axis_one",
        [
            ["baseline", str(expected.dtype), float(expected.abs().max().item()), 0.0],
            ["candidate", str(got.dtype), float(got.abs().max().item()), diff],
        ],
        ["variant", "dtype", "max|value|", "max|diff vs baseline|"],
    )
    assert torch.equal(got, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_with_fractional_scale_inv():
    src = torch.randn(4, 4, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale_inv = torch.full((4,), 1 / 4, dtype=torch.float32)

    got = dequantize_f8_e4m3(fp8, scale_inv=scale_inv, axis=0)

    expected = (fp8.to(torch.bfloat16) * scale_inv.view(-1, 1).to(torch.bfloat16)).to(torch.bfloat16)
    diff = torch.max(torch.abs(got - expected)).item()
    _print_accuracy(
        "dequantize_f8_e4m3_with_fractional_scale_inv",
        [
            ["baseline", str(expected.dtype), float(expected.abs().max().item()), 0.0],
            ["candidate", str(got.dtype), float(got.abs().max().item()), diff],
        ],
        ["variant", "dtype", "max|value|", "max|diff vs baseline|"],
    )
    assert torch.equal(got, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_raises_on_both_scale_and_inverse():
    tensor = torch.zeros(2, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        dequantize_f8_e4m3(tensor, scale=torch.ones(2), scale_inv=torch.ones(2))


def test_device_supports_native_fp8_reports_capability(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (9, 0))
    assert device_supports_native_fp8(torch.device("cuda", 0)) is True

    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 0))
    assert device_supports_native_fp8(torch.device("cuda", 0)) is False


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
def test_dequantize_f4_e2m1_matches_nvfp4tensor():
    torch.manual_seed(0)
    data = torch.randn(128, 256, dtype=torch.float32)
    scales, packed = nvfp4_quantize(data, block_size=16)

    dequant = dequantize_f4_e2m1(packed, scale=scales, axis=None, target_dtype=torch.bfloat16)
    nv_tensor = NVFP4Tensor(packed, scales, block_size=16, orig_dtype=torch.bfloat16)
    expected = nv_tensor.to_dtype(torch.bfloat16)

    diff = torch.max(torch.abs(dequant - expected)).item()
    _print_accuracy(
        "dequantize_f4_e2m1_matches_nvfp4tensor",
        [
            ["NVFP4Tensor", str(expected.dtype), float(expected.abs().max().item()), 0.0],
            ["dtype_impl", str(dequant.dtype), float(dequant.abs().max().item()), diff],
        ],
        ["variant", "dtype", "max|value|", "max|diff vs baseline|"],
    )
    assert torch.allclose(dequant, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_dequantize_f4_e2m1_cpu_vs_gpu():
    torch.manual_seed(1)
    data = torch.randn(128, 512, dtype=torch.float32)
    scales, packed = nvfp4_quantize(data, block_size=16)

    cpu = dequantize_f4_e2m1(packed, scale=scales, axis=None, target_dtype=torch.bfloat16)

    packed_gpu = packed.cuda()
    scales_gpu = scales.cuda()

    start = time.perf_counter()
    gpu = dequantize_f4_e2m1(packed_gpu, scale=scales_gpu, axis=None, target_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    diff = torch.max(torch.abs(cpu - gpu.cpu())).item()
    _print_accuracy(
        "dequantize_f4_e2m1_cpu_vs_gpu",
        [
            ["CPU", 0.0, float(cpu.abs().max().item()), 0.0],
            ["GPU", gpu_time, float(gpu.abs().max().item()), diff],
        ],
        ["variant", "time (s)", "max|value|", "max|diff vs baseline|"],
    )
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for GPU benchmark",
)
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_cpu_vs_gpu_benchmark():
    rows = 128 * 4
    cols = 128 * 3
    src = torch.randn(rows, cols, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale_inv = torch.rand(rows // 128, cols // 128, dtype=torch.float32) * 0.5

    # Warmup
    dequantize_f8_e4m3(fp8, scale_inv=scale_inv, axis=None, target_dtype=torch.bfloat16)

    start = time.perf_counter()
    cpu_result = dequantize_f8_e4m3(fp8, scale_inv=scale_inv, axis=None, target_dtype=torch.bfloat16)
    cpu_time = time.perf_counter() - start

    fp8_gpu = fp8.cuda()
    scale_inv_gpu = scale_inv.cuda()

    dequantize_f8_e4m3(fp8_gpu, scale_inv=scale_inv_gpu, axis=None, target_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    start = time.perf_counter()
    gpu_result = dequantize_f8_e4m3(fp8_gpu, scale_inv=scale_inv_gpu, axis=None, target_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    diff = torch.max(torch.abs(cpu_result - gpu_result.cpu())).item()
    _print_accuracy(
        "dequantize_f8_e4m3_cpu_vs_gpu",
        [
            ["CPU", cpu_time, float(cpu_result.abs().max().item()), 0.0],
            ["GPU", gpu_time, float(gpu_result.abs().max().item()), diff],
        ],
        ["variant", "time (s)", "max|value|", "max|diff vs baseline|"],
    )
    assert torch.allclose(cpu_result, gpu_result.cpu(), atol=1e-3, rtol=1e-3)

    # GPU should not be dramatically slower than CPU
    assert gpu_time <= cpu_time * 2, f"GPU dequant slower than expected (cpu={cpu_time:.4f}s, gpu={gpu_time:.4f}s)"
