import gc
import os
import statistics
import time

import pytest
import torch
from tabulate import tabulate

import gptqmodel.quantization.dtype as dtype_module
from gptqmodel.quantization.dtype import (
    available_float8_dtype_names,
    dequantize_f4_e2m1,
    dequantize_f8_e4m3,
    device_supports_native_fp8,
)


def _print_accuracy(title: str, rows, headers) -> None:
    table = tabulate(rows, headers=headers, floatfmt=".6f")
    print(f"\n{title}\n{table}\n")


def _print_benchmark(title: str, rows, headers) -> None:
    table = tabulate(rows, headers=headers, floatfmt=".4f", tablefmt="github")
    print(f"\n{title}\n{table}\n")


try:  # pragma: no cover - optional dependency
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, nvfp4_quantize
except Exception:  # pragma: no cover
    NVFP4Tensor = None
    nvfp4_quantize = None

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover
    psutil = None


pytestmark = [pytest.mark.cpu, pytest.mark.gpu]


def _available_fp8_formats() -> list[str]:
    return list(available_float8_dtype_names())


def _rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    with open("/proc/self/statm", "r", encoding="utf-8") as fh:
        pages = int(fh.readline().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def _tensor_mib(tensor: torch.Tensor) -> float:
    return float(tensor.numel() * tensor.element_size()) / (1024.0 * 1024.0)


def _benchmark_cpu_impl(fn, *, warmup: int = 1, iters: int = 4):
    for _ in range(warmup):
        tmp = fn()
        del tmp
    gc.collect()

    samples_ms: list[float] = []
    rss_deltas: list[float] = []
    for _ in range(iters):
        gc.collect()
        rss_before = _rss_bytes()
        start = time.perf_counter()
        out = fn()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        rss_after = _rss_bytes()
        samples_ms.append(elapsed_ms)
        rss_deltas.append(max(0, rss_after - rss_before) / (1024.0 * 1024.0))
        del out

    gc.collect()
    result = fn()
    stats = {
        "median_ms": float(statistics.median(samples_ms)),
        "rss_delta_mib": float(max(rss_deltas) if rss_deltas else 0.0),
        "output_mib": _tensor_mib(result),
    }
    return result, stats


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
    expected = nv_tensor.dequantize(torch.bfloat16)

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


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dequantize_fp8_cpu_ab_benchmark_table(target_dtype: torch.dtype):
    rows = 256
    cols = 384
    bench_rows = []

    for fmt_name in _available_fp8_formats():
        fmt = getattr(torch, fmt_name)
        src = torch.rand(rows, cols, dtype=torch.float32) if "e8m0" in fmt_name else torch.randn(rows, cols, dtype=torch.float32)
        packed = src.to(fmt)
        scale_inv = torch.rand(rows // 64, cols // 64, dtype=torch.float32) * 0.5

        ref_fn = lambda: dtype_module._dequantize_f8_reference(  # noqa: E731
            packed,
            scale_inv=scale_inv,
            axis=None,
            target_dtype=target_dtype,
        )
        fast_fn = lambda: dequantize_f8_e4m3(  # noqa: E731
            packed,
            scale_inv=scale_inv,
            axis=None,
            target_dtype=target_dtype,
        )

        ref, ref_stats = _benchmark_cpu_impl(ref_fn)
        fast, fast_stats = _benchmark_cpu_impl(fast_fn)
        diff = float(torch.max(torch.abs(ref.to(torch.float32) - fast.to(torch.float32))).item())

        input_mib = _tensor_mib(packed) + _tensor_mib(scale_inv)
        ref_throughput = (input_mib + ref_stats["output_mib"]) / max(ref_stats["median_ms"] / 1e3, 1e-9)
        fast_throughput = (input_mib + fast_stats["output_mib"]) / max(fast_stats["median_ms"] / 1e3, 1e-9)

        bench_rows.append([
            fmt_name,
            str(target_dtype).split(".")[-1],
            "reference",
            ref_stats["median_ms"],
            ref_throughput,
            ref_stats["output_mib"],
            ref_stats["rss_delta_mib"],
            0.0,
        ])
        bench_rows.append([
            fmt_name,
            str(target_dtype).split(".")[-1],
            "native",
            fast_stats["median_ms"],
            fast_throughput,
            fast_stats["output_mib"],
            fast_stats["rss_delta_mib"],
            diff,
        ])

        assert torch.equal(fast, ref)

    _print_benchmark(
        f"fp8_cpu_ab_{str(target_dtype).split('.')[-1]}",
        bench_rows,
        [
            "format",
            "target",
            "impl",
            "median ms",
            "throughput MiB/s",
            "output MiB",
            "rss delta MiB",
            "max|diff|",
        ],
    )


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dequantize_fp4_cpu_ab_benchmark_table(target_dtype: torch.dtype):
    data = torch.randn(256, 384, dtype=torch.float32)
    scales, packed = nvfp4_quantize(data, block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2) if hasattr(torch, "float4_e2m1fn_x2") else None

    ref_fn = lambda: dtype_module._dequantize_f4_reference(  # noqa: E731
        packed,
        scale=scales,
        axis=None,
        target_dtype=target_dtype,
    )
    fast_uint8_fn = lambda: dequantize_f4_e2m1(  # noqa: E731
        packed,
        scale=scales,
        axis=None,
        target_dtype=target_dtype,
    )

    ref, ref_stats = _benchmark_cpu_impl(ref_fn)
    fast_uint8, fast_uint8_stats = _benchmark_cpu_impl(fast_uint8_fn)
    bench_rows = []
    input_mib = _tensor_mib(packed) + _tensor_mib(scales)

    for impl_name, tensor_result, stats in (
        ("reference:uint8", ref, ref_stats),
        ("native:uint8", fast_uint8, fast_uint8_stats),
    ):
        throughput = (input_mib + stats["output_mib"]) / max(stats["median_ms"] / 1e3, 1e-9)
        diff = 0.0 if impl_name.startswith("reference") else float(torch.max(torch.abs(ref.to(torch.float32) - tensor_result.to(torch.float32))).item())
        bench_rows.append([
            str(target_dtype).split(".")[-1],
            impl_name,
            stats["median_ms"],
            throughput,
            stats["output_mib"],
            stats["rss_delta_mib"],
            diff,
        ])

    assert torch.allclose(fast_uint8, ref, atol=1e-3, rtol=1e-3)

    if packed_float4 is not None:
        fast_float4_fn = lambda: dequantize_f4_e2m1(  # noqa: E731
            packed_float4,
            scale=scales,
            axis=None,
            target_dtype=target_dtype,
        )
        fast_float4, fast_float4_stats = _benchmark_cpu_impl(fast_float4_fn)
        throughput = (input_mib + fast_float4_stats["output_mib"]) / max(fast_float4_stats["median_ms"] / 1e3, 1e-9)
        diff = float(torch.max(torch.abs(ref.to(torch.float32) - fast_float4.to(torch.float32))).item())
        bench_rows.append([
            str(target_dtype).split(".")[-1],
            "native:float4_x2",
            fast_float4_stats["median_ms"],
            throughput,
            fast_float4_stats["output_mib"],
            fast_float4_stats["rss_delta_mib"],
            diff,
        ])
        assert torch.allclose(fast_float4, ref, atol=1e-3, rtol=1e-3)

    _print_benchmark(
        f"fp4_cpu_ab_{str(target_dtype).split('.')[-1]}",
        bench_rows,
        [
            "target",
            "impl",
            "median ms",
            "throughput MiB/s",
            "output MiB",
            "rss delta MiB",
            "max|diff|",
        ],
    )
