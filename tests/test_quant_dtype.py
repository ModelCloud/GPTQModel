# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import statistics
import time
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from gptqmodel.quantization.dtype import (
    _DTYPE_SUPPORT_CACHE,
    _cpu_floatx_threads,
    _dequantize_f4_reference,
    _dequantize_f8_reference,
    available_float8_dtype_names,
    dequantize_f4_e2m1,
    dequantize_f8_e4m3,
    device_supports_dtype,
    device_supports_native_fp4,
    device_supports_native_fp8,
    get_device_dtype_support,
)
from gptqmodel.utils.logger import render_table


# Default to the preferred GLM FP8 checkpoint, but fall back to a real local FP8 model
# so CPU A/B runs stay realistic on machines where the original mount is absent.
_FLOATX_BENCH_ENV = os.environ.get("GPTQMODEL_FLOATX_BENCH_MODEL")
_FLOATX_BENCH_MODEL_CANDIDATES = (
    [Path(_FLOATX_BENCH_ENV)] if _FLOATX_BENCH_ENV else [
        Path("/monster/data/model/GLM-5.1-FP8"),
        Path("/root/model/DeepSeek-V3-0324"),
    ]
)
FLOATX_BENCH_MODEL_ROOT = next(
    (candidate for candidate in _FLOATX_BENCH_MODEL_CANDIDATES if candidate.exists()),
    _FLOATX_BENCH_MODEL_CANDIDATES[0],
)


def _print_accuracy(title: str, rows, headers) -> None:
    table = render_table(rows, headers=headers, floatfmt=".6f", tablefmt="grid")
    print(f"\n{title}\n{table}\n")


def _print_benchmark(title: str, rows, headers, note: str | None = None) -> None:
    table = render_table(rows, headers=headers, floatfmt=".4f", tablefmt="grid")
    note_block = f"{note}\n" if note else ""
    print(f"\n{title}\n{note_block}{table}\n")


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


def _benchmark_profile(numel: int) -> tuple[int, int]:
    # Huge realistic layers still need one untimed pass so lazy native setup does not skew A/B stats.
    if numel >= 64 * 1024 * 1024:
        return 1, 2
    return 1, 4


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


def _synthetic_fp8_benchmark_source() -> tuple[torch.Tensor, torch.Tensor, str, int, int]:
    rows = 256
    cols = 384
    src = torch.randn(rows, cols, dtype=torch.float32)
    scale_inv = torch.rand(rows // 64, cols // 64, dtype=torch.float32) * 0.5
    source = "source: synthetic fallback rows=256 cols=384 scale=[4, 6]"
    return src, scale_inv, source, rows, cols


@lru_cache(maxsize=1)
def _realistic_fp8_benchmark_spec() -> tuple[Path, str, str, int, int, tuple[int, ...]] | None:
    if not FLOATX_BENCH_MODEL_ROOT.exists():
        return None

    largest: tuple[int, Path, str, str, int, int, tuple[int, ...]] | None = None
    for path in sorted(FLOATX_BENCH_MODEL_ROOT.glob("model-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            tensor_keys = tuple(sorted(tensors.keys()))
            for key in tensor_keys:
                if not key.endswith(".weight"):
                    continue
                tensor_slice = tensors.get_slice(key)
                shape = tensor_slice.get_shape()
                if len(shape) != 2:
                    continue
                dtype_name = str(tensor_slice.get_dtype())
                if not dtype_name.startswith("F8_"):
                    continue
                scale_key = key.replace(".weight", ".weight_scale_inv")
                if scale_key not in tensor_keys:
                    continue
                area = int(shape[0]) * int(shape[1])
                if largest is None or area > largest[0]:
                    scale_shape = tuple(int(dim) for dim in tensors.get_slice(scale_key).get_shape())
                    largest = (area, path, key, scale_key, int(shape[0]), int(shape[1]), scale_shape)

    if largest is None:
        return None

    _, path, key, scale_key, rows, cols, scale_shape = largest
    return path, key, scale_key, rows, cols, scale_shape


@lru_cache(maxsize=1)
def _realistic_fp8_benchmark_source() -> tuple[torch.Tensor, torch.Tensor, str, int, int]:
    spec = _realistic_fp8_benchmark_spec()
    if spec is None:
        return _synthetic_fp8_benchmark_source()

    path, weight_key, scale_key, rows, cols, scale_shape = spec
    with safe_open(path, framework="pt", device="cpu") as tensors:
        packed = tensors.get_tensor(weight_key)
        scale_inv = tensors.get_tensor(scale_key)

    # Reuse the checkpoint's real FP8 distribution as the float source for both FP8 and FP4 decode tables.
    src = _dequantize_f8_reference(
        packed,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    ).to(torch.float32)
    source = (
        f"source: {path.name}:{weight_key} rows={rows} cols={cols} "
        f"scale={list(scale_shape)} model_root={FLOATX_BENCH_MODEL_ROOT}"
    )
    return src, scale_inv, source, rows, cols


@lru_cache(maxsize=1)
def _realistic_fp8_native_benchmark_source() -> tuple[torch.Tensor, torch.Tensor, str, int, int]:
    spec = _realistic_fp8_benchmark_spec()
    if spec is None:
        src, scale_inv, source, rows, cols = _synthetic_fp8_benchmark_source()
        packed = src.to(torch.float8_e4m3fn)
        return packed, scale_inv, f"{source} native_format=float8_e4m3fn", rows, cols

    path, weight_key, scale_key, rows, cols, scale_shape = spec
    with safe_open(path, framework="pt", device="cpu") as tensors:
        packed = tensors.get_tensor(weight_key)
        scale_inv = tensors.get_tensor(scale_key)

    source = (
        f"source: {path.name}:{weight_key} rows={rows} cols={cols} "
        f"scale={list(scale_shape)} native_format={str(packed.dtype).split('.')[-1]} "
        f"model_root={FLOATX_BENCH_MODEL_ROOT}"
    )
    return packed, scale_inv, source, rows, cols


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
def test_dequantize_f8_cpu_prefers_reference_for_standard_fp8(monkeypatch: pytest.MonkeyPatch):
    src = torch.linspace(-1, 1, steps=16, dtype=torch.float32).reshape(4, 4)
    fp8 = src.to(torch.float8_e4m3fn)
    scale_inv = torch.ones_like(src, dtype=torch.float32)
    expected = _dequantize_f8_reference(
        fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    )

    def fail_load():
        raise AssertionError("native FP8 kernel should be bypassed for standard torch FP8 dtypes")

    monkeypatch.delenv("GPTQMODEL_FLOATX_CPU_FORCE_NATIVE_FP8", raising=False)
    monkeypatch.setattr("gptqmodel.quantization.dtype._load_floatx_cpu_ops", fail_load)

    got = dequantize_f8_e4m3(
        fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    )

    assert torch.equal(got, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_raises_on_both_scale_and_inverse():
    tensor = torch.zeros(2, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        dequantize_f8_e4m3(tensor, scale=torch.ones(2), scale_inv=torch.ones(2))


def test_device_dtype_support_reports_arch_mapping(monkeypatch):
    _DTYPE_SUPPORT_CACHE.clear()


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dequantize_f8_e4m3_disable_avx2_override_matches_reference(monkeypatch, target_dtype: torch.dtype):
    src = torch.randn(32, 64, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale_inv = torch.rand(2, 4, dtype=torch.float32) * 0.5

    monkeypatch.setenv("GPTQMODEL_FLOATX_CPU_DISABLE_AVX2", "1")
    got = dequantize_f8_e4m3(fp8, scale_inv=scale_inv, axis=None, target_dtype=target_dtype)
    expected = _dequantize_f8_reference(
        fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=target_dtype,
    )

    assert torch.equal(got, expected)

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 9))

    support = get_device_dtype_support(torch.device("cuda", 0))

    assert support.capability == (8, 9)
    assert torch.float16 in support.advertised_linear_dtypes
    assert torch.float32 in support.advertised_linear_dtypes
    assert torch.bfloat16 in support.advertised_linear_dtypes
    assert torch.float8_e4m3fn in support.advertised_linear_dtypes


def test_device_supports_native_fp8_reports_capability(monkeypatch):
    _DTYPE_SUPPORT_CACHE.clear()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 9))
    assert device_supports_native_fp8(torch.device("cuda", 0)) is True
    assert device_supports_dtype(torch.device("cuda", 0), torch.float8_e4m3fn) is True

    _DTYPE_SUPPORT_CACHE.clear()
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 0))
    assert device_supports_native_fp8(torch.device("cuda", 0)) is False
    assert device_supports_dtype(torch.device("cuda", 0), torch.float8_e4m3fn) is False


@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="float4 packed dtype not available")
def test_device_supports_native_fp4_reports_capability(monkeypatch):
    _DTYPE_SUPPORT_CACHE.clear()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (10, 0))
    support = get_device_dtype_support(torch.device("cuda", 0))
    assert torch.float4_e2m1fn_x2 in support.advertised_linear_dtypes
    assert device_supports_native_fp4(torch.device("cuda", 0)) is True
    assert device_supports_dtype(torch.device("cuda", 0), torch.float4_e2m1fn_x2) is True

    _DTYPE_SUPPORT_CACHE.clear()
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 9))
    support = get_device_dtype_support(torch.device("cuda", 0))
    assert torch.float4_e2m1fn_x2 not in support.advertised_linear_dtypes
    assert device_supports_native_fp4(torch.device("cuda", 0)) is False
    assert device_supports_dtype(torch.device("cuda", 0), torch.float4_e2m1fn_x2) is False


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
@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dequantize_f4_e2m1_disable_avx2_override_matches_reference(monkeypatch, target_dtype: torch.dtype):
    torch.manual_seed(4)
    data = torch.randn(32, 64, dtype=torch.float32)
    scales, packed = nvfp4_quantize(data, block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2) if hasattr(torch, "float4_e2m1fn_x2") else packed

    monkeypatch.setenv("GPTQMODEL_FLOATX_CPU_DISABLE_AVX2", "1")
    got = dequantize_f4_e2m1(packed_float4, scale=scales, axis=None, target_dtype=target_dtype)
    expected = _dequantize_f4_reference(
        packed,
        scale=scales,
        axis=None,
        target_dtype=target_dtype,
    )

    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3)


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
    src, scale_inv, source_note, rows, cols = _realistic_fp8_benchmark_source()
    warmup, iters = _benchmark_profile(rows * cols)
    bench_rows = []

    for fmt_name in _available_fp8_formats():
        fmt = getattr(torch, fmt_name)
        fmt_src = src.abs() if "e8m0" in fmt_name else src
        packed = fmt_src.to(fmt)

        ref_fn = lambda: _dequantize_f8_reference(  # noqa: E731
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

        ref, ref_stats = _benchmark_cpu_impl(ref_fn, warmup=warmup, iters=iters)
        fast, fast_stats = _benchmark_cpu_impl(fast_fn, warmup=warmup, iters=iters)
        diff = float(torch.max(torch.abs(ref.to(torch.float32) - fast.to(torch.float32))).item())

        input_mib = _tensor_mib(packed) + _tensor_mib(scale_inv)
        ref_throughput = (input_mib + ref_stats["output_mib"]) / max(ref_stats["median_ms"] / 1e3, 1e-9)
        fast_throughput = (input_mib + fast_stats["output_mib"]) / max(fast_stats["median_ms"] / 1e3, 1e-9)
        speedup = ref_stats["median_ms"] / max(fast_stats["median_ms"], 1e-9)
        throughput_gain_pct = ((fast_throughput - ref_throughput) / max(ref_throughput, 1e-9)) * 100.0

        bench_rows.append([
            fmt_name,
            str(target_dtype).split(".")[-1],
            ref_stats["median_ms"],
            fast_stats["median_ms"],
            speedup,
            ref_throughput,
            fast_throughput,
            throughput_gain_pct,
            diff,
        ])

        assert torch.equal(fast, ref)

    _print_benchmark(
        f"fp8_cpu_ab_{str(target_dtype).split('.')[-1]}",
        bench_rows,
        [
            "format",
            "target",
            "ref ms",
            "native ms",
            "speedup x",
            "ref MiB/s",
            "native MiB/s",
            "throughput delta %",
            "max|diff|",
        ],
        note=source_note,
    )


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dequantize_fp8_cpu_real_format_ab_benchmark_table(target_dtype: torch.dtype):
    packed, scale_inv, source_note, rows, cols = _realistic_fp8_native_benchmark_source()
    warmup, iters = _benchmark_profile(rows * cols)
    enable_large_threads = (
        target_dtype is torch.bfloat16 and
        hasattr(torch, "float8_e4m3fn") and
        packed.dtype is torch.float8_e4m3fn
    )

    ref_fn = lambda: _dequantize_f8_reference(  # noqa: E731
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

    ref, ref_stats = _benchmark_cpu_impl(ref_fn, warmup=warmup, iters=iters)
    fast, fast_stats = _benchmark_cpu_impl(fast_fn, warmup=warmup, iters=iters)
    diff = float(torch.max(torch.abs(ref.to(torch.float32) - fast.to(torch.float32))).item())

    input_mib = _tensor_mib(packed) + _tensor_mib(scale_inv)
    ref_throughput = (input_mib + ref_stats["output_mib"]) / max(ref_stats["median_ms"] / 1e3, 1e-9)
    fast_throughput = (input_mib + fast_stats["output_mib"]) / max(fast_stats["median_ms"] / 1e3, 1e-9)
    speedup = ref_stats["median_ms"] / max(fast_stats["median_ms"], 1e-9)
    throughput_gain_pct = ((fast_throughput - ref_throughput) / max(ref_throughput, 1e-9)) * 100.0

    assert torch.equal(fast, ref)

    _print_benchmark(
        f"fp8_cpu_real_format_ab_{str(target_dtype).split('.')[-1]}",
        [[
            str(packed.dtype).split(".")[-1],
            str(target_dtype).split(".")[-1],
            ref_stats["median_ms"],
            fast_stats["median_ms"],
            speedup,
            ref_throughput,
            fast_throughput,
            throughput_gain_pct,
            diff,
        ]],
        [
            "format",
            "target",
            "ref ms",
            "native ms",
            "speedup x",
            "ref MiB/s",
            "native MiB/s",
            "throughput delta %",
            "max|diff|",
        ],
        note=(
            f"{source_note} "
            f"threads={_cpu_floatx_threads(rows * cols, enable_large_threads=enable_large_threads)}"
        ),
    )


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_dequantize_fp4_cpu_ab_benchmark_table(target_dtype: torch.dtype):
    data, _, source_note, rows, cols = _realistic_fp8_benchmark_source()
    warmup, iters = _benchmark_profile(rows * cols)
    scales, packed = nvfp4_quantize(data, block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2) if hasattr(torch, "float4_e2m1fn_x2") else None

    ref_fn = lambda: _dequantize_f4_reference(  # noqa: E731
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

    ref, ref_stats = _benchmark_cpu_impl(ref_fn, warmup=warmup, iters=iters)
    fast_uint8, fast_uint8_stats = _benchmark_cpu_impl(fast_uint8_fn, warmup=warmup, iters=iters)
    bench_rows = []
    input_mib = _tensor_mib(packed) + _tensor_mib(scales)
    ref_throughput = (input_mib + ref_stats["output_mib"]) / max(ref_stats["median_ms"] / 1e3, 1e-9)
    fast_uint8_throughput = (input_mib + fast_uint8_stats["output_mib"]) / max(fast_uint8_stats["median_ms"] / 1e3, 1e-9)
    fast_uint8_diff = float(torch.max(torch.abs(ref.to(torch.float32) - fast_uint8.to(torch.float32))).item())
    bench_rows.append([
        str(target_dtype).split(".")[-1],
        "native:uint8",
        ref_stats["median_ms"],
        fast_uint8_stats["median_ms"],
        ref_stats["median_ms"] / max(fast_uint8_stats["median_ms"], 1e-9),
        ref_throughput,
        fast_uint8_throughput,
        ((fast_uint8_throughput - ref_throughput) / max(ref_throughput, 1e-9)) * 100.0,
        fast_uint8_diff,
    ])

    assert torch.allclose(fast_uint8, ref, atol=1e-3, rtol=1e-3)

    if packed_float4 is not None:
        fast_float4_fn = lambda: dequantize_f4_e2m1(  # noqa: E731
            packed_float4,
            scale=scales,
            axis=None,
            target_dtype=target_dtype,
        )
        fast_float4, fast_float4_stats = _benchmark_cpu_impl(fast_float4_fn, warmup=warmup, iters=iters)
        throughput = (input_mib + fast_float4_stats["output_mib"]) / max(fast_float4_stats["median_ms"] / 1e3, 1e-9)
        diff = float(torch.max(torch.abs(ref.to(torch.float32) - fast_float4.to(torch.float32))).item())
        bench_rows.append([
            str(target_dtype).split(".")[-1],
            "native:float4_x2",
            ref_stats["median_ms"],
            fast_float4_stats["median_ms"],
            ref_stats["median_ms"] / max(fast_float4_stats["median_ms"], 1e-9),
            ref_throughput,
            throughput,
            ((throughput - ref_throughput) / max(ref_throughput, 1e-9)) * 100.0,
            diff,
        ])
        assert torch.allclose(fast_float4, ref, atol=1e-3, rtol=1e-3)

    _print_benchmark(
        f"fp4_cpu_ab_{str(target_dtype).split('.')[-1]}",
        bench_rows,
        [
            "target",
            "candidate",
            "ref ms",
            "native ms",
            "speedup x",
            "ref MiB/s",
            "native MiB/s",
            "throughput delta %",
            "max|diff|",
        ],
        note=f"{source_note} fp4_block_size=16",
    )
