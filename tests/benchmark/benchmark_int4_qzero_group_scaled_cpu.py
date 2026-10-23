#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reproducible GPTQ/AWQ CPU kernel benchmark.

The GPTQ and AWQ paths consume identical packed weights within their method.
GPTQ and AWQ share the same AVX512 compute kernel. The oneDNN W8A8 and torchao
paths are precision/quantizer comparisons, not byte-identical implementations.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import (
    TorchCpuAvx512Linear,
    pack_hf_int4,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import (
    TorchCpuAvx512AwqLinear,
)
from gptqmodel.quantization.awq.utils.packing_utils import (
    pack_awq,
    reverse_awq_order,
    unpack_awq,
)
from gptqmodel.utils.cpu_features import has_avx512_bf16
from gptqmodel.utils.logger import render_table


HF_KERNELS_GPTQ_REVISION = "c467920211e144ea621d9e8193ff71ec160ce256"
DEFAULT_SHAPES = ("4096x4096", "14336x4096", "4096x14336")
DEFAULT_TOKENS = (1, 8, 32, 128, 512)
DEFAULT_GROUP_SIZE = 128


try:
    from torchao.prototype.quantization.int4 import Int4OpaqueTensor
except Exception:
    Int4OpaqueTensor = None

try:
    from torchao.quantization import Int4WeightOnlyQuantizer
except Exception:
    Int4WeightOnlyQuantizer = None


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _parse_ints(raw: str, name: str) -> tuple[int, ...]:
    values = tuple(_positive_int(int(item.strip()), name) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"at least one {name} is required")
    return values


def _parse_shapes(raw: str) -> tuple[tuple[int, int], ...]:
    shapes = []
    for item in raw.split(","):
        text = item.strip().lower()
        if not text:
            continue
        try:
            out_features, in_features = (int(part) for part in text.split("x", 1))
        except ValueError as exc:
            raise ValueError(f"shape must be OUTxIN, got {item!r}") from exc
        _positive_int(out_features, "out_features")
        _positive_int(in_features, "in_features")
        shapes.append((out_features, in_features))
    if not shapes:
        raise ValueError("at least one shape is required")
    return tuple(shapes)


def _git_revision(path: Path | None) -> str:
    if path is None:
        return "unknown"
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        # /proc/cpuinfo is unavailable on some platforms; fall back below.
        pass
    return platform.processor() or "unknown"


def _parallel_info() -> tuple[dict[str, str], str]:
    try:
        text = torch.__config__.parallel_info()
    except Exception:
        return {}, "unavailable"
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if "at::get_num_threads()" in stripped:
            fields["aten_threads"] = stripped.partition(":")[2].strip()
        elif "omp_get_max_threads()" in stripped:
            fields["omp_max_threads"] = stripped.partition(":")[2].strip()
        elif "MKL-DNN" in stripped:
            fields["onednn"] = stripped
        elif "OpenMP" in stripped:
            fields["openmp"] = stripped
    return fields, text


def _torchao_info(torchao_src: Path | None) -> tuple[str, str, bool]:
    try:
        import torchao
    except Exception as exc:
        return "unavailable", str(exc), False
    try:
        import mslk  # noqa: F401
    except Exception:
        mslk_available = False
    else:
        mslk_available = True
    return torchao.__version__, _git_revision(torchao_src), mslk_available


def _bench(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), statistics.mean(samples)


def _gops(tokens: int, in_features: int, out_features: int, median_ms: float) -> float:
    return (2.0 * tokens * in_features * out_features) / (median_ms * 1e6)


def _pack_gptq_buffers(codes: torch.Tensor, zeros: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    in_features, out_features = codes.shape
    groups = zeros.shape[0]
    qweight = torch.zeros((in_features // 8, out_features), dtype=torch.int32)
    codes = codes.to(torch.int32)
    for nibble in range(8):
        qweight |= (codes[nibble::8] & 0xF) << (4 * nibble)
    qzeros = torch.zeros((groups, out_features // 8), dtype=torch.int32)
    zeros = zeros.to(torch.int32)
    for nibble in range(8):
        qzeros |= (zeros[:, nibble::8] & 0xF) << (4 * nibble)
    return qweight, qzeros


def _unpack_qweight(qweight: torch.Tensor, in_features: int) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 8, 1)
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(qweight.unsqueeze(1), shifts),
        0xF,
    ).to(torch.uint8)
    return codes.reshape(in_features, -1)


def _unpack_qzeros(qzeros: torch.Tensor, out_features: int) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 1, 8)
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(qzeros.unsqueeze(2), shifts),
        0xF,
    ).to(torch.uint8)
    return codes.reshape(-1, out_features)


def _dequantize_gptq(
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    zeros_expanded = zeros.repeat_interleave(group_size, dim=0).float()
    scales_expanded = scales.repeat_interleave(group_size, dim=0).float()
    return (codes.float() - zeros_expanded) * scales_expanded


def _make_layers(
    *,
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
):
    in_features, out_features = codes.shape
    qweight, qzeros = _pack_gptq_buffers(codes, zeros)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    layers = []
    for cls in (TorchCpuAvx512Linear, TorchAtenLinear):
        if group_size not in cls.SUPPORTS_GROUP_SIZE:
            layers.append(None)
            continue
        layer = cls(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            pack_dtype=torch.int32,
            register_buffers=True,
        )
        layer.post_init()
        layer.qweight.copy_(qweight)
        layer.qzeros.copy_(qzeros)
        layer.scales = scales.contiguous()
        layer.g_idx.copy_(g_idx)
        layer.eval()
        layer.transform("cpu")
        layers.append(layer)
    return layers[0], layers[1], qweight, qzeros


def _make_awq_layers(
    *,
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
):
    in_features, out_features = codes.shape
    qweight, qzeros = pack_awq(codes, zeros, 4)
    layers = []
    for cls in (TorchCpuAvx512AwqLinear, TorchAtenAwqLinear):
        if group_size not in cls.SUPPORTS_GROUP_SIZE:
            layers.append(None)
            continue
        layer = cls(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            pack_dtype=torch.int32,
            register_buffers=True,
        )
        layer.qweight.copy_(qweight)
        layer.qzeros.copy_(qzeros)
        layer.scales.copy_(scales)
        layer.eval()
        layers.append(layer)
    return layers[0], layers[1], qweight, qzeros


def _torchao_model(
    dense_weight: torch.Tensor,
    *,
    group_size: int,
):
    if Int4OpaqueTensor is not None:
        if group_size not in (32, 64, 128):
            raise ValueError("torchao Int4OpaqueTensor supports group sizes 32/64/128")
        return Int4OpaqueTensor.from_hp(dense_weight, [1, group_size])

    if Int4WeightOnlyQuantizer is None:
        raise RuntimeError("torchao INT4 CPU quantization is unavailable")
    out_features, in_features = dense_weight.shape
    model = torch.nn.Sequential(torch.nn.Linear(in_features, out_features, bias=False))
    model[0].weight.data.copy_(dense_weight)
    quantizer = Int4WeightOnlyQuantizer(
        groupsize=group_size,
        inner_k_tiles=8,
        device=torch.device("cpu"),
    )
    return quantizer.quantize(model)


def _onednn_int8_runner(
    x: torch.Tensor,
    dense_weight: torch.Tensor,
):
    in_features = dense_weight.shape[1]
    out_features = dense_weight.shape[0]
    tokens = x.shape[0]
    x_fp32 = x.float()
    x_scale = max(float(x_fp32.abs().amax().item()) / 127.0, 1e-8)
    qx = torch.clamp(
        torch.round(x_fp32 / x_scale) + 128,
        0,
        255,
    ).to(torch.uint8)
    weight_scales = torch.clamp(
        dense_weight.float().abs().amax(dim=1) / 127.0,
        min=1e-8,
    )
    qweight = torch.clamp(
        torch.round(dense_weight.float() / weight_scales[:, None]),
        -128,
        127,
    ).to(torch.int8)
    packed = torch.ops.onednn.qlinear_prepack(qweight, [tokens, in_features])
    weight_zero_points = torch.zeros(out_features, dtype=torch.int32)

    def run() -> torch.Tensor:
        return torch.ops.onednn.qlinear_pointwise(
            qx,
            float(x_scale),
            128,
            packed,
            weight_scales,
            weight_zero_points,
            None,
            1.0,
            0,
            torch.bfloat16,
            "none",
            [],
            "",
        )

    return run


def _metrics(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, float]:
    difference = output.float() - reference
    max_abs = difference.abs().max().item()
    mean_abs = difference.abs().mean().item()
    relative = difference.norm() / reference.norm().clamp_min(1e-12)
    return relative.item(), max_abs, mean_abs


def _pack_benchmark(
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    in_features: int,
    out_features: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    packer = TorchCpuAvx512Linear.pack_gptq_int4_kernel
    ret_idx = torch.arange(in_features, dtype=torch.int32)

    def fused():
        return packer(
            qweight,
            qzeros,
            torch.empty(0, dtype=torch.int32),
            out_features,
            in_features,
        )

    def python_conversion():
        codes = _unpack_qweight(qweight, in_features)
        packed = pack_hf_int4(
            codes.index_select(0, ret_idx).t().contiguous()
        )
        return packed, _unpack_qzeros(qzeros, out_features)

    fused_median, _ = _bench(fused, warmup=warmup, iters=iters)
    python_median, _ = _bench(python_conversion, warmup=warmup, iters=iters)
    return fused_median, python_median


def _awq_pack_benchmark(
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    in_features: int,
    out_features: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    ok, err = TorchCpuAvx512AwqLinear.cached_validate_once()
    if not ok:
        raise RuntimeError(str(err))
    packer = TorchCpuAvx512AwqLinear.pack_awq_int4_kernel

    def fused():
        return packer(qweight, qzeros, out_features, in_features)

    def python_conversion():
        codes, zeros = unpack_awq(qweight, qzeros, 4, count=out_features)
        codes, zeros = reverse_awq_order(codes, zeros, 4)
        codes = (codes & 0xF).to(torch.uint8)
        zeros = (zeros & 0xF).to(torch.uint8).reshape(-1, out_features)
        return pack_hf_int4(codes.t().contiguous()), zeros.contiguous()

    fused_median, _ = _bench(fused, warmup=warmup, iters=iters)
    python_median, _ = _bench(python_conversion, warmup=warmup, iters=iters)
    return fused_median, python_median


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument(
        "--tokens",
        default=",".join(str(value) for value in DEFAULT_TOKENS),
    )
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--include-awq", action="store_true")
    parser.add_argument("--skip-torchao", action="store_true")
    parser.add_argument("--torchao-src", type=Path, default=None)
    parser.add_argument("--hf-kernels-src", type=Path, default=None)
    args = parser.parse_args()

    group_size = _positive_int(args.group_size, "group_size")
    threads = _positive_int(args.threads, "threads")
    warmup = _positive_int(args.warmup, "warmup")
    iters = _positive_int(args.iters, "iters")
    shapes = _parse_shapes(args.shapes)
    token_counts = _parse_ints(args.tokens, "tokens")

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch rejects changes after the interop pool starts; keep its setting.
        pass
    if not has_avx512_bf16():
        print("This benchmark requires Linux x86-64 with AVX512-BF16.")
        return 1

    parallel_fields, parallel_text = _parallel_info()
    torchao_version, torchao_revision, mslk_available = _torchao_info(args.torchao_src)
    try:
        import kernels
    except Exception:
        kernels_version = "unavailable"
    else:
        kernels_version = getattr(kernels, "__version__", "unknown")

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    repo_root = Path(__file__).resolve().parents[2]
    hf_revision = (
        _git_revision(args.hf_kernels_src)
        if args.hf_kernels_src is not None
        else HF_KERNELS_GPTQ_REVISION
    )

    config_rows = [
        ["host", platform.platform()],
        ["cpu", _cpu_model()],
        ["cpu_features", "AVX512-BF16" if has_avx512_bf16() else "no AVX512-BF16"],
        ["allowed_cpus", str(len(affinity))],
        ["cpu_affinity", ",".join(str(cpu) for cpu in affinity) if affinity else "unavailable"],
        ["affinity_method", "inherited process affinity; benchmark does not set it"],
        ["torch_threads", str(torch.get_num_threads())],
        ["interop_threads", str(torch.get_num_interop_threads())],
        ["OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "unset")],
        ["aten_threads", parallel_fields.get("aten_threads", "unknown")],
        ["omp_max_threads", parallel_fields.get("omp_max_threads", "unknown")],
        ["onednn", parallel_fields.get("onednn", "unknown")],
        ["openmp", parallel_fields.get("openmp", "unknown")],
        ["torch", torch.__version__],
        ["torchao", torchao_version],
        ["torchao_revision", torchao_revision],
        ["mslk_available", str(mslk_available)],
        ["kernels", kernels_version],
        ["gptqmodel_revision", _git_revision(repo_root)],
        ["hf_kernels_gptq_revision", hf_revision],
        ["group_size", str(group_size)],
        ["warmup", str(warmup)],
        ["iters", str(iters)],
        ["timer", "time.perf_counter; median of per-call wall time"],
    ]

    print("Environment")
    print(render_table(config_rows, headers=["field", "value"], tablefmt="grid"))
    if os.environ.get("OMP_NUM_THREADS") is None:
        print(
            "\nOMP_NUM_THREADS is unset. The port sizes its OpenMP region from "
            "torch.get_num_threads(); the ATen and oneDNN runtimes follow their "
            "own pool configuration."
        )

    pack_rows = []
    inference_rows = []
    awq_pack_rows = []
    awq_inference_rows = []
    torchao_error = None

    for out_features, in_features in shapes:
        if out_features % 32 != 0 or in_features % group_size != 0:
            print(
                f"\nSkipping {out_features}x{in_features}: requires "
                f"out_features % 32 == 0 and in_features % {group_size} == 0."
            )
            continue

        torch.manual_seed(args.seed + out_features + in_features)
        groups = in_features // group_size
        codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.uint8)
        zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.uint8)
        scales = torch.rand((groups, out_features), dtype=torch.bfloat16) * 0.04 + 0.01
        dense_weight = _dequantize_gptq(codes, zeros, scales, group_size)
        dense_weight_kn = dense_weight.to(torch.bfloat16).contiguous()
        dense_weight_nk = dense_weight_kn.t().contiguous()

        avx_layer, aten_layer, qweight, qzeros = _make_layers(
            codes=codes,
            zeros=zeros,
            scales=scales,
            group_size=group_size,
        )
        if not args.skip_pack:
            fused_ms, python_ms = _pack_benchmark(
                qweight=qweight,
                qzeros=qzeros,
                in_features=in_features,
                out_features=out_features,
                warmup=warmup,
                iters=iters,
            )
            pack_rows.append(
                [
                    f"{out_features}x{in_features}",
                    f"{fused_ms:.3f}",
                    f"{python_ms:.3f}",
                    f"{python_ms / fused_ms:.2f}x",
                ]
            )

        torchao_model = None
        if not args.skip_torchao:
            try:
                torchao_model = _torchao_model(
                    dense_weight_nk,
                    group_size=group_size,
                )
            except Exception as exc:
                torchao_error = f"{type(exc).__name__}: {exc}"

        for tokens in token_counts:
            x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
            reference = x.float() @ dense_weight
            runners = {
                "AVX512 port": partial(
                    TorchCpuAvx512Linear.gemm_int4_qzero_group_scaled_cpu_kernel,
                    x,
                    avx_layer.qweight,
                    avx_layer.qzeros,
                    avx_layer.scales,
                    group_size,
                ),
                "dense BF16": partial(torch.matmul, x, dense_weight_kn),
            }
            if aten_layer is not None:
                runners["ATen int4pack"] = partial(
                    torch.ops.aten._weight_int4pack_mm_for_cpu,
                    x,
                    aten_layer.qweight,
                    group_size,
                    aten_layer.scales_and_zeros,
                )
            if torchao_model is not None:
                runners["torchao W4A16"] = partial(F.linear, x, torchao_model)
            if hasattr(torch.ops.onednn, "qlinear_prepack") and hasattr(
                torch.ops.onednn,
                "qlinear_pointwise",
            ):
                runners["oneDNN W8A8"] = _onednn_int8_runner(x, dense_weight_nk)

            timings = {}
            outputs = {}
            for name, runner in runners.items():
                output = runner()
                outputs[name] = output
                timings[name] = _bench(runner, warmup=warmup, iters=iters)

            aten_median = timings["ATen int4pack"][0] if "ATen int4pack" in timings else None
            for name, (median_ms, mean_ms) in timings.items():
                relative, max_abs, _ = _metrics(outputs[name], reference)
                inference_rows.append(
                    [
                        f"{out_features}x{in_features}",
                        str(tokens),
                        name,
                        f"{median_ms:.3f}",
                        f"{mean_ms:.3f}",
                        f"{_gops(tokens, in_features, out_features, median_ms):.2f}",
                        f"{aten_median / median_ms:.2f}x" if aten_median is not None else "n/a",
                        f"{relative:.4e}",
                        f"{max_abs:.4e}",
                    ]
                )

        del avx_layer, aten_layer, dense_weight, dense_weight_kn, dense_weight_nk
        if torchao_model is not None:
            del torchao_model

    if pack_rows:
        print("\nPacking")
        print(
            render_table(
                pack_rows,
                headers=[
                    "shape",
                    "fused C++ ms",
                    "previous Python ms",
                    "speedup",
                ],
                tablefmt="grid",
            )
        )

    if args.include_awq:
        for out_features, in_features in shapes:
            if out_features % 32 != 0 or in_features % group_size != 0:
                continue

            torch.manual_seed(args.seed + out_features + in_features + 1)
            groups = in_features // group_size
            codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.uint8)
            zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.uint8)
            scales = (
                torch.rand((groups, out_features), dtype=torch.bfloat16) * 0.04 + 0.01
            )
            dense_weight = (
                codes.float() - zeros.repeat_interleave(group_size, dim=0).float()
            ) * scales.repeat_interleave(group_size, dim=0).float()

            avx_layer, aten_layer, qweight, qzeros = _make_awq_layers(
                codes=codes,
                zeros=zeros,
                scales=scales,
                group_size=group_size,
            )
            if not args.skip_pack:
                fused_ms, python_ms = _awq_pack_benchmark(
                    qweight=qweight,
                    qzeros=qzeros,
                    in_features=in_features,
                    out_features=out_features,
                    warmup=warmup,
                    iters=iters,
                )
                awq_pack_rows.append(
                    [
                        f"{out_features}x{in_features}",
                        f"{fused_ms:.3f}",
                        f"{python_ms:.3f}",
                        f"{python_ms / fused_ms:.2f}x",
                    ]
                )

            for tokens in token_counts:
                x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
                reference = x.float() @ dense_weight
                runners = {
                    "AVX512 AWQ": partial(avx_layer, x),
                }
                if aten_layer is not None:
                    runners["ATen AWQ"] = partial(aten_layer, x)

                timings = {}
                outputs = {}
                for name, runner in runners.items():
                    output = runner()
                    outputs[name] = output
                    timings[name] = _bench(runner, warmup=warmup, iters=iters)

                aten_median = timings["ATen AWQ"][0] if "ATen AWQ" in timings else None
                for name, (median_ms, mean_ms) in timings.items():
                    relative, max_abs, _ = _metrics(outputs[name], reference)
                    awq_inference_rows.append(
                        [
                            f"{out_features}x{in_features}",
                            str(tokens),
                            name,
                            f"{median_ms:.3f}",
                            f"{mean_ms:.3f}",
                            f"{_gops(tokens, in_features, out_features, median_ms):.2f}",
                            f"{aten_median / median_ms:.2f}x" if aten_median is not None else "n/a",
                            f"{relative:.4e}",
                            f"{max_abs:.4e}",
                        ]
                    )

            del avx_layer, aten_layer, dense_weight

        if awq_pack_rows:
            print("\nAWQ packing")
            print(
                render_table(
                    awq_pack_rows,
                    headers=[
                        "shape",
                        "fused C++ ms",
                        "previous Python ms",
                        "speedup",
                    ],
                    tablefmt="grid",
                )
            )
        if awq_inference_rows:
            print("\nAWQ inference")
            print(
                render_table(
                    awq_inference_rows,
                    headers=[
                        "shape",
                        "M",
                        "backend",
                        "median ms",
                        "mean ms",
                        "effective GOPS",
                        "vs ATen",
                        "relative error",
                        "max abs error",
                    ],
                    tablefmt="grid",
                )
            )

    print("\nInference")
    print(
        render_table(
            inference_rows,
            headers=[
                "shape",
                "M",
                "backend",
                "median ms",
                "mean ms",
                "effective GOPS",
                "vs ATen",
                "relative error",
                "max abs error",
            ],
            tablefmt="grid",
        )
    )
    print(
        "\nAccuracy compares each output with an FP32 Torch matmul of the same "
        "dequantized weight. GPTQ and AWQ share the same AVX512 compute kernel. "
        "torchao uses a different affine quantizer and oneDNN uses W8A8, so "
        "their error columns are not GPTQ kernel error."
    )
    if torchao_error is not None:
        print(f"torchao quantization was unavailable: {torchao_error}")
    if not mslk_available:
        print(
            "torchao A16W4 uses Int4OpaqueTensor/ATen int4pack and does not "
            "require mslk; torchao DA8W4 requires mslk and is not benchmarked."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
