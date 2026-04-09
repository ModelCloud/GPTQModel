#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from tabulate import tabulate


# Keep this CPU benchmark isolated from optional GPU backend import side effects.
os.environ.setdefault("GPTQMODEL_DISABLE_BITBLAS", "1")

from gptqmodel.nn_modules.qlinear.torch_int8 import Int8PackedModule


DEFAULT_MODEL_DIR = Path("/root/GLM-4.6")
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_CHUNK_ROWS = 2048
DEFAULT_WARMUP = 3
DEFAULT_ITERS = 10
DEFAULT_OUTPUT_DTYPE = torch.bfloat16


def _positive_divisor(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _parse_batches(raw: str) -> list[int]:
    batches = [_positive_divisor(int(item.strip()), "batch") for item in raw.split(",") if item.strip()]
    if not batches:
        raise ValueError("at least one batch is required")
    return batches


def _rank_tensor(name: str) -> tuple[int, int]:
    # Prefer linear heads when the size ties with embeddings.
    if name == "lm_head.weight":
        return (3, 0)
    if "lm_head" in name:
        return (2, 0)
    if name.endswith(".weight"):
        return (1, 0)
    return (0, 0)


def find_largest_2d_tensor(model_dir: Path) -> tuple[Path, str, tuple[int, int]]:
    best: tuple[int, tuple[int, int], tuple[int, int], str, Path] | None = None
    files = sorted(model_dir.glob("model-*.safetensors"))
    mtp = model_dir / "mtp.safetensors"
    if mtp.exists():
        files.append(mtp)
    if not files:
        raise FileNotFoundError(f"no safetensors shards found under {model_dir}")

    for shard in files:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                shape = tuple(handle.get_slice(name).get_shape())
                if len(shape) != 2:
                    continue
                rows, cols = int(shape[0]), int(shape[1])
                numel = rows * cols
                rank = _rank_tensor(name)
                candidate = (numel, rank, (rows, cols), name, shard)
                if best is None or candidate > best:
                    best = candidate

    if best is None:
        raise RuntimeError(f"no 2D tensors found under {model_dir}")

    _, _, shape, name, shard = best
    return shard, name, shape


def quantize_activation_per_tensor_symmetric(x_fp32: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = max(float(x_fp32.abs().max().item()) / 127.0, 1e-8)
    qx = torch.clamp(torch.round(x_fp32 / scale), -128, 127).to(torch.int8)
    return qx, scale


def bench_ms(fn, warmup: int, iters: int) -> tuple[float, float, list[float]]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()

    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            samples.append((end - start) * 1e3)

    return statistics.mean(samples), statistics.median(samples), samples


def gops(tokens: int, in_features: int, out_features: int, ms: float) -> float:
    ops = 2.0 * tokens * in_features * out_features
    return ops / (ms * 1e6)


def capture_onednn_isa(tokens: int, in_features: int, out_features: int) -> str | None:
    code = f"""
import torch
M = {tokens}
I = {in_features}
O = {out_features}
qx = torch.randint(-128, 128, (M, I), dtype=torch.int8)
qw = torch.randint(-128, 128, (O, I), dtype=torch.int8)
ws = torch.ones((O,), dtype=torch.float32)
wzp = torch.zeros((O,), dtype=torch.int32)
packed = torch.ops.onednn.qlinear_prepack(qw, [M, I])
torch.ops.onednn.qlinear_pointwise(
    qx, 1.0, 0, packed, ws, wzp, None, 1.0, 0, torch.bfloat16, "none", [], ""
)
"""
    env = os.environ.copy()
    env["DNNL_VERBOSE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    for line in completed.stdout.splitlines():
        if "onednn_verbose" not in line or ",primitive,exec,cpu,matmul," not in line:
            continue
        fields = line.split(",")
        if len(fields) >= 7:
            return fields[6]
    return None


def build_int8_weight_state(
    shard: Path,
    tensor_name: str,
    shape: tuple[int, int],
    chunk_rows: int,
) -> dict[str, torch.Tensor]:
    out_features, in_features = shape
    int8_weight_nk = torch.empty((out_features, in_features), dtype=torch.int8)
    int8_weight_scales_fp32 = torch.empty((out_features,), dtype=torch.float32)

    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        weight_slice = handle.get_slice(tensor_name)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            weight_chunk_nk = weight_slice[start:end, :].to(torch.float32).contiguous()
            channel_scale = torch.maximum(
                weight_chunk_nk.abs().amax(dim=1),
                torch.full((end - start,), 1e-8, dtype=torch.float32),
            ) / 127.0
            int8_codes_nk = torch.clamp(
                torch.round(weight_chunk_nk / channel_scale.unsqueeze(1)),
                -128,
                127,
            ).to(torch.int8)
            int8_weight_nk[start:end] = int8_codes_nk
            int8_weight_scales_fp32[start:end] = channel_scale

    return {
        "int8_weight_nk": int8_weight_nk.contiguous(),
        "int8_weight_scales_fp32": int8_weight_scales_fp32.contiguous(),
        "int8_weight_scales_bf16": int8_weight_scales_fp32.to(torch.bfloat16).contiguous(),
        "int8_weight_zero_points": torch.zeros((out_features,), dtype=torch.int32),
    }


def compute_reference(
    shard: Path,
    tensor_name: str,
    shape: tuple[int, int],
    x_fp32: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    out_features, _ = shape
    reference = torch.empty((x_fp32.shape[0], out_features), dtype=torch.float32)

    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        weight_slice = handle.get_slice(tensor_name)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            weight_chunk_nk = weight_slice[start:end, :].to(torch.float32).contiguous()
            reference[:, start:end] = x_fp32 @ weight_chunk_nk.t()

    return reference


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark repo TorchInt8 GPTQ kernel vs oneDNN qlinear on CPU"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--batches",
        type=str,
        default=",".join(str(batch) for batch in DEFAULT_BATCHES),
        help="comma-separated batch sizes, e.g. 1,2,4,8,16,32,64,128",
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    warmup = _positive_divisor(args.warmup, "warmup")
    iters = _positive_divisor(args.iters, "iters")
    chunk_rows = _positive_divisor(args.chunk_rows, "chunk_rows")
    batches = _parse_batches(args.batches)

    if not hasattr(torch.ops.aten, "_weight_int8pack_mm"):
        raise RuntimeError("aten::_weight_int8pack_mm is unavailable in this PyTorch build")
    if not hasattr(torch.ops.onednn, "qlinear_prepack"):
        raise RuntimeError("onednn::qlinear_prepack is unavailable in this PyTorch build")
    if not hasattr(torch.ops.onednn, "qlinear_pointwise"):
        raise RuntimeError("onednn::qlinear_pointwise is unavailable in this PyTorch build")

    torch.set_num_interop_threads(1)

    shard, tensor_name, shape = find_largest_2d_tensor(args.model_dir)
    out_features, in_features = shape

    build_start = time.perf_counter()
    weight_state = build_int8_weight_state(
        shard=shard,
        tensor_name=tensor_name,
        shape=shape,
        chunk_rows=chunk_rows,
    )
    weight_build_ms = (time.perf_counter() - build_start) * 1e3

    int8_weight_nk = weight_state["int8_weight_nk"]
    int8_weight_scales_fp32 = weight_state["int8_weight_scales_fp32"]
    int8_weight_scales_bf16 = weight_state["int8_weight_scales_bf16"]
    int8_weight_zero_points = weight_state["int8_weight_zero_points"]

    torch_int8_module = Int8PackedModule(int8_weight_nk, int8_weight_scales_bf16).eval()

    config_rows = [
        ["model_dir", str(args.model_dir)],
        ["tensor", tensor_name],
        ["shard", shard.name],
        ["shape", f"{out_features} x {in_features}"],
        ["batches", ",".join(str(batch) for batch in batches)],
        ["chunk_rows", chunk_rows],
        ["threads", torch.get_num_threads()],
        ["interop_threads", torch.get_num_interop_threads()],
        ["weight_build_ms", f"{weight_build_ms:.2f}"],
        ["output_dtype", str(DEFAULT_OUTPUT_DTYPE)],
        ["note", "forward-only timings; oneDNN prepack measured separately"],
        ["quant", "TorchInt8=w8/a16 bf16, oneDNN=w8/a8 bf16"],
    ]

    perf_rows: list[list[str]] = []
    acc_rows: list[list[str]] = []
    isa_rows: list[list[str]] = []

    for index, batch in enumerate(batches):
        torch.manual_seed(args.seed + batch)
        x_bf16 = torch.randn(batch, in_features, dtype=torch.bfloat16)
        x_fp32 = x_bf16.float()
        qx_int8, x_scale = quantize_activation_per_tensor_symmetric(x_fp32)

        ref_start = time.perf_counter()
        reference = compute_reference(
            shard=shard,
            tensor_name=tensor_name,
            shape=shape,
            x_fp32=x_fp32,
            chunk_rows=chunk_rows,
        )
        reference_ms = (time.perf_counter() - ref_start) * 1e3

        prepack_start = time.perf_counter()
        onednn_weight = torch.ops.onednn.qlinear_prepack(int8_weight_nk, [batch, in_features])
        onednn_prepack_ms = (time.perf_counter() - prepack_start) * 1e3

        def run_torch_int8() -> torch.Tensor:
            return torch_int8_module(x_bf16)

        def run_onednn(
            onednn_weight_packed: torch.Tensor = onednn_weight,
        ) -> torch.Tensor:
            return torch.ops.onednn.qlinear_pointwise(
                qx_int8,
                float(x_scale),
                0,
                onednn_weight_packed,
                int8_weight_scales_fp32,
                int8_weight_zero_points,
                None,
                1.0,
                0,
                DEFAULT_OUTPUT_DTYPE,
                "none",
                [],
                "",
            )

        with torch.inference_mode():
            out_torch_int8_raw = run_torch_int8()
            out_onednn_raw = run_onednn()

        if out_torch_int8_raw.dtype != DEFAULT_OUTPUT_DTYPE:
            raise RuntimeError(
                f"expected TorchInt8 output dtype {DEFAULT_OUTPUT_DTYPE}, got {out_torch_int8_raw.dtype}"
            )
        if out_onednn_raw.dtype != DEFAULT_OUTPUT_DTYPE:
            raise RuntimeError(
                f"expected oneDNN output dtype {DEFAULT_OUTPUT_DTYPE}, got {out_onednn_raw.dtype}"
            )

        out_torch_int8 = out_torch_int8_raw.float()
        out_onednn = out_onednn_raw.float()

        torch_int8_mean_ms, torch_int8_median_ms, _ = bench_ms(
            run_torch_int8, warmup=warmup, iters=iters
        )
        onednn_mean_ms, onednn_median_ms, _ = bench_ms(
            run_onednn, warmup=warmup, iters=iters
        )

        isa = None
        if index in (0, len(batches) - 1):
            isa = capture_onednn_isa(batch, in_features, out_features)
            isa_rows.append([str(batch), isa or "unknown"])

        torch_int8_diff = out_torch_int8 - reference
        onednn_diff = out_onednn - reference
        backend_delta = out_onednn - out_torch_int8

        perf_rows.append(
            [
                str(batch),
                f"{torch_int8_mean_ms:.3f}",
                f"{torch_int8_median_ms:.3f}",
                f"{gops(batch, in_features, out_features, torch_int8_mean_ms):.2f}",
                f"{onednn_mean_ms:.3f}",
                f"{onednn_median_ms:.3f}",
                f"{gops(batch, in_features, out_features, onednn_mean_ms):.2f}",
                "TorchInt8" if torch_int8_mean_ms <= onednn_mean_ms else "oneDNN",
                f"{max(torch_int8_mean_ms, onednn_mean_ms) / min(torch_int8_mean_ms, onednn_mean_ms):.3f}x",
                f"{onednn_prepack_ms:.3f}",
                f"{reference_ms:.2f}",
            ]
        )

        acc_rows.append(
            [
                str(batch),
                f"{torch_int8_diff.abs().max().item():.6f}",
                f"{torch_int8_diff.abs().mean().item():.6f}",
                f"{math.sqrt(torch.mean(torch_int8_diff ** 2).item()):.6f}",
                f"{onednn_diff.abs().max().item():.6f}",
                f"{onednn_diff.abs().mean().item():.6f}",
                f"{math.sqrt(torch.mean(onednn_diff ** 2).item()):.6f}",
                f"{backend_delta.abs().max().item():.6f}",
                f"{backend_delta.abs().mean().item():.6f}",
            ]
        )

        del onednn_weight, reference, out_torch_int8, out_onednn

    print("Configuration")
    print(tabulate(config_rows, headers=["field", "value"], tablefmt="grid"))
    print()
    print("Performance")
    print(
        tabulate(
            perf_rows,
            headers=[
                "batch",
                "TorchInt8 mean ms",
                "TorchInt8 median ms",
                "TorchInt8 GOPS",
                "oneDNN mean ms",
                "oneDNN median ms",
                "oneDNN GOPS",
                "winner",
                "speedup",
                "oneDNN prepack ms",
                "ref build ms",
            ],
            tablefmt="grid",
        )
    )
    print()
    print("Accuracy")
    print(
        tabulate(
            acc_rows,
            headers=[
                "batch",
                "TorchInt8 max|diff|",
                "TorchInt8 mean|diff|",
                "TorchInt8 rmse",
                "oneDNN max|diff|",
                "oneDNN mean|diff|",
                "oneDNN rmse",
                "backend max|delta|",
                "backend mean|delta|",
            ],
            tablefmt="grid",
        )
    )
    print()
    print("oneDNN ISA Probe")
    print(tabulate(isa_rows, headers=["batch", "isa"], tablefmt="grid"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
