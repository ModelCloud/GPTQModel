#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
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


DEFAULT_MODEL_DIR = Path("/root/GLM-4.6")
DEFAULT_GROUP_SIZE = 128
DEFAULT_TOKENS = 1
DEFAULT_CHUNK_ROWS = 2048
DEFAULT_WARMUP = 3
DEFAULT_ITERS = 10


def _positive_divisor(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


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


def run_onednn_verbose_probe(tokens: int, in_features: int, out_features: int) -> str | None:
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
    lines = [
        line
        for line in completed.stdout.splitlines()
        if "onednn_verbose" in line and ",primitive,exec,cpu,matmul," in line
    ]
    return lines[-1] if lines else None


def build_reference_and_packed_kernels(
    shard: Path,
    tensor_name: str,
    shape: tuple[int, int],
    tokens: int,
    group_size: int,
    chunk_rows: int,
    seed: int,
) -> dict[str, object]:
    out_features, in_features = shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={group_size}"
        )
    if chunk_rows % 16 != 0:
        raise ValueError(f"chunk_rows={chunk_rows} must be divisible by 16")
    if out_features % 16 != 0:
        raise ValueError(f"out_features={out_features} must be divisible by 16")

    torch.manual_seed(seed)
    x_bf16 = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    x_fp32 = x_bf16.float()
    qx_int8, x_scale = quantize_activation_per_tensor_symmetric(x_fp32)

    groups = in_features // group_size
    int4_weight = torch.empty((out_features, in_features // 2), dtype=torch.uint8)
    scales_and_zeros = torch.zeros((groups, out_features, 2), dtype=torch.bfloat16)
    int8_weight = torch.empty((out_features, in_features), dtype=torch.int8)
    int8_weight_scales = torch.empty((out_features,), dtype=torch.float32)
    int8_weight_zero_points = torch.zeros((out_features,), dtype=torch.int32)
    reference = torch.empty((tokens, out_features), dtype=torch.float32)

    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        weight_slice = handle.get_slice(tensor_name)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            rows = end - start
            weight_chunk = weight_slice[start:end, :].to(torch.float32).contiguous()
            reference[:, start:end] = x_fp32 @ weight_chunk.t()

            int8_scales = torch.maximum(
                weight_chunk.abs().amax(dim=1),
                torch.full((rows,), 1e-8, dtype=torch.float32),
            ) / 127.0
            int8_codes = torch.clamp(
                torch.round(weight_chunk / int8_scales.unsqueeze(1)),
                -128,
                127,
            ).to(torch.int8)
            int8_weight[start:end] = int8_codes
            int8_weight_scales[start:end] = int8_scales

            grouped = weight_chunk.view(rows, groups, group_size)
            int4_scales = torch.maximum(
                grouped.abs().amax(dim=2),
                torch.full((rows, groups), 1e-8, dtype=torch.float32),
            ) / 7.0
            int4_signed = torch.clamp(
                torch.round(grouped / int4_scales.unsqueeze(-1)),
                -8,
                7,
            ).to(torch.int8)
            int4_codes = (int4_signed + 8).to(torch.int32).view(rows, in_features).contiguous()
            int4_weight[start:end] = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                int4_codes, 1
            ).contiguous()
            scales_and_zeros[:, start:end, 0] = int4_scales.transpose(0, 1).to(torch.bfloat16)

    onednn_weight = torch.ops.onednn.qlinear_prepack(int8_weight, [tokens, in_features])
    del int8_weight

    return {
        "x_bf16": x_bf16,
        "qx_int8": qx_int8,
        "x_scale": x_scale,
        "reference": reference,
        "int4_weight": int4_weight,
        "scales_and_zeros": scales_and_zeros,
        "onednn_weight": onednn_weight,
        "int8_weight_scales": int8_weight_scales,
        "int8_weight_zero_points": int8_weight_zero_points,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchAten GPTQ int4 vs oneDNN qlinear on CPU"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tokens = _positive_divisor(args.tokens, "tokens")
    warmup = _positive_divisor(args.warmup, "warmup")
    iters = _positive_divisor(args.iters, "iters")
    group_size = _positive_divisor(args.group_size, "group_size")
    chunk_rows = _positive_divisor(args.chunk_rows, "chunk_rows")

    torch.set_num_interop_threads(1)

    shard, tensor_name, shape = find_largest_2d_tensor(args.model_dir)
    out_features, in_features = shape
    build_start = time.perf_counter()
    state = build_reference_and_packed_kernels(
        shard=shard,
        tensor_name=tensor_name,
        shape=shape,
        tokens=tokens,
        group_size=group_size,
        chunk_rows=chunk_rows,
        seed=args.seed,
    )
    build_ms = (time.perf_counter() - build_start) * 1e3

    x_bf16 = state["x_bf16"]
    qx_int8 = state["qx_int8"]
    x_scale = state["x_scale"]
    reference = state["reference"]
    int4_weight = state["int4_weight"]
    scales_and_zeros = state["scales_and_zeros"]
    onednn_weight = state["onednn_weight"]
    int8_weight_scales = state["int8_weight_scales"]
    int8_weight_zero_points = state["int8_weight_zero_points"]

    def run_torch_aten() -> torch.Tensor:
        return torch.ops.aten._weight_int4pack_mm_for_cpu(
            x_bf16, int4_weight, group_size, scales_and_zeros
        )

    def run_onednn() -> torch.Tensor:
        return torch.ops.onednn.qlinear_pointwise(
            qx_int8,
            float(x_scale),
            0,
            onednn_weight,
            int8_weight_scales,
            int8_weight_zero_points,
            None,
            1.0,
            0,
            torch.bfloat16,
            "none",
            [],
            "",
        )

    with torch.inference_mode():
        out_aten = run_torch_aten().float()
        out_onednn = run_onednn().float()

    aten_mean_ms, aten_median_ms, _ = bench_ms(run_torch_aten, warmup=warmup, iters=iters)
    onednn_mean_ms, onednn_median_ms, _ = bench_ms(run_onednn, warmup=warmup, iters=iters)

    onednn_verbose_line = run_onednn_verbose_probe(tokens, in_features, out_features)

    config_rows = [
        ["model_dir", str(args.model_dir)],
        ["tensor", tensor_name],
        ["shard", shard.name],
        ["shape", f"{out_features} x {in_features}"],
        ["tokens", tokens],
        ["group_size", group_size],
        ["chunk_rows", chunk_rows],
        ["threads", torch.get_num_threads()],
        ["interop_threads", torch.get_num_interop_threads()],
        ["build_ms", f"{build_ms:.2f}"],
    ]

    results_rows = [
        [
            "TorchAten GPTQ",
            "aten._weight_int4pack_mm_for_cpu",
            "w4 / a16",
            "bf16",
            f"{aten_mean_ms:.3f}",
            f"{aten_median_ms:.3f}",
            f"{gops(tokens, in_features, out_features, aten_mean_ms):.2f}",
            "1.000x",
            f"{(out_aten - reference).abs().max().item():.6f}",
            f"{(out_aten - reference).abs().mean().item():.6f}",
            f"{math.sqrt(torch.mean((out_aten - reference) ** 2).item()):.6f}",
        ],
        [
            "oneDNN qlinear",
            "onednn.qlinear_pointwise",
            "w8 / a8",
            "bf16",
            f"{onednn_mean_ms:.3f}",
            f"{onednn_median_ms:.3f}",
            f"{gops(tokens, in_features, out_features, onednn_mean_ms):.2f}",
            f"{aten_mean_ms / onednn_mean_ms:.3f}x",
            f"{(out_onednn - reference).abs().max().item():.6f}",
            f"{(out_onednn - reference).abs().mean().item():.6f}",
            f"{math.sqrt(torch.mean((out_onednn - reference) ** 2).item()):.6f}",
        ],
    ]

    print("Configuration")
    print(tabulate(config_rows, headers=["field", "value"], tablefmt="grid"))
    print()
    print("Results")
    print(
        tabulate(
            results_rows,
            headers=[
                "backend",
                "kernel",
                "quant",
                "out",
                "mean ms",
                "median ms",
                "effective GOPS",
                "vs ATen",
                "max|diff|",
                "mean|diff|",
                "rmse",
            ],
            tablefmt="grid",
        )
    )
    print()
    print("oneDNN Verbose")
    if onednn_verbose_line is None:
        print("No oneDNN matmul verbose line captured.")
    else:
        print(onednn_verbose_line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
