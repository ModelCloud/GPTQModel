# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from gptqmodel.quantization.npu_linalg import npu_inverse_cholesky_factor
from gptqmodel.utils.torch import HAS_NPU


@dataclass
class BenchResult:
    size: int
    cpu_ms: float
    npu_ms: float
    speedup: float
    factor_max_abs: float
    inverse_max_abs: float
    inverse_mean_abs: float


def _spd_matrix(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(size, size, generator=generator, dtype=torch.float32)
    return values.matmul(values.T) + torch.eye(size, dtype=torch.float32) * 0.25


def _cpu_reference(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(matrix)),
        upper=True,
    )


def _time_ms(
    fn: Callable[[], torch.Tensor],
    *,
    repeat: int,
    warmup: int,
    sync: Callable[[], None],
) -> tuple[float, torch.Tensor]:
    result = None
    for _ in range(warmup):
        result = fn()
        sync()

    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        sync()
        timings.append((time.perf_counter() - start) * 1000.0)

    return statistics.median(timings), result


def run_benchmark(
    sizes: Iterable[int],
    *,
    repeat: int,
    warmup: int,
    device: str,
    block_size: int,
) -> list[BenchResult]:
    if not HAS_NPU:
        raise RuntimeError("Ascend NPU is not available.")

    npu_device = torch.device(device)
    results = []

    for size in sizes:
        matrix_cpu = _spd_matrix(size, seed=9000 + size)
        matrix_npu = matrix_cpu.to(device=npu_device)

        cpu_ms, cpu_factor = _time_ms(
            lambda: _cpu_reference(matrix_cpu),
            repeat=repeat,
            warmup=warmup,
            sync=lambda: None,
        )
        npu_ms, npu_factor = _time_ms(
            lambda: npu_inverse_cholesky_factor(matrix_npu, block_size=block_size),
            repeat=repeat,
            warmup=warmup,
            sync=torch.npu.synchronize,
        )

        npu_factor_cpu = npu_factor.cpu()
        factor_diff = (npu_factor_cpu - cpu_factor).abs()

        reconstructed = npu_factor.T.matmul(npu_factor).cpu()
        expected_inverse = torch.linalg.inv(matrix_cpu)
        inverse_diff = (reconstructed - expected_inverse).abs()

        results.append(
            BenchResult(
                size=size,
                cpu_ms=cpu_ms,
                npu_ms=npu_ms,
                speedup=cpu_ms / npu_ms if npu_ms > 0 else float("inf"),
                factor_max_abs=factor_diff.max().item(),
                inverse_max_abs=inverse_diff.max().item(),
                inverse_mean_abs=inverse_diff.mean().item(),
            )
        )

    return results


def _parse_sizes(raw: str) -> list[int]:
    sizes = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            sizes.append(int(item))
    if not sizes:
        raise ValueError("At least one matrix size is required.")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the NPU-native GPTQ Hessian inverse helper against CPU torch.linalg.",
    )
    parser.add_argument("--sizes", default="64,128,256,512", help="Comma-separated square Hessian sizes.")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()

    results = run_benchmark(
        _parse_sizes(args.sizes),
        repeat=max(1, args.repeat),
        warmup=max(0, args.warmup),
        device=args.device,
        block_size=max(1, args.block_size),
    )

    print("size,block_size,cpu_ms,npu_ms,cpu_div_npu,factor_max_abs,inverse_max_abs,inverse_mean_abs")
    for result in results:
        print(
            f"{result.size},"
            f"{max(1, args.block_size)},"
            f"{result.cpu_ms:.3f},"
            f"{result.npu_ms:.3f},"
            f"{result.speedup:.3f},"
            f"{result.factor_max_abs:.6e},"
            f"{result.inverse_max_abs:.6e},"
            f"{result.inverse_mean_abs:.6e}"
        )


if __name__ == "__main__":
    main()
