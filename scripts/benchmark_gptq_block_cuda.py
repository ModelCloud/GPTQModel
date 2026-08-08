#!/usr/bin/env python
"""Benchmark the native CUDA GPTQ block kernel against the eager reference."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def _run_nvidia_smi(query: str, *, gpu_index: int | None = None) -> list[list[str]]:
    command = ["nvidia-smi"]
    if gpu_index is not None:
        command.extend(("-i", str(gpu_index)))
    command.extend((f"--query-{query}", "--format=csv,noheader,nounits"))
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _idle_gpu(
    gpu_index: int, samples: int, interval: float, memory_limit_mib: int
) -> dict[str, str]:
    inventory = _run_nvidia_smi(
        "gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu"
    )
    devices = {int(row[0]): row for row in inventory}
    if gpu_index not in devices:
        raise RuntimeError(
            f"physical GPU {gpu_index} is unavailable; found {sorted(devices)}"
        )

    accepted = None
    for sample in range(samples):
        row = {
            int(item[0]): item
            for item in _run_nvidia_smi(
                "gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu"
            )
        }
        current = row[gpu_index]
        compute_processes = _run_nvidia_smi(
            "compute-apps=gpu_uuid,pid,process_name,used_memory",
            gpu_index=gpu_index,
        )
        memory_used = int(current[4])
        utilization = int(current[5])
        if compute_processes:
            raise RuntimeError(
                f"physical GPU {gpu_index} has foreign compute processes: {compute_processes}"
            )
        if utilization != 0 or memory_used > memory_limit_mib:
            raise RuntimeError(
                f"physical GPU {gpu_index} is not idle: utilization={utilization}%, "
                f"memory={memory_used} MiB (limit={memory_limit_mib} MiB)"
            )
        accepted = current
        if sample + 1 < samples:
            time.sleep(interval)

    assert accepted is not None
    return {
        "index": accepted[0],
        "pci_bus_id": accepted[1],
        "uuid": accepted[2],
        "name": accepted[3],
        "memory_used": accepted[4],
        "utilization": accepted[5],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, required=True, help="Physical nvidia-smi GPU index"
    )
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--count", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 8), default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--memory-limit-mib", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.count % args.group_size != 0:
        raise ValueError("group-size must divide count")
    device = _idle_gpu(
        args.gpu, args.idle_samples, args.idle_interval, args.memory_limit_mib
    )
    print(
        "Idle gate accepted: "
        f"physical={device['index']} pci={device['pci_bus_id']} uuid={device['uuid']} name={device['name']} "
        f"utilization={device['utilization']}% memory={device['memory_used']}MiB samples={args.idle_samples} "
        f"memory_limit={args.memory_limit_mib}MiB"
    )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]

    import torch

    from gptqmodel.utils.gptq_block import gptq_block_cuda, prewarm_gptq_block_cuda

    torch.manual_seed(20260808)
    cuda = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(cuda)
    maxq = 2**args.bits - 1
    groups = args.count // args.group_size
    weights = torch.randn(args.rows, args.count, device=cuda, dtype=torch.float32)
    factor = torch.randn(args.count, args.count, device=cuda, dtype=torch.float32)
    covariance = factor @ factor.T + 0.5 * torch.eye(args.count, device=cuda)
    hessian_inverse = torch.linalg.cholesky(
        torch.linalg.inv(covariance), upper=True
    ).contiguous()
    scale = torch.rand(args.rows, groups, device=cuda, dtype=torch.float32) + 0.125
    zero = torch.randint(
        0, maxq + 1, (args.rows, groups), device=cuda, dtype=torch.int32
    ).float()

    def eager():
        working = weights.clone()
        quantized = torch.empty_like(working)
        errors = torch.empty_like(working)
        for column in range(args.count):
            values = working[:, column]
            q_scale = scale[:, column // args.group_size]
            q_zero = zero[:, column // args.group_size]
            q = q_scale * (
                torch.clamp(torch.round(values / q_scale) + q_zero, 0, maxq) - q_zero
            )
            error = (values - q) / hessian_inverse[column, column]
            quantized[:, column] = q
            errors[:, column] = error
            working[:, column:] = torch.addr(
                working[:, column:], error, hessian_inverse[column, column:], alpha=-1.0
            )
        return quantized, errors

    def native():
        return gptq_block_cuda(
            weights, hessian_inverse, scale, zero, maxq, args.group_size
        )

    prewarm_gptq_block_cuda()
    reference_quantized, reference_errors = eager()
    actual_quantized, actual_errors = native()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_quantized, reference_quantized, atol=0, rtol=0
    )
    torch.testing.assert_close(actual_errors, reference_errors, atol=0, rtol=0)

    repeated = [native() for _ in range(10)]
    torch.cuda.synchronize()
    quantized_spread = max(
        (result[0] - repeated[0][0]).abs().max().item() for result in repeated[1:]
    )
    error_spread = max(
        (result[1] - repeated[0][1]).abs().max().item() for result in repeated[1:]
    )

    def timed(function) -> list[float]:
        for _ in range(args.warmup):
            function()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    eager_ms = timed(eager)
    native_ms = timed(native)

    def accuracy_metrics(actual, reference):
        delta = actual - reference
        return {
            "mae": delta.abs().mean().item(),
            "rmse": delta.square().mean().sqrt().item(),
            "relative_l2": (
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)
            ).item(),
            "max_abs": delta.abs().max().item(),
            "finite": bool(torch.isfinite(actual).all()),
        }

    quantized_metrics = accuracy_metrics(actual_quantized, reference_quantized)
    error_metrics = accuracy_metrics(actual_errors, reference_errors)
    group_index = torch.arange(args.count, device=cuda) // args.group_size
    actual_codes = torch.round(
        actual_quantized / scale[:, group_index] + zero[:, group_index]
    ).to(torch.int32)
    reference_codes = torch.round(
        reference_quantized / scale[:, group_index] + zero[:, group_index]
    ).to(torch.int32)
    code_mismatches = torch.count_nonzero(actual_codes != reference_codes).item()
    eager_median = statistics.median(eager_ms)
    native_median = statistics.median(native_ms)
    p95_index = max(0, math.ceil(0.95 * args.repetitions) - 1)

    print("+----------+---------+--------------+----------------------+-------+-----+")
    print("| Physical | Logical | PCI bus      | Device               | CC    | SMs |")
    print("+----------+---------+--------------+----------------------+-------+-----+")
    print(
        f"| {args.gpu:8d} | cuda:0  | {device['pci_bus_id']:12s} | {properties.name[:20]:20s} | "
        f"{properties.major}.{properties.minor:<3d} | {properties.multi_processor_count:3d} |"
    )
    print("+----------+---------+--------------+----------------------+-------+-----+")
    print(
        f"Software: Python={sys.version.split()[0]} Torch={torch.__version__} CUDA={torch.version.cuda}"
    )
    print(
        "+---------------+---------+------+-------+-------+----------+----------+---------+----------+"
    )
    print(
        "| Algorithm     | Dtype   | Rows | Count | Group | Median ms| P95 ms   | Speedup | Max spread|"
    )
    print(
        "+---------------+---------+------+-------+-------+----------+----------+---------+----------+"
    )
    print(
        f"| Eager         | FP32    | {args.rows:4d} | {args.count:5d} | {args.group_size:5d} | "
        f"{eager_median:8.3f} | {sorted(eager_ms)[p95_index]:8.3f} | {1.0:7.2f} | {0.0:8.1e} |"
    )
    print(
        f"| Native CUDA   | FP32    | {args.rows:4d} | {args.count:5d} | {args.group_size:5d} | "
        f"{native_median:8.3f} | {sorted(native_ms)[p95_index]:8.3f} | "
        f"{eager_median / native_median:7.2f} | {max(quantized_spread, error_spread):8.1e} |"
    )
    print(
        "+---------------+---------+------+-------+-------+----------+----------+---------+----------+"
    )
    print(
        "+---------------+-------+-----------+-----------+-----------+-----------+-----------+"
    )
    print(
        "| Tensor        | Dtype | MAE       | RMSE      | Rel L2    | Max abs   | Finite    |"
    )
    print(
        "+---------------+-------+-----------+-----------+-----------+-----------+-----------+"
    )
    print(
        f"| Quantized     | FP32  | {quantized_metrics['mae']:9.2e} | {quantized_metrics['rmse']:9.2e} | "
        f"{quantized_metrics['relative_l2']:9.2e} | {quantized_metrics['max_abs']:9.2e} | "
        f"{str(quantized_metrics['finite']):9s} |"
    )
    print(
        f"| Error         | FP32  | {error_metrics['mae']:9.2e} | {error_metrics['rmse']:9.2e} | "
        f"{error_metrics['relative_l2']:9.2e} | {error_metrics['max_abs']:9.2e} | "
        f"{str(error_metrics['finite']):9s} |"
    )
    print(
        "+---------------+-------+-----------+-----------+-----------+-----------+-----------+"
    )
    print(f"Exact logical-code mismatches: {code_mismatches}")


if __name__ == "__main__":
    main()
