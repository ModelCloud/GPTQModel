#!/usr/bin/env python3
"""Benchmark batched find_params for representative group sizes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization import QuantizeConfig, Quantizer, ScaleSearchConfig  # noqa: E402


def build_hessians(columns: int, group_size: int, device: str):
    generator = torch.Generator(device=device).manual_seed(123)
    H = torch.randn(columns, max(columns // 2, 1), generator=generator, device=device, dtype=torch.float32)
    H = H.matmul(H.t())
    H = (H + H.t()) * 0.5
    diagonals = []
    blocks = []
    for g in range(0, columns, group_size):
        ge = min(g + group_size, columns)
        blocks.append(H[g:ge, g:ge])
        diagonals.append(H.diagonal()[g:ge])
    return torch.stack(blocks, dim=0), torch.stack(diagonals, dim=0)


def benchmark(rows: int, columns: int, group_size: int, method: ScaleSearchConfig, device: str):
    generator = torch.Generator(device=device).manual_seed(42)
    W = torch.randn((rows, columns), generator=generator, device=device, dtype=torch.float32)
    H_blocks, H_diag = build_hessians(columns, group_size, device)
    W_3d = W.reshape(rows, columns // group_size, group_size)
    if method == ScaleSearchConfig.ACTIVATION:
        batched_hessian = H_diag
    elif method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
        batched_hessian = H_blocks
    else:
        batched_hessian = None

    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=False,
        offload_to_disk=False,
        mse=2.0,
        scale_search=method,
    )
    q = Quantizer(qcfg)
    q.configure(perchannel=True, grid=100, maxshrink=0.8)
    q.maxq = q.maxq.to(device)

    for _ in range(3):
        q.find_params_batched(W_3d, weight=True, hessian=batched_hessian)

    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        q.find_params_batched(W_3d, weight=True, hessian=batched_hessian)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    return {
        "rows": rows,
        "columns": columns,
        "group_size": group_size,
        "method": method.value,
        "ms_mean": statistics.mean(times),
        "ms_median": statistics.median(times),
        "ms_min": min(times),
        "ms_max": max(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for group_size in (32, 64, 128):
        for method in (ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID):
            print(f"Benchmarking batched group_size={group_size} method={method.value}", flush=True)
            results.append(benchmark(args.rows, args.columns, group_size, method, "cuda:0"))
            print(json.dumps(results[-1], indent=2), flush=True)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
