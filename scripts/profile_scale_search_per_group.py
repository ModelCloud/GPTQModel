#!/usr/bin/env python3
"""Simulate per-group find_params loop as GPTQ does for group_size=1."""

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


def run_per_group(
    rows: int,
    columns: int,
    group_size: int,
    method: ScaleSearchConfig,
    bits: int = 4,
    grid: int = 100,
    maxshrink: float = 0.8,
    device: str = "cuda",
) -> dict:
    generator = torch.Generator(device=device).manual_seed(42)
    W = torch.randn((rows, columns), generator=generator, device=device, dtype=torch.float32)
    if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
        activations = torch.randn((columns, max(columns // 2, 1)), generator=generator, device=device, dtype=torch.float32)
        H = activations.matmul(activations.t())
        H = (H + H.t()) * 0.5
    else:
        H = torch.diag(torch.rand(columns, device=device, generator=generator) + 0.5)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        offload_to_disk=False,
        mse=2.0,
        scale_search=method,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=grid, maxshrink=maxshrink)
    quantizer.maxq = quantizer.maxq.to(device)

    # Warmup
    for _ in range(2):
        for g in range(0, columns, group_size):
            quantizer.find_params(W[:, g:g+group_size], weight=True, hessian=H[g:g+group_size, g:g+group_size])

    torch.cuda.synchronize()
    repeats = 5
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for g in range(0, columns, group_size):
            quantizer.find_params(W[:, g:g+group_size], weight=True, hessian=H[g:g+group_size, g:g+group_size])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    return {
        "rows": rows,
        "columns": columns,
        "group_size": group_size,
        "num_groups": columns // group_size,
        "method": method.value,
        "ms_mean": statistics.mean(times),
        "ms_median": statistics.median(times),
        "ms_min": min(times),
        "ms_max": max(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for group_size in (32, 64, 128):
        for method in (ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID):
            print(f"group_size={group_size} method={method.value}", flush=True)
            results.append(run_per_group(args.rows, args.columns, group_size, method, device=args.device))
            print(json.dumps(results[-1], indent=2), flush=True)
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
