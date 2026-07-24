#!/usr/bin/env python3
"""End-to-end GPTQ quantize timing for grouped scale-search (batched path)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig  # noqa: E402
from gptqmodel.quantization.gptq import GPTQ  # noqa: E402


def benchmark(rows: int, cols: int, group_size: int, method: ScaleSearchConfig, device: str):
    torch.manual_seed(42)
    layer = nn.Linear(cols, rows, bias=False, dtype=torch.float16, device=device)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=False,
        offload_to_disk=False,
        mse=2.0,
        scale_search=method,
    )
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    inp = torch.randn(4, cols, dtype=torch.float16, device=device)
    gptq.add_batch(inp, None)

    gptq.quantize(blocksize=128)
    del gptq

    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        gptq = GPTQ(layer, qcfg=qcfg)
        gptq.quantizer.configure(perchannel=True)
        gptq.add_batch(inp, None)
        torch.cuda.synchronize()
        start = time.perf_counter()
        gptq.quantize(blocksize=128)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    return {
        "rows": rows,
        "cols": cols,
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
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for group_size in (32, 64, 128):
        for method in (ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID):
            print(f"Benchmarking group_size={group_size} method={method.value}", flush=True)
            results.append(benchmark(args.rows, args.cols, group_size, method, "cuda:0"))
            print(json.dumps(results[-1], indent=2), flush=True)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
