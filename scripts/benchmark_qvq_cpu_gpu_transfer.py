#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure pinned transfers under externally selected CPU/NUMA placement."""

import argparse
import json
import statistics
import time
from pathlib import Path

from cpu_runtime_inventory import cpu_runtime_inventory
from gpu_idle_preflight import bootstrap_gpu_idle_preflight, recheck_gpu_exclusivity


def numa_mapping(address):
    for line in Path("/proc/self/maps").read_text().splitlines():
        low, high = (int(value, 16) for value in line.split()[0].split("-"))
        if low <= address < high:
            return [
                entry for entry in Path("/proc/self/numa_maps").read_text().splitlines()
                if int(entry.split()[0], 16) == low
            ]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="external-placement")
    parser.add_argument("--mib", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.mib <= 0 or args.repeats <= 0:
        parser.error("mib and repeats must be positive")
    inventory = cpu_runtime_inventory()
    state = bootstrap_gpu_idle_preflight([])
    import torch

    torch.set_num_threads(min(4, inventory["allowed_logical_cpus"]))
    torch.set_num_interop_threads(1)
    size = args.mib * 1024 * 1024
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True).fill_(23)
    gpu = torch.empty_like(host, device="cuda").fill_(23)
    torch.cuda.synchronize()
    rows = []
    for label, source, destination in (("d2h", gpu, host), ("h2d", host, gpu)):
        for _ in range(5):
            destination.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        recheck_gpu_exclusivity(state)
        samples, walls = [], []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            start.record()
            destination.copy_(source, non_blocking=True)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            walls.append((time.perf_counter() - before) * 1000)
        rows.append({
            "direction": label, "median_cuda_ms": statistics.median(samples),
            "median_wall_ms": statistics.median(walls),
            "GBps": size / 1e6 / statistics.median(samples),
            "cuda_ms": samples, "wall_ms": walls,
        })
    result = {
        "label": args.label, "cpu": inventory, "gpu": state.as_dict(), "bytes": size,
        "torch_version": torch.__version__, "numa_mapping": numa_mapping(host.data_ptr()), "rows": rows,
        "scope": "Pinned-copy bandwidth only; not a collector speedup",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
