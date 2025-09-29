# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


# p2p_bandwidth_test.py
# Measure inter-GPU copy bandwidth using chunked fp16 tensors.
# Default: stream 40 GiB total (in 1 GiB chunks) from 0->1 and 1->0.
#
# Usage:
#   python p2p_bandwidth_test.py             # 40 GiB total, 1 GiB chunks
#   python p2p_bandwidth_test.py --total-gib 80 --chunk-gib 2
#
# Notes:
# - This avoids allocating a single 40 GiB tensor (which would OOM or be risky).
# - If P2P is available, CUDA will use it; otherwise it falls back to host staging.
# - For accurate timing we synchronize before/after and use perf_counter.

import argparse
import math
import time

import torch


def gib_to_elems_fp16(gib: float) -> int:
    # 1 GiB = 1024**3 bytes; fp16 = 2 bytes/elem
    return int((gib * (1024**3)) // 2)

def format_gibs_per_s(bytes_moved, seconds):
    return (bytes_moved / (1024**3)) / seconds

def run_direction(src_dev: int, dst_dev: int, total_gib: float, chunk_gib: float) -> float:
    assert total_gib > 0 and chunk_gib > 0 and total_gib >= chunk_gib
    n_chunks = math.ceil(total_gib / chunk_gib)
    # Round chunk so that n_chunks * chunk_gib >= total_gib
    chunk_elems = gib_to_elems_fp16(chunk_gib)

    # Pre-allocate reusable src/dst chunk buffers
    with torch.cuda.device(src_dev):
        src = torch.empty(chunk_elems, dtype=torch.float16, device=f"cuda:{src_dev}")
        # Fill once to avoid lazy allocations later
        src.uniform_()

    with torch.cuda.device(dst_dev):
        dst = torch.empty(chunk_elems, dtype=torch.float16, device=f"cuda:{dst_dev}")

    # Warmup: single copy (not timed)
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)

    # Timed streaming of N chunks
    bytes_per_chunk = chunk_elems * 2  # fp16 = 2 bytes
    total_bytes = n_chunks * bytes_per_chunk

    t0 = time.perf_counter()
    for _ in range(n_chunks):
        # reuse the same buffers; content doesn't matter for bandwidth
        dst.copy_(src, non_blocking=True)
    # Ensure all queued copies are complete
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)
    t1 = time.perf_counter()

    seconds = t1 - t0
    gibs = total_bytes / (1024**3)
    bw = format_gibs_per_s(total_bytes, seconds)
    print(f"[cuda:{src_dev} -> cuda:{dst_dev}] Transferred {gibs:.2f} GiB in {seconds:.3f} s -> {bw:.2f} GiB/s")
    return bw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=0, help="source GPU id")
    parser.add_argument("--dst", type=int, default=1, help="destination GPU id")
    parser.add_argument("--total-gib", type=float, default=40.0, help="total GiB to stream per direction")
    parser.add_argument("--chunk-gib", type=float, default=1.0, help="chunk size GiB per copy")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("Need at least 2 CUDA devices.")

    # Basic info
    print(f"Detected {torch.cuda.device_count()} CUDA devices.")
    print(f"Testing {args.total_gib:.2f} GiB total per direction in {args.chunk_gib:.2f} GiB chunks.")
    print(f"CUDA P2P (device_can_access_peer): "
          f"{args.src}->{args.dst}={torch.cuda.can_device_access_peer(args.src, args.dst)}, "
          f"{args.dst}->{args.src}={torch.cuda.can_device_access_peer(args.dst, args.src)}")

    # Run both directions
    bw_fwd = run_direction(args.src, args.dst, args.total_gib, args.chunk_gib)
    bw_bwd = run_direction(args.dst, args.src, args.total_gib, args.chunk_gib)

    # Summary
    print(f"Average bandwidth: {(bw_fwd + bw_bwd)/2:.2f} GiB/s")

if __name__ == "__main__":
    main()
