# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import argparse
import math
import time

import torch

from models.model_test import ModelTest

# cpu_gpu_bandwidth_test.py
# Measure HtoD and DtoH bandwidth with pageable vs pinned CPU memory.
#
# Usage examples:
#   python cpu_gpu_bandwidth_test.py                        # 40 GiB total, 1 GiB chunks, GPU:0
#   python cpu_gpu_bandwidth_test.py --total-gib 80 --chunk-gib 2 --gpu 1
#
# Notes:
# - We stream in chunks to avoid allocating a single massive tensor.
# - For HtoD, pinned CPU memory + non_blocking=True is required for true async copies.
# - For DtoH, pinned CPU memory also enables non_blocking transfers.
# - We synchronize before/after timing to get accurate results.
# - dtype is fp16 to match your earlier test (2 bytes/elem).


def gib_to_elems_fp16(gib: float) -> int:
    # 1 GiB = 1024**3 bytes; fp16 = 2 bytes/elem
    return int((gib * (1024 ** 3)) // 2)


def gibs_per_s(bytes_moved: int, seconds: float) -> float:
    return (bytes_moved / (1024 ** 3)) / seconds


def make_cpu_tensor(num_elems: int, pin: bool) -> torch.Tensor:
    # Pageable vs pinned CPU tensor
    return torch.empty(num_elems, dtype=torch.float16, device="cpu", pin_memory=pin)


def make_gpu_tensor(num_elems: int, gpu: int) -> torch.Tensor:
    with torch.cuda.device(gpu):
        return torch.empty(num_elems, dtype=torch.float16, device=f"cuda:{gpu}")


def run_htod(gpu: int, total_gib: float, chunk_gib: float, pinned: bool) -> float:
    n_chunks = math.ceil(total_gib / chunk_gib)
    chunk_elems = gib_to_elems_fp16(chunk_gib)
    bytes_per_chunk = chunk_elems * 2
    total_bytes = n_chunks * bytes_per_chunk

    # Buffers
    src_cpu = make_cpu_tensor(chunk_elems, pin=pinned)
    # Touch once to ensure physical allocation before timing
    src_cpu.uniform_()

    dst_gpu = make_gpu_tensor(chunk_elems, gpu)

    # Warmup (not timed)
    dst_gpu.copy_(src_cpu, non_blocking=True)
    torch.cuda.synchronize(gpu)

    # Timed loop
    t0 = time.perf_counter()
    for _ in range(n_chunks):
        dst_gpu.copy_(src_cpu, non_blocking=True)  # non_blocking is effective only if pinned=True
    torch.cuda.synchronize(gpu)
    t1 = time.perf_counter()

    secs = t1 - t0
    bw = gibs_per_s(total_bytes, secs)
    label = "Pinned" if pinned else "Pageable"
    print(f"[CPU to GPU {label}] {total_bytes / (1024 ** 3):.2f} GiB in {secs:.3f} s -> {bw:.2f} GiB/s")
    return bw


def run_dtoh(gpu: int, total_gib: float, chunk_gib: float, pinned: bool) -> float:
    n_chunks = math.ceil(total_gib / chunk_gib)
    chunk_elems = gib_to_elems_fp16(chunk_gib)
    bytes_per_chunk = chunk_elems * 2
    total_bytes = n_chunks * bytes_per_chunk

    # Buffers
    src_gpu = make_gpu_tensor(chunk_elems, gpu)
    src_gpu.uniform_()

    dst_cpu = make_cpu_tensor(chunk_elems, pin=pinned)

    # Warmup (not timed)
    dst_cpu.copy_(src_gpu, non_blocking=True)
    torch.cuda.synchronize(gpu)

    # Timed loop
    t0 = time.perf_counter()
    for _ in range(n_chunks):
        dst_cpu.copy_(src_gpu, non_blocking=True)  # effective non_blocking only if pinned=True
    torch.cuda.synchronize(gpu)
    t1 = time.perf_counter()

    secs = t1 - t0
    bw = gibs_per_s(total_bytes, secs)
    label = "Pinned" if pinned else "Pageable"
    print(f"[GPU to CPU {label}] {total_bytes / (1024 ** 3):.2f} GiB in {secs:.3f} s -> {bw:.2f} GiB/s")
    return bw



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

def cpu_cpu():
    parser = argparse.ArgumentParser(description="CPU<->GPU bandwidth test with pinned vs pageable CPU memory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to test against")
    parser.add_argument("--total-gib", type=float, default=40.0, help="Total GiB to stream per direction per mode")
    parser.add_argument("--chunk-gib", type=float, default=1.0, help="Chunk size GiB per copy")
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    if args.chunk_gib <= 0 or args.total_gib <= 0 or args.total_gib < args.chunk_gib:
        raise SystemExit("Invalid sizes: ensure total-gib >= chunk-gib > 0.")

    print(f"CUDA devices: {torch.cuda.device_count()}, testing GPU {args.gpu}")
    print(f"Total per run: {args.total_gib:.2f} GiB in {args.chunk_gib:.2f} GiB chunks (fp16).")
    print("non_blocking=True is only truly async when CPU memory is pinned.\n")

    # HtoD: pageable vs pinned
    bw_htod_pageable = run_htod(args.gpu, args.total_gib, args.chunk_gib, pinned=False)
    bw_htod_pinned = run_htod(args.gpu, args.total_gib, args.chunk_gib, pinned=True)

    # DtoH: pageable vs pinned
    bw_dtoh_pageable = run_dtoh(args.gpu, args.total_gib, args.chunk_gib, pinned=False)
    bw_dtoh_pinned = run_dtoh(args.gpu, args.total_gib, args.chunk_gib, pinned=True)

    print("\nSummary (GiB/s):")
    print(f"  CPU to GPU Pageable: {bw_htod_pageable:.2f}")
    print(f"  CPU to GPU Pinned  : {bw_htod_pinned:.2f}")
    print(f"  GPU to CPU Pageable: {bw_dtoh_pageable:.2f}")
    print(f"  GPU to CPU Pinned  : {bw_dtoh_pinned:.2f}")


def gib_to_elems_fp16(gib: float) -> int:
    # 1 GiB = 1024**3 bytes; fp16 = 2 bytes/elem
    return int((gib * (1024 ** 3)) // 2)


def format_gibs_per_s(bytes_moved, seconds):
    return (bytes_moved / (1024 ** 3)) / seconds


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
    gibs = total_bytes / (1024 ** 3)
    bw = format_gibs_per_s(total_bytes, seconds)
    print(f"[cuda:{src_dev} -> cuda:{dst_dev}] Transferred {gibs:.2f} GiB in {seconds:.3f} s -> {bw:.2f} GiB/s")
    return bw


def gpu_gpu():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=0, help="source GPU id")
    parser.add_argument("--dst", type=int, default=1, help="destination GPU id")
    parser.add_argument("--total-gib", type=float, default=40.0, help="total GiB to stream per direction")
    parser.add_argument("--chunk-gib", type=float, default=1.0, help="chunk size GiB per copy")
    args, _ = parser.parse_known_args()

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
    print(f"Average bandwidth: {(bw_fwd + bw_bwd) / 2:.2f} GiB/s")


class Test(ModelTest):
    def test(self):
        cpu_cpu()
        gpu_gpu()
