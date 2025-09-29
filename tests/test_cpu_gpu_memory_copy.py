# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


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

import argparse
import math
import time

import torch


def gib_to_elems_fp16(gib: float) -> int:
    # 1 GiB = 1024**3 bytes; fp16 = 2 bytes/elem
    return int((gib * (1024**3)) // 2)

def gibs_per_s(bytes_moved: int, seconds: float) -> float:
    return (bytes_moved / (1024**3)) / seconds

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
        dst_gpu.copy_(src_cpu, non_blocking=True)   # non_blocking is effective only if pinned=True
    torch.cuda.synchronize(gpu)
    t1 = time.perf_counter()

    secs = t1 - t0
    bw = gibs_per_s(total_bytes, secs)
    label = "Pinned" if pinned else "Pageable"
    print(f"[CPU to GPU {label}] {total_bytes/(1024**3):.2f} GiB in {secs:.3f} s -> {bw:.2f} GiB/s")
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
        dst_cpu.copy_(src_gpu, non_blocking=True)   # effective non_blocking only if pinned=True
    torch.cuda.synchronize(gpu)
    t1 = time.perf_counter()

    secs = t1 - t0
    bw = gibs_per_s(total_bytes, secs)
    label = "Pinned" if pinned else "Pageable"
    print(f"[GPU to CPU {label}] {total_bytes/(1024**3):.2f} GiB in {secs:.3f} s -> {bw:.2f} GiB/s")
    return bw

def main():
    parser = argparse.ArgumentParser(description="CPU<->GPU bandwidth test with pinned vs pageable CPU memory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to test against")
    parser.add_argument("--total-gib", type=float, default=40.0, help="Total GiB to stream per direction per mode")
    parser.add_argument("--chunk-gib", type=float, default=1.0, help="Chunk size GiB per copy")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    if args.chunk_gib <= 0 or args.total_gib <= 0 or args.total_gib < args.chunk_gib:
        raise SystemExit("Invalid sizes: ensure total-gib >= chunk-gib > 0.")

    print(f"CUDA devices: {torch.cuda.device_count()}, testing GPU {args.gpu}")
    print(f"Total per run: {args.total_gib:.2f} GiB in {args.chunk_gib:.2f} GiB chunks (fp16).")
    print("non_blocking=True is only truly async when CPU memory is pinned.\n")

    # HtoD: pageable vs pinned
    bw_htod_pageable = run_htod(args.gpu, args.total_gib, args.chunk_gib, pinned=False)
    bw_htod_pinned   = run_htod(args.gpu, args.total_gib, args.chunk_gib, pinned=True)

    # DtoH: pageable vs pinned
    bw_dtoh_pageable = run_dtoh(args.gpu, args.total_gib, args.chunk_gib, pinned=False)
    bw_dtoh_pinned   = run_dtoh(args.gpu, args.total_gib, args.chunk_gib, pinned=True)

    print("\nSummary (GiB/s):")
    print(f"  CPU to GPU Pageable: {bw_htod_pageable:.2f}")
    print(f"  CPU to GPU Pinned  : {bw_htod_pinned:.2f}")
    print(f"  GPU to CPU Pageable: {bw_dtoh_pageable:.2f}")
    print(f"  GPU to CPU Pinned  : {bw_dtoh_pinned:.2f}")

if __name__ == "__main__":
    main()
