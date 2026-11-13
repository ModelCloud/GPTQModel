# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks for the ttbfp CPU kernels.

    make bench

Timing uses process CPU time, which is far more stable than wall clock on a
shared host.  Set OMP_NUM_THREADS to change the thread count; set
OMP_WAIT_POLICY=passive to stop idle OpenMP threads from spinning.
"""

from __future__ import annotations

import os
import time

import numpy as np

import ttbfp


def cpu_ms(fn, reps=10, warmup=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(3):
        t = time.process_time()
        for _ in range(reps):
            fn()
        best = min(best, (time.process_time() - t) / reps * 1e3)
    return best


def bench_matmul(fmt, M, K, N):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((K, N)).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    et, dt = ttbfp.transpose_packed(e, d, fmt)
    have_s8 = fmt == ttbfp.BF8_B
    if have_s8:
        est, s8 = ttbfp.prepack_s8(w, fmt)

    reps = 20 if M <= 4 else 8
    t_row = cpu_ms(lambda: ttbfp.matmul(a, e, d, fmt, cols=N), reps)
    t_pre = cpu_ms(lambda: ttbfp.matmul_prepacked(a, et, dt, fmt, cols=N), reps)
    if have_s8:
        t_s8 = cpu_ms(lambda: ttbfp.matmul_prepacked_s8(a, est, s8, fmt, cols=N),
                      reps)

    b = ttbfp.unpack(e, d, fmt, cols=N)
    t_deq = cpu_ms(lambda: ttbfp.unpack(e, d, fmt, cols=N), reps)
    t_blas = cpu_ms(lambda: a @ b, reps)

    name = {8: "bf8_b", 4: "bf4_b", 2: "bf2_b"}[fmt]
    flops = 2.0 * M * K * N
    print(f"{name}  M={M:<4} K={K:<5} N={N:<5}")
    print(f"  row-major packed   : {t_row:8.3f} ms  {flops/(t_row*1e6):7.2f} GFLOP/s")
    print(f"  block-major packed : {t_pre:8.3f} ms  {flops/(t_pre*1e6):7.2f} GFLOP/s"
          f"   ({t_row/t_pre:5.2f}x)")
    if have_s8:
        print(f"  int8 block-major   : {t_s8:8.3f} ms  "
              f"{flops/(t_s8*1e6):7.2f} GFLOP/s   ({t_row/t_s8:5.2f}x)")
    else:
        print("  int8 block-major   : n/a (bf8_b only; block-major stays packed)")
    print(f"  dequant + BLAS     : {t_deq + t_blas:8.3f} ms  "
          f"({t_deq:.3f} dequant + {t_blas:.3f} BLAS)")


def bench_pack(fmt, rows=4096, cols=4096):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    e, d = ttbfp.pack(x, fmt)
    t_pack = cpu_ms(lambda: ttbfp.pack(x, fmt), 5)
    t_tr = cpu_ms(lambda: ttbfp.transpose_packed(e, d, fmt), 5)
    t_unpack = cpu_ms(lambda: ttbfp.unpack(e, d, fmt, cols=cols), 5)
    mb = x.nbytes / 1e6
    name = {8: "bf8_b", 4: "bf4_b", 2: "bf2_b"}[fmt]
    print(f"pack    {name} {rows}x{cols}: {t_pack:8.3f} ms ({mb/t_pack/1e3:6.2f} GB/s in)")
    print(f"prepack {name} {rows}x{cols}: {t_tr:8.3f} ms")
    print(f"unpack  {name} {rows}x{cols}: {t_unpack:8.3f} ms ({mb/t_unpack/1e3:6.2f} GB/s out)")


def main():
    print(f"ttbfp {ttbfp.version()}  isa={ttbfp.cpu_features()}  "
          f"omp_threads={os.environ.get('OMP_NUM_THREADS', 'default')}  "
          f"wait_policy={os.environ.get('OMP_WAIT_POLICY', 'default')}")
    print()
    for fmt in (ttbfp.BF8_B, ttbfp.BF4_B):
        for M in (1, 4, 16, 64):
            bench_matmul(fmt, M, 4096, 4096)
            print()
    for fmt in (ttbfp.BF8_B, ttbfp.BF4_B):
        bench_pack(fmt)
        print()


if __name__ == "__main__":
    main()
