#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A/B two libttbfp builds in one process.

    OMP_WAIT_POLICY=passive OMP_NUM_THREADS=8 \
        python bench/ab_compare.py --old /tmp/old.so --new ./libttbfp.so

Both libraries are loaded side by side and run on the same buffers, back to
back, so host load and page-cache state cancel out.  Timing uses process CPU
time: wall clock is unreliable on a busy host (samples quantize to the
scheduler slice, and spinning OpenMP workers inflate it further).

The script also asserts that both builds produce bitwise identical output.
"""

from __future__ import annotations

import argparse
import ctypes as C
import os
import sys
import time

import numpy as np

P = C.c_void_p


def load(path):
    lib = C.CDLL(path)
    lib.ttbfp_quantize.argtypes = [C.c_int, P, C.c_int64, C.c_int64, P, P]
    lib.ttbfp_quantize.restype = None
    lib.ttbfp_matmul.argtypes = [C.c_int, P, C.c_int64, P, P, C.c_int64,
                                 C.c_int64, C.c_int64, P, C.c_int64, C.c_int]
    lib.ttbfp_matmul.restype = None
    if hasattr(lib, "ttbfp_transpose_packed"):
        lib.ttbfp_transpose_packed.argtypes = [C.c_int, P, P, C.c_int64,
                                               C.c_int64, P, P]
        lib.ttbfp_transpose_packed.restype = None
        lib.ttbfp_matmul_prepacked.argtypes = [C.c_int, P, C.c_int64, P, P,
                                               C.c_int64, C.c_int64, C.c_int64,
                                               P, C.c_int64, C.c_int]
        lib.ttbfp_matmul_prepacked.restype = None
    if hasattr(lib, "ttbfp_matmul_prepacked_s8"):
        lib.ttbfp_convert_s8.argtypes = [C.c_int, P, P, C.c_int64, C.c_int64,
                                         P, P]
        lib.ttbfp_convert_s8.restype = None
        lib.ttbfp_matmul_prepacked_s8.argtypes = [C.c_int, P, C.c_int64, P, P,
                                                  C.c_int64, C.c_int64,
                                                  C.c_int64, P, C.c_int64,
                                                  C.c_int]
        lib.ttbfp_matmul_prepacked_s8.restype = None
    return lib


def cpu_ms(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(3):
        t = time.process_time()
        for _ in range(reps):
            fn()
        best = min(best, (time.process_time() - t) / reps * 1e3)
    return best


def pack(lib, w, fmt):
    k, n = w.shape
    nblk = (n + 15) // 16
    db = {8: 16, 4: 8, 2: 4}[fmt]
    e = np.zeros((k, nblk), np.uint8)
    d = np.zeros((k, nblk * db), np.uint8)
    lib.ttbfp_quantize(fmt, w.ctypes.data, k, n, e.ctypes.data, d.ctypes.data)
    return e, d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="baseline libttbfp.so")
    ap.add_argument("--new", default="./libttbfp.so")
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--rows", default="1,4,16,64")
    ap.add_argument("--formats", default="8,4")
    ap.add_argument("--reps-small", type=int, default=20)
    ap.add_argument("--reps-large", type=int, default=8)
    args = ap.parse_args(argv)

    if os.environ.get("OMP_WAIT_POLICY", "").lower() != "passive":
        print("warning: set OMP_WAIT_POLICY=passive or the numbers are "
              "inflated by spinning OpenMP workers", file=sys.stderr)

    old = load(args.old)
    new = load(args.new)
    rng = np.random.default_rng(0)
    k, n = args.K, args.N
    w = rng.standard_normal((k, n)).astype(np.float32)

    rows = [int(x) for x in args.rows.split(",")]
    fmts = [int(x) for x in args.formats.split(",")]
    worst = 0.0
    for fmt in fmts:
        e, d = pack(new, w, fmt)
        have_pre = hasattr(new, "ttbfp_matmul_prepacked")
        if have_pre:
            nblk = e.shape[1]
            db = {8: 16, 4: 8, 2: 4}[fmt]
            et = np.empty((nblk, k), np.uint8)
            dt = np.empty((nblk, k * db), np.uint8)
            new.ttbfp_transpose_packed(fmt, e.ctypes.data, d.ctypes.data, k, n,
                                       et.ctypes.data, dt.ctypes.data)
        have_s8 = (fmt == 8 and
                   hasattr(new, "ttbfp_matmul_prepacked_s8"))
        if have_s8:
            nblk = e.shape[1]
            est = np.empty((nblk, k), np.uint8)
            s8 = np.empty((nblk, k * 16), np.uint8)
            new.ttbfp_convert_s8(fmt, e.ctypes.data, d.ctypes.data, k, n,
                                 est.ctypes.data, s8.ctypes.data)
        for m in rows:
            a = rng.standard_normal((m, k)).astype(np.float32)
            reps = args.reps_small if m <= 4 else args.reps_large
            out = np.empty((m, n), np.float32)
            ref = np.empty((m, n), np.float32)

            t_old = cpu_ms(lambda: old.ttbfp_matmul(
                fmt, a.ctypes.data, k, e.ctypes.data, d.ctypes.data, m, k, n,
                ref.ctypes.data, n, 1), reps)
            t_row = cpu_ms(lambda: new.ttbfp_matmul(
                fmt, a.ctypes.data, k, e.ctypes.data, d.ctypes.data, m, k, n,
                out.ctypes.data, n, 1), reps)
            line = (f"fmt={fmt} M={m:<3} old={t_old:7.2f}  "
                    f"row-major={t_row:7.2f}")
            best = t_row
            if have_pre:
                t_pre = cpu_ms(lambda: new.ttbfp_matmul_prepacked(
                    fmt, a.ctypes.data, k, et.ctypes.data, dt.ctypes.data, m,
                    k, n, out.ctypes.data, n, 1), reps)
                line += f"  block-major={t_pre:7.2f}"
                best = t_pre
                if have_s8:
                    t_s8 = cpu_ms(lambda: new.ttbfp_matmul_prepacked_s8(
                        fmt, a.ctypes.data, k, est.ctypes.data, s8.ctypes.data,
                        m, k, n, out.ctypes.data, n, 1), reps)
                    line += f"  int8={t_s8:7.2f} ms  x{t_old/t_s8:5.2f}"
                    best = t_s8
            else:
                line += f" ms  x{t_old/t_row:5.2f}"
            same = np.array_equal(ref.view(np.uint32), out.view(np.uint32))
            worst = max(worst, t_old / best)
            print(line + f"  bitwise={same}", flush=True)
            if not same:
                print("ERROR: builds disagree; refusing to report speedups",
                      file=sys.stderr)
                return 1
    print(f"best speedup: {worst:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
