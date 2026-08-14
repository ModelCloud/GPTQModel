"""Benchmark the native MLX QVQ V4 inner GEMV on representative decoder shapes."""

from __future__ import annotations

import argparse
import ctypes
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _performance_qos() -> bool:
    if platform.system() != "Darwin":
        return True
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    set_qos = libsystem.pthread_set_qos_class_self_np
    set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
    set_qos.restype = ctypes.c_int
    return set_qos(0x21, 0) == 0


def _median_ms(fn, *, warmup: int, samples: int, iterations: int) -> float:
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        outputs = [fn() for _ in range(iterations)]
        mx.eval(*outputs)
        timings.append((time.perf_counter() - started) * 1000 / iterations)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    from gptqmodel.quantization.qvq_rates import qvq_transition_bits
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    if not _performance_qos():
        raise RuntimeError("failed to request Darwin user-interactive QoS")
    shapes = (
        (1, 2048, 2048),
        (1, 2048, 8192),
        (1, 8192, 2048),
        (4, 2048, 2048),
        (4, 8192, 2048),
        (16, 2048, 8192),
        (32, 8192, 8192),
    )
    print("Rate  M   KxN          median ms  payload GB/s")
    print("----  --  -----------  ---------  ------------")
    for bits in (1, 1.5, 2, 2.5, 3, 3.5, 4):
        transition_bits = qvq_transition_bits(bits, vector_size=4)
        for m, k, n in shapes:
            rng = np.random.default_rng(20260814 + transition_bits + m + k + n)
            x = mx.array((rng.standard_normal((m, k)) * 0.1).astype(np.float16))
            trellis = mx.array(
                rng.integers(
                    -(1 << 31),
                    (1 << 31) - 1,
                    size=((k // 16) * (n // 16), 2 * transition_bits),
                    dtype=np.int32,
                )
            )
            fn = lambda x=x, trellis=trellis, bits=bits, n=n: qvq_mlx_gemv(
                x, trellis, bits, out_features=n, vector_size=4
            )
            median = _median_ms(
                fn,
                warmup=args.warmup,
                samples=args.samples,
                iterations=args.iterations,
            )
            payload_gbps = k * n * bits / 8 / (median / 1000) / 1e9
            print(f"W{bits:<3}  {m:<2}  {k:>4}x{n:<4}  {median:>9.4f}  {payload_gbps:>12.2f}")


if __name__ == "__main__":
    main()
