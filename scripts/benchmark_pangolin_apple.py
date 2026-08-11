# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the Apple MPS or MLX Pangolin GEMV with synchronized timings.

Run on Apple performance cores with, for example::

    taskpolicy -t 1 -l 1 env OMP_NUM_THREADS=12 VECLIB_MAXIMUM_THREADS=12 \
      OPENBLAS_NUM_THREADS=12 PYTHONPATH=. python3 \
      scripts/benchmark_pangolin_apple.py --backend mlx

Use ``--revision origin/main`` and then omit it in a separate process for an
A/B comparison. ``--prevalidated`` models a qlinear caller that already knows
the group-index bounds and 32-lane uniformity.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from gptqmodel.utils.planar_packing import planar_pack_cols, planar_pack_rows


def _load_function(backend: str, revision: str | None):
    module_name = f"pangolin_{backend}"
    function_name = f"pangolin_{backend}_gemv"
    if revision is None:
        module = __import__(f"gptqmodel.utils.{module_name}", fromlist=[function_name])
        return getattr(module, function_name)

    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{revision}:gptqmodel/utils/{module_name}.py",
        ],
        cwd=root,
        text=True,
    )
    qualified_name = f"gptqmodel.utils._benchmark_{module_name}"
    with tempfile.TemporaryDirectory() as directory:
        module_path = Path(directory) / f"{module_name}.py"
        module_path.write_text(source)
        spec = importlib.util.spec_from_file_location(qualified_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load {revision}:{module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(qualified_name, None)
    return getattr(module, function_name)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mps", "mlx"), required=True)
    parser.add_argument("--revision")
    parser.add_argument("--bits", type=_parse_csv_ints, default=tuple(range(2, 9)))
    parser.add_argument("--m", type=_parse_csv_ints, default=(1, 2, 3))
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--prevalidated", action="store_true")
    args = parser.parse_args()

    if args.k % 32 or args.n % 32 or args.k % args.group_size:
        raise ValueError("K/N must be divisible by 32 and K by group size")

    gemv = _load_function(args.backend, args.revision)
    supports_uniform_hint = "_g_idx_block_uniform" in inspect.signature(gemv).parameters
    if args.backend == "mlx":
        import mlx.core as mx

        convert = lambda tensor: mx.array(tensor.numpy())
        synchronize = mx.eval
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        convert = lambda tensor: tensor.to("mps")
        synchronize = lambda output: torch.mps.synchronize()

    groups = args.k // args.group_size
    print(
        f"backend={args.backend} revision={args.revision or 'working-tree'} "
        f"K={args.k} N={args.n} group_size={args.group_size} "
        f"warmup={args.warmup} samples={args.samples} "
        f"iterations={args.iterations} prevalidated={args.prevalidated}"
    )
    print("bits," + ",".join(f"M{m}" for m in args.m))
    for bits in args.bits:
        generator = torch.Generator().manual_seed(15000 + bits)
        codes = torch.randint(
            0,
            1 << bits,
            (args.k, args.n),
            generator=generator,
            dtype=torch.int32,
        )
        zeros = torch.randint(
            0,
            1 << bits,
            (groups, args.n),
            generator=generator,
            dtype=torch.int32,
        )
        scales = (torch.rand((groups, args.n), generator=generator) * 0.2 + 0.01).half()
        packed = tuple(
            map(
                convert,
                (
                    planar_pack_rows(codes, bits),
                    scales,
                    planar_pack_cols(zeros, bits),
                    torch.arange(args.k, dtype=torch.int32) // args.group_size,
                ),
            )
        )
        medians = []
        for m in args.m:
            x = convert(torch.randn((m, args.k), generator=generator).half())
            keywords = {"planar": bits in (3, 5, 6, 7)}
            if args.prevalidated:
                keywords["_g_idx_validated"] = True
                if supports_uniform_hint:
                    keywords["_g_idx_block_uniform"] = True

            def call(x=x, packed=packed, bits=bits, keywords=keywords):
                return gemv(x, *packed, bits, **keywords)

            for _ in range(args.warmup):
                synchronize(call())
            samples = []
            for _ in range(args.samples):
                started = time.perf_counter_ns()
                for _ in range(args.iterations):
                    synchronize(call())
                samples.append(
                    (time.perf_counter_ns() - started) / args.iterations / 1_000_000
                )
            medians.append(statistics.median(samples))
        print(f"{bits}," + ",".join(f"{value:.4f}" for value in medians))


if __name__ == "__main__":
    main()
