# SPDX-License-Identifier: Apache-2.0

"""Run a bounded standard-P32 MLX workload for Xcode Metal profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_mlx import QVQMLXLinear
from gptqmodel.utils.qvq_p32_mlx import (
    qvq_mlx_p32_window_gemv,
    qvq_mlx_repack_p32_planar_to_window,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--bits", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--active-calls", type=int, default=10)
    parser.add_argument("--full-module", action="store_true")
    parser.add_argument("--capture-gputrace", type=Path)
    args = parser.parse_args()
    if args.m <= 0 or args.k <= 0 or args.n <= 0 or args.k % 16 or args.n % 16:
        parser.error("standard P32 requires positive M and K/N divisible by 16")

    generator = torch.Generator().manual_seed(20260831)
    transition_bits = qvq_transition_bits(args.bits)
    tile_count = (args.k // 16) * (args.n // 16)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (128, tile_count),
        generator=generator,
        dtype=torch.int32,
    )
    planar_torch = planar_pack_rows(edges, transition_bits).T.contiguous()
    selector_bits = torch.randint(0, 2, (tile_count * 8,), generator=generator, dtype=torch.uint8)
    selectors = mx.array(pack_qvq_binary_bank_ids(selector_bits).numpy())
    planar = mx.array(planar_torch.numpy())
    bank_alt = mx.array(np.array([2], dtype=np.uint8))
    window = qvq_mlx_repack_p32_planar_to_window(planar, args.bits)
    mx.eval(window)
    dtype = np.float16 if args.full_module else np.float32
    x = mx.array(np.random.default_rng(20260831).standard_normal((args.m, args.k)).astype(dtype))
    module = None
    if args.full_module:
        module = QVQMLXLinear(
            bits=args.bits,
            in_features=args.k,
            out_features=args.n,
            trellis=planar,
            SU=mx.ones((args.k,), dtype=mx.float32),
            SV=mx.ones((args.n,), dtype=mx.float32),
            bank_ids=selectors,
            v2b2_p32=True,
            bank_alt_id=bank_alt,
        )

    def run_once():
        output = (
            module(x)
            if module is not None
            else qvq_mlx_p32_window_gemv(
                x,
                window,
                args.bits,
                out_features=args.n,
                bank_ids=selectors,
                bank_alt_id=2,
            )
        )
        mx.eval(output)
        mx.synchronize()

    print(f"device_info={mx.device_info()}")
    print(f"shape=M{args.m} K{args.k} N{args.n} bits={args.bits}")
    for _ in range(args.warmup):
        run_once()
    if args.capture_gputrace is not None:
        args.capture_gputrace.parent.mkdir(parents=True, exist_ok=True)
        mx.metal.start_capture(str(args.capture_gputrace))
    for _ in range(args.active_calls):
        run_once()
    if args.capture_gputrace is not None:
        mx.metal.stop_capture()
        print(f"gputrace={args.capture_gputrace}")


if __name__ == "__main__":
    main()
