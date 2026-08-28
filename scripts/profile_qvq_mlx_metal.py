# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded LR32 MLX workload for Xcode Metal profiling.

This intentionally profiles the inner kernel with immutable alternative-bank
metadata already resolved, so the trace does not include the compatibility
``bank_alt_id.item()`` path.  Use ``QVQMLXLinear`` benchmarks separately for
complete-module latency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_LR_RINGS_PER_TILE,
    local_ring_states_from_edges,
    pack_local_ring_states,
    pack_qvq_binary_bank_ids,
)
from gptqmodel.utils.qvq_mlx import QVQMLXLinear, qvq_mlx_gemv


def _payload(k: int, n: int, bits: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    tile_count = (k // 32) * (n // 8)
    transition_bits = int(bits * 2)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (tile_count, QVQ_V2B2_P32_LR_RINGS_PER_TILE, 16),
        generator=generator,
        dtype=torch.int64,
    )
    states = local_ring_states_from_edges(edges, bits=bits)
    trellis = pack_local_ring_states(states, bits=bits)
    selectors = torch.randint(0, 2, (tile_count * 8,), generator=generator, dtype=torch.uint8)
    return trellis, pack_qvq_binary_bank_ids(selectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--bits", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--active-calls", type=int, default=10)
    parser.add_argument(
        "--full-module",
        action="store_true",
        help="Profile complete QVQMLXLinear, including Hadamard transforms and epilogue.",
    )
    parser.add_argument("--capture-gputrace", type=Path)
    args = parser.parse_args()

    if args.m <= 0 or args.k <= 0 or args.n <= 0 or args.k % 32 or args.n % 8:
        parser.error("LR32 requires positive M and K/N divisible by K32/N8")

    import mlx.core as mx

    trellis_torch, bank_ids_torch = _payload(args.k, args.n, args.bits, seed=20260827)
    x_dtype = np.float16 if args.full_module else np.float32
    x = mx.array(np.random.default_rng(20260827).standard_normal((args.m, args.k)).astype(x_dtype))
    trellis = mx.array(trellis_torch.numpy())
    bank_ids = mx.array(bank_ids_torch.numpy())
    bank_alt_id = mx.array(np.array([2], dtype=np.uint8))

    module = None
    if args.full_module:
        module = QVQMLXLinear(
            bits=args.bits,
            in_features=args.k,
            out_features=args.n,
            trellis=trellis,
            SU=mx.ones((args.k,), dtype=mx.float32),
            SV=mx.ones((args.n,), dtype=mx.float32),
            vector_size=2,
            trellis_window=16,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=bank_alt_id,
        )

    def run_once():
        if module is None:
            output = qvq_mlx_gemv(
                x,
                trellis,
                args.bits,
                out_features=args.n,
                bank_ids=bank_ids,
                bank_alt_id=bank_alt_id,
                v2b2_p32_lr=True,
                output_fp32=True,
                _bank_alt_id_value=2,
            )
        else:
            output = module(x)
        mx.eval(output)
        mx.synchronize()

    print(f"device={mx.default_device()}")
    print(f"device_info={mx.device_info()}")
    mode = "full-module" if args.full_module else "inner-gemv"
    print(f"mode={mode}")
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
