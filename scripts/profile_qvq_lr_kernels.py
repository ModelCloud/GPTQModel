# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Minimal NVTX harness for profiling V2B2-P32-LR against legacy V2B2-P32."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_LR_RINGS_PER_TILE,
    local_ring_states_from_edges,
    pack_local_ring_states,
    pack_qvq_binary_bank_ids,
    reconstruct_local_ring_inner_weight,
    reconstruct_qvq_inner_weight,
)
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda, qvq_cuda_gemv


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=float, default=3.0, choices=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=11008)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def _repeat(label: str, fn, *, warmup: int, iterations: int) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(label)
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k % 32 or args.n % 8 or args.k % 16 or args.n % 16:
        raise ValueError("--k and --n must be divisible by both LR K32/N8 and legacy K16/N16 tiles")
    if not prewarm_qvq_cuda():
        raise RuntimeError("QVQ CUDA extension failed to load")

    torch.manual_seed(20260829)
    generator = torch.Generator(device="cpu").manual_seed(20260829 + round(args.bits * 2) * 100 + args.k + args.n)
    transition_bits = round(args.bits * 2)

    lr_tiles = (args.k // 32) * (args.n // 8)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (lr_tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, 8),
        generator=generator,
        dtype=torch.int64,
    )
    lr_trellis = pack_local_ring_states(local_ring_states_from_edges(edges, bits=args.bits), bits=args.bits).cuda()
    lr_selectors = torch.randint(
        0, 2, (lr_tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,), generator=generator, dtype=torch.uint8
    )
    lr_bank_ids = pack_qvq_binary_bank_ids(lr_selectors).cuda()

    legacy_tiles = (args.k // 16) * (args.n // 16)
    legacy_edges = torch.randint(
        0,
        1 << transition_bits,
        (128, legacy_tiles),
        generator=generator,
        dtype=torch.int32,
    )
    legacy_trellis = planar_pack_rows(legacy_edges, transition_bits).T.contiguous().cuda()
    legacy_selectors = torch.randint(
        0, 2, (legacy_tiles * 8,), generator=generator, dtype=torch.uint8
    )
    legacy_bank_ids = pack_qvq_binary_bank_ids(legacy_selectors).cuda()

    x = torch.randn((args.m, args.k), generator=generator, dtype=torch.float32).half().cuda()

    def run_lr():
        return qvq_cuda_gemv(
            x,
            lr_trellis,
            args.bits,
            out_features=args.n,
            output_fp32=True,
            bank_ids=lr_bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=3,
        )

    def run_legacy():
        return qvq_cuda_gemv(
            x,
            legacy_trellis,
            args.bits,
            out_features=args.n,
            output_fp32=True,
            bank_ids=legacy_bank_ids,
            v2b2_p32=True,
            bank_alt_id=3,
        )

    _repeat(f"qvq_lr_w{transition_bits}_m{args.m}", run_lr, warmup=args.warmup, iterations=args.iterations)
    _repeat(
        f"qvq_non_lr_w{transition_bits}_m{args.m}",
        run_legacy,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(
        f"profile harness complete: device={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
        f"sms={torch.cuda.get_device_properties().multi_processor_count} "
        f"shape=({args.m},{args.k},{args.n}) bits={args.bits}",
        flush=True,
    )


if __name__ == "__main__":
    main()
