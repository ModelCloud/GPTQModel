#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""One-kernel NCU harness for Qwen3.8 W3 production LR or RS-WGMMA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_LR_RING_STEPS,
    QVQ_V2B2_P32_LR_RINGS_PER_TILE,
    local_ring_states_from_edges,
    pack_local_ring_states,
    pack_qvq_binary_bank_ids,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda, qvq_cuda_gemv
from gptqmodel.utils.qvq_wgmma_cuda import qvq_wgmma_w3_m16
from scripts.benchmark_qvq_lr_vs_gptq_llama32_1b import QWEN38_27B_SHAPES


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("production", "wgmma"), required=True)
    parser.add_argument(
        "--shape",
        choices=tuple(case.name for case in QWEN38_27B_SHAPES),
        default="qwen38_mlp_down",
    )
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.split <= 0:
        parser.error("--split must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    return args


def _profiler_call(name: str) -> None:
    result = getattr(torch.cuda.cudart(), name)()
    if result not in (None, 0):
        raise RuntimeError(f"{name} failed with CUDA status {result}")


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        raise RuntimeError(f"H200 is required, got {properties.name} CC {properties.major}.{properties.minor}")

    case = next(case for case in QWEN38_27B_SHAPES if case.name == args.shape)
    generator = torch.Generator().manual_seed(20260830)
    tiles = (case.in_features // 32) * (case.out_features // 8)
    edges = torch.randint(
        0,
        1 << 6,
        (tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    trellis = pack_local_ring_states(local_ring_states_from_edges(edges, bits=3.0), bits=3.0).cuda()
    selectors = torch.randint(
        0,
        2,
        (tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,),
        generator=generator,
        dtype=torch.uint8,
    )
    bank_ids = pack_qvq_binary_bank_ids(selectors).cuda()
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    x = (torch.randn((16, case.in_features), generator=generator, dtype=torch.float32) * 0.1).half().cuda()

    if args.kernel == "production":
        if not prewarm_qvq_cuda():
            raise RuntimeError("QVQ CUDA extension failed to load")

        def call():
            return qvq_cuda_gemv(
                x,
                trellis,
                3.0,
                out_features=case.out_features,
                output_fp32=True,
                bank_ids=bank_ids,
                v2b2_p32_lr=True,
                bank_alt_id=3,
            )
    else:
        def call():
            return qvq_wgmma_w3_m16(
                x,
                trellis,
                levels,
                bank_ids,
                out_features=case.out_features,
                bank_alt_id=3,
                split_count=args.split,
            )

    for _ in range(args.warmup):
        output = call()
    torch.cuda.synchronize()
    _profiler_call("cudaProfilerStart")
    torch.cuda.nvtx.range_push(f"qvq_{args.kernel}_w3_{case.name}_split{args.split}")
    try:
        for _ in range(args.iterations):
            output = call()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        _profiler_call("cudaProfilerStop")
    print(
        f"profile complete: kernel={args.kernel} shape=M16/K{case.in_features}/N{case.out_features} "
        f"split={args.split} output={tuple(output.shape)} dtype={output.dtype} "
        f"device={properties.name} sms={properties.multi_processor_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
