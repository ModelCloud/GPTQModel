#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""One-kernel NCU harness for exact standard-P32 Qwen3.8 kernels."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    repack_p32_planar_to_window,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    pgc16_levels_for_version,
)
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_wgmma_cuda import (
    qvq_p32_window_wgmma_m16_tma,
    qvq_p32_window_wgmma_m16_tma_ordered_split,
    qvq_p32_window_wgmma_w3_m16,
)


@dataclass(frozen=True)
class ShapeCase:
    """Qwen3.8-27B projection shape used by the Hopper profiling harness."""

    name: str
    in_features: int
    out_features: int


QWEN38_27B_SHAPES = (
    ShapeCase("qwen38_full_q_gate", 5120, 12288),
    ShapeCase("qwen38_full_kv", 5120, 1024),
    ShapeCase("qwen38_attn_out", 6144, 5120),
    ShapeCase("qwen38_linear_qkv", 5120, 10240),
    ShapeCase("qwen38_linear_z", 5120, 6144),
    ShapeCase("qwen38_mlp_gate_up", 5120, 17408),
    ShapeCase("qwen38_mlp_down", 17408, 5120),
    ShapeCase("qwen38_flash_next_full_q", 2560, 12288),
    ShapeCase("qwen38_flash_next_full_kv", 2560, 512),
    ShapeCase("qwen38_flash_next_attn_out", 6144, 2560),
    ShapeCase("qwen38_flash_next_linear_qkv", 2560, 10240),
    ShapeCase("qwen38_flash_next_linear_z", 2560, 6144),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=("p32_wgmma", "p32_tma_wgmma", "p32_tma_wgmma_ordered"),
        required=True,
    )
    parser.add_argument(
        "--shape",
        choices=tuple(case.name for case in QWEN38_27B_SHAPES),
        default="qwen38_mlp_down",
    )
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--bits", type=float, choices=(2.0, 2.5, 3.0, 3.5), default=3.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.split <= 0:
        parser.error("--split must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.kernel == "p32_wgmma" and args.bits != 3.0:
        parser.error("the synchronous P32 prototype supports only W3")
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
    if (properties.major, properties.minor) != (9, 0) or not (
        "H100" in properties.name or "H200" in properties.name
    ):
        raise RuntimeError(
            f"an H100/H200 SM90 GPU is required, got {properties.name} "
            f"CC {properties.major}.{properties.minor}"
        )

    case = next(case for case in QWEN38_27B_SHAPES if case.name == args.shape)
    generator = torch.Generator().manual_seed(20260830)
    transition_bits = qvq_transition_bits(args.bits, vector_size=2)
    tiles = (case.in_features // 16) * (case.out_features // 16)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (128, tiles),
        generator=generator,
        dtype=torch.int32,
    )
    planar = planar_pack_rows(edges, transition_bits).T.contiguous()
    trellis = repack_p32_planar_to_window(planar, bits=args.bits).cuda()
    selector_count = tiles * 8
    selectors = torch.randint(0, 2, (selector_count,), generator=generator, dtype=torch.uint8)
    bank_ids = pack_qvq_binary_bank_ids(selectors).cuda()
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    x = (torch.randn((16, case.in_features), generator=generator, dtype=torch.float32) * 0.1).half().cuda()

    if args.kernel == "p32_wgmma":
        def call():
            return qvq_p32_window_wgmma_w3_m16(
                x,
                trellis,
                levels,
                bank_ids,
                out_features=case.out_features,
                bank_alt_id=3,
                split_count=args.split,
            )
    elif args.kernel == "p32_tma_wgmma":
        def call():
            return qvq_p32_window_wgmma_m16_tma(
                x,
                trellis,
                levels,
                bank_ids,
                args.bits,
                out_features=case.out_features,
                bank_alt_id=3,
                split_count=args.split,
            )
    else:
        def call():
            return qvq_p32_window_wgmma_m16_tma_ordered_split(
                x,
                trellis,
                levels,
                bank_ids,
                args.bits,
                out_features=case.out_features,
                bank_alt_id=3,
                split_count=args.split,
            )

    for _ in range(args.warmup):
        output = call()
    torch.cuda.synchronize()
    _profiler_call("cudaProfilerStart")
    torch.cuda.nvtx.range_push(f"qvq_{args.kernel}_w{args.bits:g}_{case.name}_split{args.split}")
    try:
        for _ in range(args.iterations):
            output = call()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        _profiler_call("cudaProfilerStop")
    print(
        f"profile complete: kernel={args.kernel} rate=W{args.bits:g} "
        f"shape=M16/K{case.in_features}/N{case.out_features} "
        f"split={args.split} output={tuple(output.shape)} dtype={output.dtype} "
        f"device={properties.name} sms={properties.multi_processor_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
