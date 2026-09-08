"""Issue a small, isolated grouped rank-8 launch for Nsight Compute."""

from __future__ import annotations

import argparse

import torch

from gptqmodel.utils.qvq_ampere_cuda import (
    prewarm_qvq_ampere_grouped,
    qvq_p32_window_ampere_grouped_packed,
)
from scripts.bench_qvq_grouped_rank8 import build_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument(
        "--preset",
        choices=(
            "qwen38-27b-proxy",
            "qwen38-flash-next-qkv",
            "qwen38-flash-next-gate-up",
        ),
        default="qwen38-27b-proxy",
    )
    args = parser.parse_args()
    prewarm_qvq_ampere_grouped()
    if args.preset == "qwen38-flash-next-qkv":
        case = build_case(
            args.m,
            size_k=2560,
            widths=(12288, 512, 512),
            alt_ids=(3, 1, 1),
            split_counts=None,
        )
    elif args.preset == "qwen38-flash-next-gate-up":
        case = build_case(
            args.m,
            size_k=2560,
            widths=(640, 640),
            alt_ids=(3, 1),
            split_counts=None,
        )
    else:
        case = build_case(args.m)
    input, payload, levels, packed_a, packed_b, rank8_as, rank8_bs, _ = case
    for _ in range(5):
        qvq_p32_window_ampere_grouped_packed(
            input,
            payload,
            levels,
            rank8_as=rank8_as,
            rank8_bs=rank8_bs,
            rank8_packed_a=packed_a,
            rank8_packed_b=packed_b,
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"grouped_rank8_M{args.m}")
    qvq_p32_window_ampere_grouped_packed(
        input,
        payload,
        levels,
        rank8_as=rank8_as,
        rank8_bs=rank8_bs,
        rank8_packed_a=packed_a,
        rank8_packed_b=packed_b,
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
