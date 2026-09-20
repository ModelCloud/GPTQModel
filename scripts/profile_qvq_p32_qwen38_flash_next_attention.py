#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Replay one production Flash-Next attention projection CUDA graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts.benchmark_qvq_p32_qwen38_flash_next_model_h100 import _payload


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        choices=("full_qkv", "linear_qkv_z", "attention_output"),
        default="linear_qkv_z",
    )
    parser.add_argument("--bits", type=float, choices=(2.0, 2.5, 3.0, 3.5), default=3.0)
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), default=1)
    args = parser.parse_args()

    import torch

    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.quantization.qvq_rates import qvq_transition_bits
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_h100_grouped_ordered_split_counts,
        qvq_p32_window_wgmma_group_plan,
        qvq_p32_window_wgmma_grouped_ordered_packed,
        qvq_p32_window_wgmma_grouped_packed,
        qvq_p32_window_wgmma_m16_tma_ordered_split,
        qvq_pack_p32_window_hopper_group,
    )

    common._assert_h100(torch)
    device = torch.device("cuda:0")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    transition_bits = qvq_transition_bits(args.bits, vector_size=2)
    generator = torch.Generator(device=device).manual_seed(
        20269200 + int(args.bits * 10) * 100 + args.m
    )

    if args.site == "attention_output":
        payload = _payload(
            torch,
            bits=args.bits,
            k=6144,
            n=2560,
            seed=20269210 + int(args.bits * 10),
        )
        value = (
            torch.randn((args.m, 6144), generator=generator, device=device) * 0.02
        ).half()
        split = {2.0: 12, 2.5: 6, 3.0: 12, 3.5: 24}[args.bits]

        def call():
            return (
                qvq_p32_window_wgmma_m16_tma_ordered_split(
                    value,
                    payload[1],
                    levels,
                    payload[2],
                    args.bits,
                    out_features=2560,
                    bank_alt_id=3,
                    split_count=split,
                ),
            )

    else:
        widths = (12288, 512, 512) if args.site == "full_qkv" else (10240, 6144)
        payloads = tuple(
            _payload(
                torch,
                bits=args.bits,
                k=2560,
                n=width,
                seed=20269220 + int(args.bits * 10) * 10 + index,
            )
            for index, width in enumerate(widths)
        )
        value = torch.zeros((16, 2560), dtype=torch.float16, device=device)
        value[: args.m] = (
            torch.randn((args.m, 2560), generator=generator, device=device) * 0.02
        ).half()
        properties = torch.cuda.get_device_properties(device)
        splits = qvq_h100_grouped_ordered_split_counts(
            device_name=properties.name,
            compute_capability=(properties.major, properties.minor),
            in_features=2560,
            out_features=widths,
            transition_bits=transition_bits,
        )
        if splits is None:
            raise RuntimeError(f"missing split schedule for {args.site}")
        plan = qvq_p32_window_wgmma_group_plan(
            value,
            tuple(payload[1] for payload in payloads),
            levels,
            tuple(payload[2] for payload in payloads),
            args.bits,
            out_features=widths,
            bank_alt_ids=(3,) * len(widths),
            split_counts=splits,
        )
        packed = qvq_pack_p32_window_hopper_group(
            tuple(payload[1] for payload in payloads),
            tuple(payload[2] for payload in payloads),
            plan,
        )
        grouped = (
            qvq_p32_window_wgmma_grouped_ordered_packed
            if any(split != 1 for split in splits)
            else qvq_p32_window_wgmma_grouped_packed
        )

        def call():
            return grouped(value, packed, levels)

    with torch.inference_mode():
        for _ in range(3):
            call()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    print(sum(float(tensor.float().sum().item()) for tensor in output))


if __name__ == "__main__":
    _main()
