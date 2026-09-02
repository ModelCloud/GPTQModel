#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile one retained grouped gate/up N64 P32 WGMMA launch on H100."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts import benchmark_qvq_phase13_gateup_split as experiment


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=float, choices=experiment.RATES, default=3.0)
    parser.add_argument("--m", type=int, choices=experiment.M_VALUES, default=1)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_group_plan,
        qvq_p32_window_wgmma_grouped_packed,
        qvq_pack_p32_window_hopper_group,
    )

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    windows, selectors = experiment._payloads(torch, args.bits, device)
    plan = qvq_p32_window_wgmma_group_plan(
        torch.empty((16, experiment.K), dtype=torch.float16, device=device),
        windows,
        levels,
        selectors,
        args.bits,
        out_features=experiment.WIDTHS,
        bank_alt_ids=experiment.ALT_IDS,
        split_counts=(1, 1),
    )
    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)
    padded_input = torch.zeros((16, experiment.K), dtype=torch.float16, device=device)
    generator = torch.Generator(device=device).manual_seed(
        20262100 + int(args.bits * 10) * 100 + args.m
    )
    padded_input[: args.m] = (
        torch.randn((args.m, experiment.K), generator=generator, device=device) * 0.02
    ).half()

    def call():
        return qvq_p32_window_wgmma_grouped_packed(padded_input, payload, levels)

    expected = call()
    for _ in range(10):
        actual = call()
    torch.cuda.synchronize(device)
    if not all(
        torch.equal(actual_child, expected_child)
        for actual_child, expected_child in zip(actual, expected)
    ):
        raise RuntimeError("grouped gate/up profile warmup is not bit-exact")

    torch.cuda.cudart().cudaProfilerStart()
    profiled = call()
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()
    if not all(
        torch.equal(actual_child, expected_child)
        for actual_child, expected_child in zip(profiled, expected)
    ):
        raise RuntimeError("profiled grouped gate/up output changed")
    print(
        f"profiled W{args.bits:g} M{args.m} grouped gate/up N64 on "
        f"{device_info['name']} uuid={device_info['uuid']}"
    )


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
