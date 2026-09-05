#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile M64-reuse or M128-reuse for one unsplit Llama down projection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("reuse4", "reuse8"))
    parser.add_argument("--bits", type=float, default=3)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1)
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import _pgc16_levels
    from gptqmodel.utils.qvq_wgmma_cuda import (
        QVQHopperGroupedP32Payload,
        qvq_p32_window_wgmma_group_plan,
        qvq_p32_window_wgmma_grouped_reuse4_packed,
        qvq_p32_window_wgmma_grouped_reuse8_packed,
    )
    from scripts import benchmark_qvq_hopper_large_m_mlp as mlp_bench

    if torch.cuda.get_device_name() != "NVIDIA H100":
        raise RuntimeError("large-M down profiling requires the physical H100")
    device = torch.device("cuda:0")
    parent = mlp_bench._qvq_mlp(torch, args.bits, device)
    down = parent.down_proj
    input = (
        torch.randn(
            (args.m, mlp_bench.INTERMEDIATE),
            generator=torch.Generator(device=device).manual_seed(20260906 + args.m),
            device=device,
        )
        * 0.02
    ).half()
    # Populate the canonical module's transient window and selector cache.
    with torch.inference_mode():
        down._inner_forward(input)
    with down._qvq_cuda_bank_cache_lock:
        window = down._prepare_hopper_p32_window(device)
        cached = down._qvq_cuda_bank_cache
        if cached is None:
            raise RuntimeError("down selector cache was not prepared")
        bank_ids = cached[5]
        bank_alt_id = cached[6]
    levels = _pgc16_levels(device, down.codebook_version)
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        (window,),
        levels,
        (bank_ids,),
        args.bits,
        out_features=(down.out_features,),
        bank_alt_ids=(bank_alt_id,),
        split_counts=(1,),
    )
    payload = QVQHopperGroupedP32Payload(window, bank_ids, plan)
    kernel = (
        qvq_p32_window_wgmma_grouped_reuse4_packed
        if args.variant == "reuse4"
        else qvq_p32_window_wgmma_grouped_reuse8_packed
    )

    def call():
        return kernel(input, payload, levels)

    with torch.inference_mode():
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.replays):
            graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    _main(_args())
