#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sweep ordered split-K for the H100 Llama down P32 inner operation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts import benchmark_qvq_hopper_large_m as large_m
from scripts import benchmark_qvq_hopper_large_m_mlp as mlp_bench


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=(2, 2.5, 3, 3.5))
    parser.add_argument("--m-values", nargs="+", type=int, default=(32, 64, 128, 256))
    parser.add_argument("--splits", nargs="+", type=int, default=(1, 2, 4, 8, 16))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qvq_hopper_large_m/down_split_sweep.json"),
    )
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import _pgc16_levels
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_single_large_m_packed,
    )

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    results = []
    for bits in args.rates:
        module = mlp_bench._qvq_mlp(torch, bits, device).down_proj
        inputs = {
            m: (
                torch.randn(
                    (m, mlp_bench.INTERMEDIATE),
                    generator=torch.Generator(device=device).manual_seed(13000 + m),
                    device=device,
                )
                * 0.02
            ).half()
            for m in args.m_values
        }
        # Warm the canonical module caches once outside graph capture.
        module._inner_forward(inputs[args.m_values[0]])
        with module._qvq_cuda_bank_cache_lock:
            window = module._prepare_hopper_p32_window(device)
            packed_selectors = module._qvq_cuda_bank_cache[5]
            alt_id = module._qvq_cuda_bank_cache[6]
        levels = _pgc16_levels(device, module.codebook_version)
        references = {}
        for m, input in inputs.items():
            for split in args.splits:

                def call(
                    input=input,
                    split=split,
                    window=window,
                    levels=levels,
                    packed_selectors=packed_selectors,
                    bits=bits,
                    out_features=module.out_features,
                    alt_id=alt_id,
                ):
                    return (
                        qvq_p32_window_wgmma_single_large_m_packed(
                            input,
                            window,
                            levels,
                            packed_selectors,
                            bits,
                            out_features=out_features,
                            bank_alt_id=alt_id,
                            split_count=split,
                        ),
                    )

                timing, (actual,) = large_m._graph_timing(
                    torch, call, args, device_info
                )
                if split == 1:
                    references[m] = actual
                    max_abs_vs_split1 = 0.0
                else:
                    max_abs_vs_split1 = float(
                        (actual.float() - references[m].float()).abs().max().item()
                    )
                row = {
                    "bits": bits,
                    "mkn": [m, module.in_features, module.out_features],
                    "split_count": split,
                    "timing": timing,
                    "speedup_vs_split1": (
                        timing["median_us"] / timing["median_us"]
                        if split == 1
                        else next(
                            item["timing"]["median_us"]
                            for item in results
                            if item["bits"] == bits
                            and item["mkn"][0] == m
                            and item["split_count"] == 1
                        )
                        / timing["median_us"]
                    ),
                    "max_abs_vs_split1": max_abs_vs_split1,
                }
                results.append(row)
                print(
                    f"W{bits:g} MKN={tuple(row['mkn'])} split={split}: "
                    f"{timing['median_us']:.3f}us "
                    f"vs_split1={row['speedup_vs_split1']:.3f}x "
                    f"max_abs={max_abs_vs_split1:.3e}",
                    flush=True,
                )
    payload = {
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
        },
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
