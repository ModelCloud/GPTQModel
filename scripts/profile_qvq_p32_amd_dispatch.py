#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Launch one exact P32 cached-GEMM dispatch repeatedly for rocprofv3 tracing."""

from __future__ import annotations

import argparse
import os
import time

from benchmark_qvq_p32_amd_dispatch_sweep import _rocm_snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--layout", choices=("view", "contiguous"), default="view")
    parser.add_argument("--blas", choices=("default", "cublas", "cublaslt", "ck"), default="default")
    parser.add_argument("--n-split", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--m-split", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.n_split > 1 and args.m_split > 1:
        parser.error("choose at most one split dimension")
    if args.n % args.n_split:
        parser.error("N must be divisible by --n-split")
    if args.m % args.m_split:
        parser.error("M must be divisible by --m-split")
    return args


def _idle_gate(args: argparse.Namespace) -> dict[str, object]:
    accepted = None
    for sample in range(3):
        accepted = _rocm_snapshot(args.physical_gpu)
        if accepted["utilization_percent"] != 0 or accepted["process_ids"]:
            raise RuntimeError(f"ROCm idle gate failed on sample {sample + 1}: {accepted}")
        if sample < 2:
            time.sleep(1.0)
    assert accepted is not None
    print(
        f"ROCm idle gate: physical={args.physical_gpu} pci={accepted['pci_bus_id']} "
        f"unique_id={accepted['unique_id']} utilization=0% samples=3 valid=True",
        flush=True,
    )
    return accepted


def main() -> None:
    args = _parse_args()
    _idle_gate(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)

    import torch

    x = torch.empty((args.m, args.k), dtype=torch.float16, device="cuda")
    weight_nk = torch.empty((args.n, args.k), dtype=torch.float16, device="cuda")
    weight = weight_nk.t() if args.layout == "view" else weight_nk.t().contiguous()
    torch.backends.cuda.preferred_blas_library(args.blas)

    if args.n_split > 1:
        width = args.n // args.n_split
        grouped_x = x.unsqueeze(0).expand(args.n_split, -1, -1)
        grouped_weight = weight_nk.reshape(args.n_split, width, args.k).transpose(1, 2)

        def launch():
            return (
                torch.bmm(grouped_x, grouped_weight, out_dtype=torch.float32)
                .permute(1, 0, 2)
                .reshape(args.m, args.n)
            )

    elif args.m_split > 1:
        grouped_x = x.reshape(args.m_split, args.m // args.m_split, args.k)
        grouped_weight = weight.unsqueeze(0).expand(args.m_split, -1, -1)

        def launch():
            return torch.bmm(grouped_x, grouped_weight, out_dtype=torch.float32).reshape(args.m, args.n)

    else:

        def launch():
            return torch.mm(x, weight, out_dtype=torch.float32)

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        launch()
    torch.cuda.synchronize()
    print(
        f"profiled M={args.m} K={args.k} N={args.n} layout={args.layout} blas={args.blas} "
        f"n_split={args.n_split} m_split={args.m_split} launches={args.iterations}",
        flush=True,
    )


if __name__ == "__main__":
    main()
