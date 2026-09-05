#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Launch one folded QVQ public forward repeatedly for rocprofv3 tracing."""

from __future__ import annotations

import argparse
import os
import time

from benchmark_qvq_p32_amd_dispatch_sweep import QWEN38_27B_SHAPES, _rocm_snapshot
from benchmark_qvq_p32_amd_fold_ceiling import SHAPE_AXES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--shape", choices=tuple(shape[0] for shape in QWEN38_27B_SHAPES), required=True)
    parser.add_argument("--bits", type=float, default=3.0, choices=(2.0, 2.5, 3.0, 3.5))
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def _idle_gate(physical_gpu: int) -> None:
    for sample in range(3):
        snapshot = _rocm_snapshot(physical_gpu)
        if snapshot["utilization_percent"] != 0 or snapshot["process_ids"]:
            raise RuntimeError(f"ROCm idle gate failed on sample {sample + 1}: {snapshot}")
        if sample < 2:
            time.sleep(1.0)
    print(
        f"ROCm idle gate: physical={physical_gpu} pci={snapshot['pci_bus_id']} "
        f"unique_id={snapshot['unique_id']} utilization=0% samples=3 valid=True",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    _idle_gate(args.physical_gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/qvq-triton-mi355x")

    import torch

    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile

    _, k, n = next(shape for shape in QWEN38_27B_SHAPES if shape[0] == args.shape)
    input_hadamard, output_hadamard = SHAPE_AXES[args.shape]
    generator = torch.Generator(device="cuda").manual_seed(20260904 + int(args.bits * 10) + k + n)
    tile_count = (k // 16) * (n // 16)
    planar = torch.randint(
        -(1 << 31),
        1 << 31,
        (tile_count, qvq_words_per_tile(args.bits, vector_size=2)),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    bank_ids = pack_qvq_binary_bank_ids(
        torch.randint(0, 2, (tile_count * 8,), dtype=torch.uint8, device="cuda", generator=generator)
    )
    layer = QVQLinear(
        bits=args.bits,
        in_features=k,
        out_features=n,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
        tensors={
            "trellis": planar,
            "SU": torch.ones(k, dtype=torch.float32, device="cuda"),
            "SV": torch.ones(n, dtype=torch.float32, device="cuda"),
            "bank_ids": bank_ids,
            "bank_alt_id": torch.tensor([3], dtype=torch.uint8, device="cuda"),
        },
    ).eval()
    x = torch.randn((args.m, k), dtype=torch.float16, device="cuda", generator=generator) * 0.01

    layer(x)
    for _ in range(args.warmup):
        layer(x)
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        layer(x)
    torch.cuda.synchronize()
    window = layer._qvq_cuda_window_cache[3]
    assert window._qvq_p32_amd_dense_cache is None
    print(
        f"profiled folded public forward shape={args.shape} W{args.bits:g} M={args.m} "
        f"K={k} N={n} launches={args.iterations}",
        flush=True,
    )


if __name__ == "__main__":
    main()
