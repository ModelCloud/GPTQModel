# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark for the W2 <4,2,16,fused> family-grid Viterbi op.

Reproduces the synthetic proxy used by docs/qvq_grid_kernel_opt_status.md: the real
workload shape is 3 families x 128 sequences (block-LDLQ site) plus a 3 x 32 case.
Reports ms/call for the fused kernel path (batch >= 40 flattened) with the same op
the pipeline calls, so numbers are comparable across rounds.

Usage: .venv-qvq-profile/bin/python scripts/benchmark_qvq_family_grid.py [--iters 30]
"""

import argparse
import time

import torch

from gptqmodel.quantization.qvq_codecs.pgc16 import pgc16_codebook_v2_bank
from gptqmodel.utils.qvq_cuda import _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op


def build_inputs(batch: int, weighted: bool, constrained: bool, seed: int = 20260823):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    families = 3
    sequences = torch.randn((families, batch, 128, 2), generator=generator, device="cuda")
    codebooks = torch.stack(
        tuple(
            torch.stack(
                (
                    pgc16_codebook_v2_bank(0, bits=2.0, dtype=torch.float32),
                    pgc16_codebook_v2_bank(family, bits=2.0, dtype=torch.float32),
                )
            )
            for family in (1, 2, 3)
        )
    ).to(device="cuda", dtype=torch.float16)
    overlap = (
        torch.randint(0, 1 << 12, (families, batch), generator=generator, device="cuda", dtype=torch.int64)
        if constrained
        else None
    )
    step_weights = (
        (0.1 + torch.rand((families, batch, 128), generator=generator, device="cuda")) if weighted else None
    )
    return sequences, codebooks, overlap, step_weights


def time_case(op, batch: int, weighted: bool, constrained: bool, iters: int) -> float:
    sequences, codebooks, overlap, step_weights = build_inputs(batch, weighted, constrained)
    for _ in range(3):
        op(sequences, codebooks, 4, 16, overlap, step_weights)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        op(sequences, codebooks, 4, 16, overlap, step_weights)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    op = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()
    print(f"device: {torch.cuda.get_device_name(0)}")
    for batch in (128, 32):
        for weighted in (True, False):
            for constrained in (True, False):
                ms = time_case(op, batch, weighted, constrained, args.iters)
                print(
                    f"3x{batch:<4} weighted={int(weighted)} constrained={int(constrained)}"
                    f"  {ms:8.3f} ms/call"
                )


if __name__ == "__main__":
    main()
