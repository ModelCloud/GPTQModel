# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark the production B2-P32 family-grid Viterbi operator."""

import argparse

import torch

from gptqmodel.quantization.qvq import qvq_transition_bits
from gptqmodel.quantization.qvq_codecs.pgc16 import pgc16_codebook_v2_bank
from gptqmodel.utils.qvq_cuda import (
    _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op,
    _qvq_cuda_viterbi_v2_segment_family_midpoint_trusted_op,
)


def build_inputs(bits: float, families: int, batch: int, weighted: bool, constrained: bool, seed: int = 20260823):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    sequences = torch.randn((families, batch, 128, 2), generator=generator, device="cuda")
    codebooks = torch.stack(
        tuple(
            torch.stack(
                (
                    pgc16_codebook_v2_bank(0, bits=bits, dtype=torch.float32),
                    pgc16_codebook_v2_bank(family, bits=bits, dtype=torch.float32),
                )
            )
            for family in tuple(index % 3 + 1 for index in range(families))
        )
    ).to(device="cuda", dtype=torch.float16)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    overlap = (
        torch.randint(
            0,
            1 << (16 - transition_bits),
            (families, batch),
            generator=generator,
            device="cuda",
            dtype=torch.int64,
        )
        if constrained
        else None
    )
    step_weights = (
        (0.1 + torch.rand((families, batch, 128), generator=generator, device="cuda")) if weighted else None
    )
    return sequences, codebooks, overlap, step_weights


def time_case(op, bits: float, families: int, batch: int, weighted: bool, constrained: bool, iters: int) -> float:
    sequences, codebooks, overlap, step_weights = build_inputs(bits, families, batch, weighted, constrained)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    for _ in range(3):
        op(sequences, codebooks, transition_bits, 16, overlap, step_weights)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op(sequences, codebooks, transition_bits, 16, overlap, step_weights)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def time_provisional_case(
    full_op,
    midpoint_op,
    bits: float,
    families: int,
    batch: int,
    weighted: bool,
    iters: int,
) -> tuple[float, float]:
    sequences, codebooks, _, step_weights = build_inputs(bits, families, batch, weighted, False)
    sequences = torch.roll(sequences, shifts=64, dims=2).contiguous()
    if step_weights is not None:
        step_weights = torch.roll(step_weights, shifts=64, dims=2).contiguous()
    transition_bits = qvq_transition_bits(bits, vector_size=2)

    def measure(call) -> float:
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters

    full_ms = measure(lambda: full_op(sequences, codebooks, transition_bits, 16, None, step_weights))
    midpoint_ms = measure(lambda: midpoint_op(sequences, codebooks, transition_bits, 16, step_weights))
    return full_ms, midpoint_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--bits", type=float, default=3.0)
    parser.add_argument("--families", type=int, default=2)
    parser.add_argument("--batches", default="1,8,32,64,128")
    parser.add_argument("--midpoint-ab", action="store_true")
    args = parser.parse_args()
    batches = tuple(int(value) for value in args.batches.split(","))
    op = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()
    midpoint_op = _qvq_cuda_viterbi_v2_segment_family_midpoint_trusted_op()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"bits={args.bits:g} families={args.families}")
    if args.midpoint_ab:
        for batch in batches:
            for weighted in (True, False):
                full_ms, midpoint_ms = time_provisional_case(
                    op,
                    midpoint_op,
                    args.bits,
                    args.families,
                    batch,
                    weighted,
                    args.iters,
                )
                print(
                    f"{args.families}x{batch:<4} weighted={int(weighted)}  "
                    f"full={full_ms:8.3f} ms  midpoint={midpoint_ms:8.3f} ms  "
                    f"speedup={full_ms / midpoint_ms:6.3f}x"
                )
        return
    for batch in batches:
        for weighted in (True, False):
            for constrained in (True, False):
                ms = time_case(
                    op,
                    args.bits,
                    args.families,
                    batch,
                    weighted,
                    constrained,
                    args.iters,
                )
                print(
                    f"{args.families}x{batch:<4} weighted={int(weighted)} constrained={int(constrained)}  "
                    f"{ms:8.3f} ms/call"
                )


if __name__ == "__main__":
    main()
