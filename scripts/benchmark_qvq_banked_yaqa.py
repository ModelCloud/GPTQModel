# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A/B profile V2, V2B2-P32, and V2B4-P64 under identical YAQA geometry."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from gptqmodel.quantization.qvq import (
    _canonical_qvq_codebook,
    _canonical_qvq_v2b2_pair_stacks,
    _canonical_qvq_v2b4_bank_stack,
    _canonical_qvq_v2b4_banks,
    yaqa_inner,
    yaqa_inner_v2b2_p32,
    yaqa_inner_v2b4_p64,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bits", type=float, required=True)
    parser.add_argument("--in-features", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260861, 20260862, 20260863))
    parser.add_argument("--trellis-batch-size", type=int, default=3)
    return parser


def _loss(
    source: torch.Tensor,
    candidate: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
) -> float:
    error = candidate.to(torch.float32) - source.to(torch.float32)
    return float(torch.einsum("ij,ik,kl,lj->", error, input_hessian, error, output_hessian).item())


def _artifact(result):
    if len(result) == 2:
        weight, states = result
        return weight, (states,)
    if len(result) == 3:
        weight, states, selectors = result
        return weight, (states, selectors)
    weight, states, selectors, family = result
    return weight, (states, selectors, family)


def main() -> None:
    args = _parser().parse_args()
    if args.in_features % 16 or args.out_features % 16:
        raise ValueError("YAQA benchmark dimensions must be divisible by 16.")
    if args.repetitions < 1 or len(args.seeds) < 1:
        raise ValueError("YAQA benchmark requires positive repetitions and at least one seed.")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    canonical = _canonical_qvq_codebook(
        device=device,
        vector_size=2,
        bits=args.bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=dtype,
    )
    banks = _canonical_qvq_v2b4_banks(
        device=device,
        bits=args.bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=dtype,
    )
    bank_stack = _canonical_qvq_v2b4_bank_stack(
        device=device,
        bits=args.bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=dtype,
    )
    pair_stacks = _canonical_qvq_v2b2_pair_stacks(
        device=device,
        bits=args.bits,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=dtype,
    )

    timings = {arm: [] for arm in ("V2", "B2 fixed", "B2 reselect", "B4")}
    losses = {arm: [] for arm in timings}
    exact = {arm: True for arm in timings}
    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        source = (torch.randn((args.in_features, args.out_features), generator=generator) * 0.05).to(device)
        input_samples = torch.randn((53, args.in_features), generator=generator).to(device)
        output_samples = torch.randn((47, args.out_features), generator=generator).to(device)
        input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
        output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
        input_hessian.diagonal().add_(0.1)
        output_hessian.diagonal().add_(0.1)
        common = {
            "bits": args.bits,
            "trellis_batch_size": args.trellis_batch_size,
        }
        runners = {
            "V2": lambda: yaqa_inner(source, input_hessian, output_hessian, canonical, **common),
            "B2 fixed": lambda: yaqa_inner_v2b2_p32(
                source,
                input_hessian,
                output_hessian,
                banks,
                bank_codebook_pair_stacks=pair_stacks,
                family_mode="fixed",
                fixed_family_id=1,
                **common,
            ),
            "B2 reselect": lambda: yaqa_inner_v2b2_p32(
                source,
                input_hessian,
                output_hessian,
                banks,
                bank_codebook_pair_stacks=pair_stacks,
                family_mode="reselect",
                **common,
            ),
            "B4": lambda: yaqa_inner_v2b4_p64(
                source,
                input_hessian,
                output_hessian,
                banks,
                bank_codebook_stack=bank_stack,
                **common,
            ),
        }
        for arm, runner in runners.items():
            reference_weight, reference_payload = _artifact(runner())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for _ in range(args.repetitions):
                started = time.perf_counter()
                candidate_weight, candidate_payload = _artifact(runner())
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings[arm].append((time.perf_counter() - started) * 1000.0)
                exact[arm] &= torch.equal(candidate_weight, reference_weight) and all(
                    torch.equal(candidate, reference)
                    for candidate, reference in zip(candidate_payload, reference_payload, strict=True)
                )
            losses[arm].append(_loss(source, reference_weight, input_hessian, output_hessian))

    baseline_loss = statistics.mean(losses["V2"])
    print("+-------------+-----------+-----------+-----------+-----------+--------+")
    print("| Arm         | Median ms | Mean loss | vs V2 loss| Runs/seeds| Exact  |")
    print("+-------------+-----------+-----------+-----------+-----------+--------+")
    for arm in timings:
        mean_loss = statistics.mean(losses[arm])
        loss_delta = (mean_loss / baseline_loss - 1.0) * 100.0
        print(
            f"| {arm:<11} | {statistics.median(timings[arm]):>9.3f} | {mean_loss:>9.4e} | "
            f"{loss_delta:>+8.2f}% | {len(timings[arm]):>3}/{len(args.seeds):<3} | "
            f"{'yes' if exact[arm] else 'NO':<6} |"
        )
    print("+-------------+-----------+-----------+-----------+-----------+--------+")
    if not all(exact.values()):
        raise RuntimeError("YAQA benchmark observed nondeterministic states, selectors, family IDs, or weights.")


if __name__ == "__main__":
    main()
