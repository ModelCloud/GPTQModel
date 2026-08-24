# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and bit-exactness gate for the QVQ Viterbi segmented CUDA kernels.

Benchmarks ``viterbi_v2_segment_grid_trusted`` (the op used by YAQA quantization)
and optionally ``viterbi_v2_segment_banked`` across production-like batch sizes,
rates, and bank configurations. With ``--check``, every configuration is also
validated bit-exactly against the eager PyTorch recurrence oracle.

Example:
    python scripts/benchmark_qvq_v2_segment_grid.py --batches 8 32 64 128
    python scripts/benchmark_qvq_v2_segment_grid.py --check
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def _build_case(
    *,
    seed: int,
    batch: int,
    bits: float,
    transition_bits: int,
    bank_count: int,
    constrained: bool,
    weighted: bool,
    codebook_dtype: torch.dtype,
):
    from gptqmodel.quantization.qvq_codecs import pgc16_codebook_v2_bank

    generator = torch.Generator(device="cuda").manual_seed(seed)
    sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
    codebooks = torch.stack(
        tuple(pgc16_codebook_v2_bank(bank, bits=bits, dtype=torch.float32) for bank in range(bank_count))
    ).to(device="cuda", dtype=codebook_dtype)
    overlap = None
    if constrained:
        overlap = torch.randint(
            0,
            1 << (16 - transition_bits),
            (batch,),
            generator=generator,
            device="cuda",
            dtype=torch.int64,
        )
    step_weights = None
    if weighted:
        step_weights = (0.1 + torch.rand((batch, 128), generator=generator, device="cuda")).contiguous()
    return sequences, codebooks, overlap, step_weights


def _time_op(fn, *, warmup: int, iters: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(stops[i]) * 1000.0 for i in range(iters)]  # microseconds


def _check_exactness(args, rate: float, transition_bits: int, bank_count: int, segment_steps: int) -> bool:
    """Compare grid-trusted output against the eager recurrence oracle (torch.equal)."""

    from unittest.mock import patch

    from gptqmodel.quantization.qvq import _batched_v2_banked_viterbi_quantize
    from gptqmodel.utils.qvq_cuda import (
        _qvq_cuda_viterbi_v2_segment_grid_trusted_op,
        qvq_cuda_viterbi_v2_segment_banked,
    )

    batch = 3
    sequences, codebooks, overlap, step_weights = _build_case(
        seed=99_000 + int(rate * 2) * 10 + bank_count,
        batch=batch,
        bits=rate,
        transition_bits=transition_bits,
        bank_count=bank_count,
        constrained=args.constrained,
        weighted=args.weighted,
        codebook_dtype=torch.float16 if args.half else torch.float32,
    )

    # Force the public quantizer onto its eager recurrence while tensors stay on CUDA.
    with patch("torch.cuda.get_device_capability", lambda *_: (7, 5)):
        reference = _batched_v2_banked_viterbi_quantize(
            sequences,
            codebooks,
            bits=rate,
            segment_steps=segment_steps,
            overlap=overlap,
            step_weights=step_weights,
        )

    expected = qvq_cuda_viterbi_v2_segment_banked(
        sequences, codebooks, rate, segment_steps, overlap, step_weights
    )
    actual = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
        sequences, codebooks, transition_bits, segment_steps, overlap, step_weights
    )

    ok = True
    # actual = (states, squared_error, segment_bank_ids); reference is a result object.
    if not (
        torch.equal(expected[0], actual[0])
        and torch.equal(expected[1], actual[1])
        and torch.equal(expected[2], actual[2])
    ):
        print(f"  MISMATCH[grid vs banked] rate={rate} banks={bank_count}")
        ok = False
    # The legacy trusted path must also match the eager oracle exactly.
    if not (
        torch.equal(reference.states, expected[0])
        and torch.equal(reference.segment_bank_ids, expected[2])
        and torch.equal(reference.squared_error, expected[1])
    ):
        print(f"  BANKED-vs-ORACLE MISMATCH rate={rate} banks={bank_count}")
        ok = False
    del reference, expected, actual, sequences, codebooks, overlap, step_weights
    torch.cuda.empty_cache()
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+", default=(1.0, 2.0, 2.5, 3.0), help="QVQ rates (W)")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=("2x16", "4x32"),
        help="bank_count x segment_steps configs, e.g. 2x16",
    )
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 4, 8, 16, 32, 64, 128, 256))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--half", action="store_true", help="float16 codebooks (default)")
    parser.set_defaults(half=True)
    parser.add_argument("--float32-codebook", action="store_true")
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="also time viterbi_v2_segment_banked")
    parser.add_argument("--check", action="store_true", help="bit-exactness gate vs eager oracle")
    args = parser.parse_args()

    from gptqmodel.quantization.qvq_rates import qvq_transition_bits
    from gptqmodel.utils.qvq_cuda import (
        _qvq_cuda_viterbi_v2_segment_grid_trusted_op,
        prewarm_qvq_cuda,
        qvq_cuda_viterbi_v2_segment_banked,
    )

    assert prewarm_qvq_cuda()
    grid_op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    codebook_dtype = torch.float32 if args.float32_codebook else torch.float16

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} | sm_{props.major}{props.minor} | "
          f"{props.multi_processor_count} SMs | torch {torch.__version__}")
    print(f"warmup={args.warmup} iters={args.iters} half_codebook={not args.float32_codebook} "
          f"constrained={args.constrained} weighted={args.weighted}")

    if args.check:
        print("\n== Bit-exactness gate ==")
        all_ok = True
        for rate in args.rates:
            transition_bits = qvq_transition_bits(rate, vector_size=2)
            for config in args.configs:
                bank_count, segment_steps = (int(v) for v in config.split("x"))
                ok = _check_exactness(args, rate, transition_bits, bank_count, segment_steps)
                status = "EXACT" if ok else "FAIL"
                all_ok &= ok
                print(f"  rate=W{rate} shift={transition_bits} b{bank_count}-p{1 << (16 - transition_bits)} "
                      f"seg{segment_steps}: {status}")
        if not all_ok:
            sys.exit(1)

    print("\n== Timing ==")
    header = f"{'config':<22} {'batch':>6} {'median_us':>11} {'p25':>9} {'p75':>9} {'seq/s':>10}"
    print(header)
    print("-" * len(header))
    for rate in args.rates:
        transition_bits = qvq_transition_bits(rate, vector_size=2)
        for config in args.configs:
            bank_count, segment_steps = (int(v) for v in config.split("x"))
            label = f"W{rate}/s{transition_bits}/b{bank_count}"
            for batch in args.batches:
                sequences, codebooks, overlap, step_weights = _build_case(
                    seed=args.seed + int(rate * 2) * 1000 + batch,
                    batch=batch,
                    bits=rate,
                    transition_bits=transition_bits,
                    bank_count=bank_count,
                    constrained=args.constrained,
                    weighted=args.weighted,
                    codebook_dtype=codebook_dtype,
                )

                def run_grid():
                    grid_op(sequences, codebooks, transition_bits, segment_steps, overlap, step_weights)

                samples = _time_op(run_grid, warmup=args.warmup, iters=args.iters)
                median = statistics.median(samples)
                p25 = sorted(samples)[len(samples) // 4]
                p75 = sorted(samples)[(len(samples) * 3) // 4]
                throughput = batch / (median / 1e6)
                print(f"{label:<22} {batch:>6} {median:>11.1f} {p25:>9.1f} {p75:>9.1f} {throughput:>10.0f}")

                if args.legacy:

                    def run_legacy():
                        qvq_cuda_viterbi_v2_segment_banked(
                            sequences, codebooks, rate, segment_steps, overlap, step_weights
                        )

                    samples = _time_op(run_legacy, warmup=args.warmup, iters=args.iters)
                    median = statistics.median(samples)
                    print(f"{label + ' legacy':<22} {batch:>6} {median:>11.1f} "
                          f"{sorted(samples)[len(samples) // 4]:>9.1f} "
                          f"{sorted(samples)[(len(samples) * 3) // 4]:>9.1f} {batch / (median / 1e6):>10.0f}")

                torch.cuda.empty_cache()  # release this case's buffers before the next allocation wave


if __name__ == "__main__":
    main()
