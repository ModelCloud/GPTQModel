#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Time the native CPU banked QVQ Viterbi kernel and gate its outputs bit-exactly.

The kernel is a quantization kernel, so any change to it must keep the selected
states, squared errors, and segment bank ids bit-identical. Capture a baseline
artifact before touching the kernel:

    python scripts/benchmark_qvq_viterbi_banked_cpu.py --save-baseline /tmp/banked_base.pt

then, after rebuilding the modified extension, compare against it:

    python scripts/benchmark_qvq_viterbi_banked_cpu.py --compare-baseline /tmp/banked_base.pt
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import torch

from gptqmodel.quantization.qvq import pack_qvq_bank_ids, pack_trellis_states
from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi_banked


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--vector-size", type=int, default=2, choices=(2, 4))
    parser.add_argument("--state-count", type=int, default=65536)
    parser.add_argument("--bank-count", type=int, default=2)
    parser.add_argument("--transition-bits", type=int, default=5)
    parser.add_argument("--segment-steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps torch defaults")
    parser.add_argument(
        "--assert-affinity-count",
        type=int,
        default=0,
        help="fail unless this many distinct singleton thread affinities exist before timing",
    )
    parser.add_argument("--step-weights", action="store_true", help="pass per-step weights")
    parser.add_argument("--overlap", action="store_true", help="pass tail-biting overlap constraints")
    parser.add_argument("--save-baseline", type=Path)
    parser.add_argument("--compare-baseline", type=Path)
    return parser


def _case_key(args: argparse.Namespace, batch: int) -> str:
    return (
        f"b{batch}_s{args.steps}_v{args.vector_size}_n{args.state_count}"
        f"_k{args.bank_count}_t{args.transition_bits}_g{args.segment_steps}"
        f"_w{int(args.step_weights)}_o{int(args.overlap)}_seed{args.seed}"
    )


def _inputs(args: argparse.Namespace, batch: int) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator().manual_seed(args.seed + batch)
    sequences = torch.randn(batch, args.steps, args.vector_size, generator=generator, dtype=torch.float32)
    codebooks = torch.randn(
        args.bank_count,
        args.state_count,
        args.vector_size,
        generator=generator,
        dtype=torch.float32,
    )
    step_weights = None
    if args.step_weights:
        step_weights = torch.rand(batch, args.steps, generator=generator, dtype=torch.float32) + 0.5
    overlap = None
    if args.overlap:
        overlap_states = args.state_count >> args.transition_bits
        overlap = torch.randint(0, overlap_states, (batch,), generator=generator, dtype=torch.int64)
    return {
        "sequences": sequences,
        "codebooks": codebooks,
        "step_weights": step_weights,
        "overlap": overlap,
    }


def _run(args: argparse.Namespace, tensors: dict[str, torch.Tensor | None]):
    return qvq_cpu_viterbi_banked(
        tensors["sequences"],
        tensors["codebooks"],
        args.transition_bits,
        args.segment_steps,
        overlap=tensors["overlap"],
        step_weights=tensors["step_weights"],
    )


def _assert_affinity(expected_count: int) -> None:
    singleton_cpus: set[int] = set()
    for status in Path("/proc/self/task").glob("*/status"):
        for line in status.read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                allowed = line.split(":", 1)[1].strip()
                if allowed.isdigit():
                    singleton_cpus.add(int(allowed))
                break
    if len(singleton_cpus) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} distinct singleton worker affinities, found "
            f"{len(singleton_cpus)}: {sorted(singleton_cpus)}"
        )
    print(f"affinity: {expected_count} distinct singleton worker CPUs verified once before timing")


def main() -> int:
    args = _parser().parse_args()
    if not qvq_cpu_supported():
        raise SystemExit("native QVQ CPU kernel unavailable on this machine")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    baseline = None
    if args.compare_baseline is not None:
        baseline = torch.load(args.compare_baseline, map_location="cpu", weights_only=False)

    artifact: dict[str, dict[str, object]] = {}
    rows: list[tuple[str, ...]] = []
    exactness_failures: list[str] = []
    affinity_checked = False

    for batch in args.batch_sizes:
        tensors = _inputs(args, batch)
        for _ in range(args.warmup):
            _run(args, tensors)
        if args.assert_affinity_count and not affinity_checked:
            _assert_affinity(args.assert_affinity_count)
            affinity_checked = True
        timings: list[float] = []
        states = squared_error = segment_bank_ids = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            states, squared_error, segment_bank_ids = _run(args, tensors)
            timings.append((time.perf_counter() - start) * 1e3)

        key = _case_key(args, batch)
        median_ms = statistics.median(timings)
        artifact[key] = {
            "states": states,
            "squared_error": squared_error,
            "segment_bank_ids": segment_bank_ids,
            "median_ms": median_ms,
            "min_ms": min(timings),
            "max_ms": max(timings),
        }
        if args.vector_size == 2 and args.state_count == 65536 and args.steps % 32 == 0:
            try:
                artifact[key]["packed_words"] = pack_trellis_states(
                    states, bits=args.transition_bits / 2, vector_size=2
                )
            except ValueError:
                pass
            artifact[key]["packed_bank_ids"] = pack_qvq_bank_ids(segment_bank_ids.reshape(-1))

        speedup = "-"
        exact = "-"
        error_delta = "-"
        if baseline is not None:
            reference = baseline.get(key)
            if reference is None:
                exact = "no-baseline"
            else:
                mismatches = [
                    name
                    for name, value in (
                        ("states", states),
                        ("segment_bank_ids", segment_bank_ids),
                        ("packed_words", artifact[key].get("packed_words")),
                        ("packed_bank_ids", artifact[key].get("packed_bank_ids")),
                    )
                    if value is not None and name in reference and not torch.equal(value, reference[name])
                ]
                exact = "yes" if not mismatches else "NO:" + ",".join(mismatches)
                error_delta = f"{(squared_error - reference['squared_error']).abs().max().item():.3g}"
                if mismatches:
                    exactness_failures.append(f"{key} -> {','.join(mismatches)}")
                speedup = f"{reference['median_ms'] / median_ms:.2f}x"

        rows.append(
            (
                str(batch),
                str(args.steps),
                str(args.state_count),
                str(args.bank_count),
                f"{median_ms:.1f}",
                f"{min(timings):.1f}",
                f"{max(timings):.1f}",
                f"{max(timings) - min(timings):.1f}",
                f"{median_ms / batch:.3f}",
                speedup,
                exact,
                error_delta,
            )
        )

    header = (
        "batch",
        "steps",
        "states",
        "banks",
        "median_ms",
        "min_ms",
        "max_ms",
        "spread_ms",
        "ms/row",
        "speedup",
        "discrete-exact",
        "loss-max-abs",
    )
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    line = "  ".join(name.ljust(widths[i]) for i, name in enumerate(header))
    print(f"machine: {platform.processor() or platform.machine()}  torch: {torch.__version__}  "
          f"threads: {torch.get_num_threads()}")
    print(f"V={args.vector_size} transition_bits={args.transition_bits} segment_steps={args.segment_steps} "
          f"step_weights={args.step_weights} overlap={args.overlap}")
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(header))))

    if args.save_baseline is not None:
        torch.save(artifact, args.save_baseline)
        print(f"saved baseline artifact to {args.save_baseline}")

    if exactness_failures:
        print("BIT-EXACTNESS FAILURES:")
        for failure in exactness_failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
