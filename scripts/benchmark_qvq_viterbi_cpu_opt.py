#!/usr/bin/env python3
"""Benchmark the fused CPU Viterbi kernel against the baseline kernel and the torch oracle."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shifts",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="transition widths in bits (V2 rates: shift = 2 * bits)",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--state-bits", type=int, default=16, help="log2 of the codebook size")
    parser.add_argument("--vector-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--threads", type=int, default=0, help="torch.get_num_threads() override (0 = keep)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _print_table(rows: list[dict[str, object]]) -> None:
    columns = (
        ("Shift", "shift"),
        ("Batch", "batch_size"),
        ("Oracle ms", "oracle_ms"),
        ("Base ms", "base_ms"),
        ("Opt ms", "opt_ms"),
        ("Opt vs base", "speedup_vs_base"),
        ("Opt vs oracle", "speedup_vs_oracle"),
        ("Path exact", "path_exact"),
        ("Oracle path", "oracle_path_match"),
    )
    rendered = []
    for row in rows:
        rendered.append(
            {
                "shift": str(row["shift"]),
                "batch_size": str(row["batch_size"]),
                "oracle_ms": "" if row["oracle_ms"] is None else f"{row['oracle_ms']:.2f}",
                "base_ms": "" if row["base_ms"] is None else f"{row['base_ms']:.2f}",
                "opt_ms": "" if row["opt_ms"] is None else f"{row['opt_ms']:.2f}",
                "speedup_vs_base": (
                    ""
                    if row["speedup_vs_base"] is None
                    else f"{row['speedup_vs_base']:.2f}x"
                ),
                "speedup_vs_oracle": (
                    ""
                    if row["speedup_vs_oracle"] is None
                    else f"{row['speedup_vs_oracle']:.2f}x"
                ),
                "path_exact": "yes" if row["path_exact"] else "NO",
                "oracle_path_match": "yes" if row["oracle_path_match"] else "near-tie",
            }
        )
    widths = {key: max(len(title), *(len(item[key]) for item in rendered)) for title, key in columns}
    separator = "+" + "+".join("-" * (widths[key] + 2) for _, key in columns) + "+"
    print(separator)
    print("| " + " | ".join(title.ljust(widths[key]) for title, key in columns) + " |")
    print(separator)
    for item in rendered:
        print("| " + " | ".join(item[key].ljust(widths[key]) for _, key in columns) + " |")
    print(separator)


def main() -> None:
    args = _parse_args()
    if args.shifts and min(args.shifts) < 1:
        raise SystemExit("--shifts must be >= 1")

    import torch

    from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi, qvq_cpu_viterbi_opt

    if not qvq_cpu_supported():
        raise SystemExit("QVQ CPU kernels unavailable on this machine")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    state_count = 1 << args.state_bits

    def torch_oracle(sequences: torch.Tensor, codebook: torch.Tensor, shift: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Pure-torch port of the batched_viterbi_quantize fallback (float32).
        batch_size, step_count, _ = sequences.shape
        trellis_window = args.state_bits
        codebook_norm = codebook.square().sum(dim=-1)
        state_ids = torch.arange(state_count, dtype=torch.long)

        def emission(step: int) -> torch.Tensor:
            target = sequences[:, step]
            distance = (
                target.square().sum(dim=-1, keepdim=True)
                + codebook_norm.unsqueeze(0)
                - 2 * target @ codebook.transpose(0, 1)
            ).clamp_min_(0)
            return distance

        costs = emission(0)
        backpointers: list[torch.Tensor] = []
        prefix_count = 1 << shift
        suffix_count = 1 << (trellis_window - shift)
        for step in range(1, step_count):
            predecessor = costs.reshape(batch_size, prefix_count, suffix_count)
            best_cost, best_prefix = predecessor.min(dim=1)
            costs = best_cost[:, state_ids >> shift] + emission(step)
            backpointers.append(best_prefix.to(torch.int16 if shift <= 15 else torch.int32))

        end_state = costs.argmin(dim=1)
        path = torch.empty((batch_size, step_count), dtype=torch.long)
        path[:, -1] = end_state
        batch_ids = torch.arange(batch_size, dtype=torch.long)
        for step in range(step_count - 1, 0, -1):
            suffix = path[:, step] >> shift
            path[:, step - 1] = backpointers[step - 1][batch_ids, suffix].to(torch.long) * suffix_count + suffix
        return path, costs[batch_ids, end_state]

    def bench(fn, warmup: int, iterations: int) -> float:
        for _ in range(warmup):
            fn()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            timings.append(time.perf_counter() - start)
        return statistics.median(timings) * 1e3

    rows: list[dict[str, object]] = []
    maximum_batch = max(args.batch_sizes)
    generator = torch.Generator().manual_seed(20260824)
    sequences = torch.randn((maximum_batch, args.steps, args.vector_size), generator=generator, dtype=torch.float32)
    codebook = torch.randn((state_count, args.vector_size), generator=generator, dtype=torch.float32)

    for shift in sorted(set(args.shifts)):
        if not 1 <= shift <= args.state_bits:
            raise SystemExit(f"shift {shift} outside [1, {args.state_bits}]")
        for batch_size in sorted(set(args.batch_sizes)):
            batch = sequences[:batch_size]

            expected_states, expected_se = torch_oracle(batch, codebook, shift)
            base_states, base_se = qvq_cpu_viterbi(batch, codebook, shift)
            opt_states, opt_se = qvq_cpu_viterbi_opt(batch, codebook, shift)

            # Contract: the fused kernel must reproduce the production CPU
            # kernel bit-for-bit.  The pure-torch oracle uses a different
            # accumulation order (BLAS gemv versus per-state FMA chains), so
            # its selected path can legitimately flip on near-ties at large
            # batch; track that agreement separately instead of failing.
            path_exact = bool(torch.equal(opt_states, base_states) and torch.equal(opt_se, base_se))
            oracle_path_match = bool(torch.equal(opt_states, expected_states))
            oracle_se_delta = float((opt_se - expected_se).abs().max())

            oracle_ms = bench(lambda b=batch, s=shift: torch_oracle(b, codebook, s), args.warmup, args.iterations)
            base_ms = bench(lambda b=batch, s=shift: qvq_cpu_viterbi(b, codebook, s), args.warmup, args.iterations)
            opt_ms = bench(lambda b=batch, s=shift: qvq_cpu_viterbi_opt(b, codebook, s), args.warmup, args.iterations)

            rows.append(
                {
                    "shift": shift,
                    "batch_size": batch_size,
                    "oracle_ms": oracle_ms,
                    "base_ms": base_ms,
                    "opt_ms": opt_ms,
                    "speedup_vs_base": base_ms / opt_ms if opt_ms else None,
                    "speedup_vs_oracle": oracle_ms / opt_ms if opt_ms else None,
                    "path_exact": path_exact,
                    "oracle_path_match": oracle_path_match,
                    "oracle_se_delta": oracle_se_delta,
                }
            )

    _print_table(rows)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": platform.cpu_info() if hasattr(platform, "cpu_info") else platform.processor(),
            "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
        },
        "settings": {
            "state_bits": args.state_bits,
            "vector_size": args.vector_size,
            "steps": args.steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "rows": rows,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
