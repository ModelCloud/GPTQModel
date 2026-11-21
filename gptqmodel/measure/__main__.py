# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``python -m gptqmodel.measure``: measure a checkpoint against its reference.

Exit codes are meant for CI: 0 when nothing failed (or nothing was graded), 1
when a metric failed, and 2 when the measurement itself could not run.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import (
    GradedThresholdConfig,
    KLDConfig,
    MeasureConfig,
    MetricGrades,
    PPLConfig,
    RMSEConfig,
    TOPKConfig,
)


__all__ = ["build_parser", "main"]


def _grade(value: Optional[str]) -> Optional[GradedThresholdConfig]:
    """``"<=0.05"`` -> PASS at or below 0.05, anything else fails."""

    if value is None:
        return None
    return GradedThresholdConfig(grades={MetricGrades.PASS: value})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gptqmodel.measure",
        description="Measure how far a quantized checkpoint drifted from its reference.",
    )
    parser.add_argument("--model", required=True, help="quantized checkpoint to measure")
    parser.add_argument("--reference", required=True, help="fp16/bf16 checkpoint to compare against")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        metavar="FILE",
        help="text file to measure on; repeatable (default: stdin)",
    )
    parser.add_argument("--device", default=None, help="device for the quantized model")
    parser.add_argument(
        "--reference-device",
        default=None,
        help="device for the reference model (default: same as --device); "
        "'cpu' keeps the fp16 copy out of VRAM at the cost of speed",
    )
    parser.add_argument("--window", type=int, default=512, help="PPL window")
    parser.add_argument("--stride", type=int, default=256, help="PPL window stride")
    parser.add_argument("--positions", type=int, default=64, help="sampled positions per batch")
    parser.add_argument("--batch", type=int, default=1, help="rows per forward")
    parser.add_argument("--max-batches", type=int, default=4, help="rows to read")

    parser.add_argument("--ppl", action="store_true", help="measure PPL (default: all metrics)")
    parser.add_argument("--kld", action="store_true", help="measure KLD")
    parser.add_argument("--rmse", action="store_true", help="measure RMSE")
    parser.add_argument("--topk", default=None, metavar="K[,K]", help="top-k widths, e.g. 1,5")

    parser.add_argument("--ppl-pass", default=None, metavar="EXPR", help='e.g. "<=25"')
    parser.add_argument("--kld-pass", default=None, metavar="EXPR", help='e.g. "<=0.05"')
    parser.add_argument("--rmse-pass", default=None, metavar="EXPR", help='e.g. "<=0.5"')
    parser.add_argument("--topk-pass", default=None, metavar="EXPR", help='e.g. ">=0.95"')
    parser.add_argument(
        "--fail-on",
        choices=("warn", "fail"),
        default="fail",
        help="exit non-zero on FAIL (default) or also on WARN",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the JSON report instead of a table (library logs may precede it; "
        "use --json-out for a clean file)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="FILE",
        help="write the JSON report to FILE and keep the table on stdout",
    )
    return parser


def _metrics(args: argparse.Namespace) -> tuple:
    widths = tuple(int(width) for width in args.topk.split(",")) if args.topk else (1,)
    selected = []
    if args.ppl or not any((args.ppl, args.kld, args.rmse, args.topk)):
        selected.append(PPLConfig(window=args.window, stride=args.stride, threshold=_grade(args.ppl_pass)))
    if args.kld or not selected:
        selected.append(
            KLDConfig(positions=args.positions, threshold=_grade(args.kld_pass))
        )
    if args.rmse or not selected:
        selected.append(RMSEConfig(threshold=_grade(args.rmse_pass)))
    if args.topk or not selected:
        selected.append(TOPKConfig(k=widths, threshold=_grade(args.topk_pass)))
    return tuple(selected)


def _read_texts(files: Sequence[str]) -> List[str]:
    if not files:
        return [sys.stdin.read()]
    texts = []
    for name in files:
        texts.append(Path(name).read_text(errors="replace"))
    return texts


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from transformers import AutoModelForCausalLM

        from .. import GPTQModel

        candidate = GPTQModel.load(args.model, device=args.device)
        reference = AutoModelForCausalLM.from_pretrained(args.reference, torch_dtype="bfloat16")
        reference_device = args.reference_device or args.device
        if reference_device:
            reference = reference.to(reference_device)
        report = candidate.measure(
            MeasureConfig(
                metrics=_metrics(args),
                dataset=_read_texts(args.text),
                reference=reference,
                batch=args.batch,
                max_batches=args.max_batches,
                # No device override: the runner reads each model's own device,
                # which is what lets the reference stay on the CPU.
            )
        )
    except Exception as exc:  # noqa: BLE001 - the CLI turns every failure into exit 2
        print(f"measurement failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_json(), indent=2) + "\n")
        print(report.markdown())
    elif args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(report.markdown())

    if report.overall is MetricGrades.FAIL:
        return 1
    if args.fail_on == "warn" and report.overall is MetricGrades.WARN:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
