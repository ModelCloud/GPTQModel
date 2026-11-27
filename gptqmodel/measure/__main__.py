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

import torch

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


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _window(text: str) -> int:
    value = _positive_int(text)
    if value <= 1:
        raise argparse.ArgumentTypeError(f"a window needs at least two tokens, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more, got {value}")
    return value


def _topk(text: str) -> str:
    try:
        widths = [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a comma-separated list of integers") from None
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError(f"top-k widths must be positive, got {text!r}")
    return text


def _grade_expression(text: str) -> str:
    """Reject a malformed bound before any model is loaded."""

    try:
        GradedThresholdConfig(grades={MetricGrades.PASS: text})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text


def _check_device(name: Optional[str], flag: str) -> Optional[str]:
    """Return an error message for a device this machine cannot run on."""

    if name is None:
        return None
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError):
        return f"{flag} {name!r} is not a torch device"
    # Read the index from the string: `torch.device("cuda:999").index` wraps to
    # -25 (the index is parsed into a small signed integer), so comparing
    # `device.index` against the device count would accept GPU 999 silently.
    raw_index = name.rsplit(":", 1)[1] if ":" in name else None
    index = int(raw_index) if raw_index is not None and raw_index.isdigit() else None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return f"{flag} {name!r} needs CUDA, which is not available here"
        if index is not None and index >= torch.cuda.device_count():
            return (
                f"{flag} {name!r} asks for GPU {index}, but this machine has "
                f"{torch.cuda.device_count()}"
            )
    elif device.type == "mps" and not torch.backends.mps.is_available():
        return f"{flag} {name!r} needs MPS, which is not available here"
    return None


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
    parser.add_argument("--window", type=_window, default=512, help="PPL window")
    parser.add_argument("--stride", type=_non_negative_int, default=256, help="PPL window stride")
    parser.add_argument("--positions", type=_positive_int, default=64, help="sampled positions per batch")
    parser.add_argument("--batch", type=_positive_int, default=1, help="rows per forward")
    parser.add_argument("--max-batches", type=_positive_int, default=4, help="rows to read")

    parser.add_argument("--ppl", action="store_true", help="measure PPL (default: all metrics)")
    parser.add_argument("--kld", action="store_true", help="measure KLD")
    parser.add_argument("--rmse", action="store_true", help="measure RMSE")
    parser.add_argument("--topk", type=_topk, default=None, metavar="K[,K]", help="top-k widths, e.g. 1,5")

    parser.add_argument("--ppl-pass", type=_grade_expression, default=None, metavar="EXPR", help='e.g. "<=25"')
    parser.add_argument("--kld-pass", type=_grade_expression, default=None, metavar="EXPR", help='e.g. "<=0.05"')
    parser.add_argument("--rmse-pass", type=_grade_expression, default=None, metavar="EXPR", help='e.g. "<=0.5"')
    parser.add_argument("--topk-pass", type=_grade_expression, default=None, metavar="EXPR", help='e.g. ">=0.95"')
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
    all_metrics = not any((args.ppl, args.kld, args.rmse, args.topk))
    # A threshold is a request for its metric: `--ppl --kld-pass "<0"` used to
    # run PPL only and report "no thresholds configured", silently dropping the
    # KLD grade.
    selected = []
    if args.ppl or all_metrics or args.ppl_pass is not None:
        selected.append(PPLConfig(window=args.window, stride=args.stride, threshold=_grade(args.ppl_pass)))
    if args.kld or all_metrics or args.kld_pass is not None:
        selected.append(
            KLDConfig(positions=args.positions, threshold=_grade(args.kld_pass))
        )
    if args.rmse or all_metrics or args.rmse_pass is not None:
        selected.append(RMSEConfig(threshold=_grade(args.rmse_pass)))
    if args.topk or all_metrics or args.topk_pass is not None:
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
    parser = build_parser()
    args = parser.parse_args(argv)

    problems = [
        problem
        for problem in (
            _check_device(args.device, "--device"),
            _check_device(args.reference_device, "--reference-device"),
        )
        if problem
    ]
    if args.json and args.json_out:
        problems.append("--json and --json-out both ask for JSON; pick one")
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

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

    try:
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report.to_json(), indent=2) + "\n")
        # `render()` is the LogBar column printer, so values line up in a terminal.
        print(report.render())
    except OSError as exc:
        print(f"cannot write the report: {exc}", file=sys.stderr)
        return 2

    if report.overall is MetricGrades.FAIL:
        return 1
    if args.fail_on == "warn" and report.overall is MetricGrades.WARN:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
