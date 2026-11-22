# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The `gptqmodel` command line.

Subcommands live in their own modules; this file only parses and dispatches, so
adding `inspect` or `quantize` later does not touch the existing ones.
"""

import argparse
import json
import sys
from typing import List, Optional, Sequence


__all__ = ["build_parser", "main"]

_DESCRIPTION = "GPT-QModel command line: environment checks and measurements."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gptqmodel", description=_DESCRIPTION)
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser(
        "doctor",
        help="check torch, GPUs, dependencies and optional backends",
        description="Print an environment report suitable for a bug report.",
    )
    doctor.add_argument(
        "--model",
        default=None,
        help="also inspect the config of a local checkpoint (no weights are loaded)",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="machine-readable report on stdout (the startup banner may precede it; "
        "use --json-out for a clean file)",
    )
    doctor.add_argument(
        "--json-out",
        default=None,
        metavar="FILE",
        help="write the JSON report to FILE and keep the table on stdout",
    )
    doctor.add_argument(
        "--fail-on",
        choices=("fail", "warn"),
        default="fail",
        help="exit non-zero on FAIL (default) or also on WARN",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv: List[str] = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        from .doctor import collect, exit_code, render, to_json  # noqa: PLC0415 - lazy

        checks = collect(args.model)
        if args.json_out:
            from pathlib import Path

            Path(args.json_out).write_text(json.dumps(to_json(checks), indent=2) + "\n")
            print(render(checks))
        elif args.json:
            print(json.dumps(to_json(checks), indent=2))
        else:
            print(render(checks))
        return exit_code(checks, args.fail_on)

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
