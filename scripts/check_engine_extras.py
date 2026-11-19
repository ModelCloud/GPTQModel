#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Resolve every advertised engine extra on every supported interpreter.

An engine extra shares one environment with the core package, so it is only
installable when the engine release's published Torch/Transformers pins
intersect ``requirements.txt`` on that interpreter (see docs/engine-extras.md).
When they do not, the resolver fails before installing anything and the extra
must not advertise that engine on that lane.

This script runs the same compilation the resolver would, reports the selected
engine, Torch, and Transformers versions per lane, and fails when an advertised
extra cannot be satisfied. It needs network access to PyPI and ``uv`` on PATH.

Usage:

    python3 scripts/check_engine_extras.py
    python3 scripts/check_engine_extras.py --python-version 3.11
    python3 scripts/check_engine_extras.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHONS = ("3.11", "3.12", "3.13", "3.14")
DEFAULT_PLATFORM = "x86_64-manylinux_2_35"
ENGINES = ("vllm", "sglang")
REPORTED_PINS = ("vllm", "sglang", "torch", "transformers")
_PIN_LINE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s;]+)")


@dataclass(frozen=True)
class LaneResult:
    extra: str
    python_version: str
    status: str  # resolved | excluded | unresolved
    pins: dict[str, str]
    detail: str = ""


def engine_extras(pyproject: Path = REPO_ROOT / "pyproject.toml") -> dict[str, list[str]]:
    """Extras that name an inference engine, in declaration order."""

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    return {
        name: list(deps)
        for name, deps in extras.items()
        if any(_requirement_name(dep) in ENGINES for dep in deps)
    }


def _requirement_name(requirement: str) -> str:
    head = requirement.split(";", 1)[0].strip()
    match = re.match(r"^(?P<name>[A-Za-z0-9._-]+)", head)
    return match.group("name").lower() if match else ""


def selected_pins(output: str) -> dict[str, str]:
    """Pinned versions of the engine/Torch/Transformers packages in a compile."""

    pins: dict[str, str] = {}
    for line in output.splitlines():
        match = _PIN_LINE.match(line.strip())
        if match and match.group("name").lower() in REPORTED_PINS:
            pins[match.group("name").lower()] = match.group("version")
    return pins


def compile_lane(
    extra: str,
    python_version: str,
    *,
    root: Path = REPO_ROOT,
    python_platform: str = DEFAULT_PLATFORM,
    timeout: int = 900,
) -> LaneResult:
    """Compile one extra for one interpreter and classify the outcome."""

    cmd = [
        "uv",
        "--no-config",
        "pip",
        "compile",
        "--python-version",
        python_version,
        "--python-platform",
        python_platform,
        "--extra",
        extra,
        "pyproject.toml",
    ]
    try:
        completed = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return LaneResult(extra, python_version, "unresolved", {}, f"timed out after {timeout}s")

    if completed.returncode != 0:
        lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
        detail = next(
            (line for line in lines if line.startswith("cause:")),
            lines[0] if lines else f"exit {completed.returncode}",
        )
        return LaneResult(extra, python_version, "unresolved", {}, detail)

    pins = selected_pins(completed.stdout)
    engine_pinned = any(name in pins for name in ENGINES)
    status = "resolved" if engine_pinned else "excluded"
    return LaneResult(extra, python_version, status, pins)


def check(
    *,
    root: Path = REPO_ROOT,
    pythons: tuple[str, ...] = DEFAULT_PYTHONS,
    extras: dict[str, list[str]] | None = None,
    python_platform: str = DEFAULT_PLATFORM,
) -> list[LaneResult]:
    extras = engine_extras(root / "pyproject.toml") if extras is None else extras
    return [
        compile_lane(name, version, root=root, python_platform=python_platform)
        for name in extras
        for version in pythons
    ]


def report(results: list[LaneResult]) -> None:
    header = f"{'extra':<10}{'python':<9}{'status':<12}{'engine':<14}{'torch':<10}{'transformers':<14}"
    print(header)
    print("-" * len(header))
    for result in results:
        engine = next(
            (f"{name} {result.pins[name]}" for name in ENGINES if name in result.pins), "-"
        )
        print(
            f"{result.extra:<10}{result.python_version:<9}{result.status:<12}{engine:<14}"
            f"{result.pins.get('torch', '-'):<10}{result.pins.get('transformers', '-'):<14}"
        )
        if result.detail:
            print(f"  {result.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--python-version",
        action="append",
        dest="python_versions",
        help=f"interpreter to compile for; repeatable (default: {', '.join(DEFAULT_PYTHONS)})",
    )
    parser.add_argument(
        "--extra",
        action="append",
        dest="extras",
        help="engine extra to check; repeatable (default: every engine extra)",
    )
    parser.add_argument("--python-platform", default=DEFAULT_PLATFORM)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="checkout holding pyproject.toml and requirements.txt (default: this repository)",
    )
    parser.add_argument("--json", action="store_true", help="emit the per-lane results as JSON")
    args = parser.parse_args(argv)

    if shutil.which("uv") is None:
        print("error: uv is not on PATH; install it from https://docs.astral.sh/uv/", file=sys.stderr)
        return 2

    declared = engine_extras(args.root / "pyproject.toml")
    if args.extras:
        missing = sorted(set(args.extras) - set(declared))
        if missing:
            print(f"error: not an engine extra: {', '.join(missing)}", file=sys.stderr)
            return 2
        declared = {name: declared[name] for name in args.extras}

    pythons = tuple(args.python_versions or DEFAULT_PYTHONS)
    results = check(
        root=args.root,
        pythons=pythons,
        extras=declared,
        python_platform=args.python_platform,
    )

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        report(results)

    failures = [result for result in results if result.status == "unresolved"]
    if failures:
        print(
            f"\n{len(failures)} advertised lane(s) do not resolve; "
            "narrow the extra marker or wait for the engine to catch up.",
            file=sys.stderr,
        )
        return 1

    advertised = sum(1 for result in results if result.status == "resolved")
    print(f"\nok: every advertised lane resolves ({advertised} engine combination(s) selected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
