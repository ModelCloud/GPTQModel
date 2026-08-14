#!/usr/bin/env python3
"""Run pytest with native numerical thread pools capped."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

MAX_LIMIT = 16
DEFAULT_LIMIT = MAX_LIMIT
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def effective_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("thread limit must be at least 1")
    return min(limit, MAX_LIMIT)


def limited_env(limit: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(effective_limit(limit))
    for name in THREAD_ENV_VARS:
        env[name] = value
    return env


def run_in_target(python: str, pytest_args: list[str], limit: int) -> int:
    limit = effective_limit(limit)
    bootstrap = """
import sys
try:
    import pytest
    from threadpoolctl import threadpool_limits
except ImportError as exc:
    print(f"CPU test limiter dependency missing: {exc}", file=sys.stderr)
    raise SystemExit(2)
with threadpool_limits(limits=int(sys.argv[1])):
    raise SystemExit(pytest.main(sys.argv[2:]))
"""
    command = [python, "-c", bootstrap, str(limit), *pytest_args]
    return subprocess.call(command, env=limited_env(limit))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="project Python interpreter")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"native thread limit (values above {MAX_LIMIT} are capped at {MAX_LIMIT})",
    )
    parser.add_argument("--print-env", action="store_true")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    limit = effective_limit(args.limit)
    if args.print_env:
        print(" ".join(f"{name}={limit}" for name in THREAD_ENV_VARS))
        return 0
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return run_in_target(args.python, pytest_args, limit)


if __name__ == "__main__":
    raise SystemExit(main())
