#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B measurement with a measured noise floor.

The existing benchmark scripts time one arm once:

    for _ in range(warmup): fn()
    start = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - start) * 1e3 / iters

That number carries no uncertainty, and the usual follow-up - run arm A, then
run arm B in a second process - compares two different GPU clock states rather
than two kernels. Two runs of the *same* marlin build on one laptop differed by
6.4% on average and 23.2% worst case here, which is larger than most changes
worth making.

This module times both arms inside one process, alternating which goes first,
and reports the spread it measured next to the speedup. `--self-check` runs
both arms on identical work: anything it reports outside the noise floor is the
protocol's bias, not an effect.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


DEFAULT_ROUNDS = 10
DEFAULT_ITERS = 100
DEFAULT_WARMUP = 20


@dataclass(frozen=True)
class ArmStats:
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.samples) < 4 or any(not math.isfinite(x) or x <= 0 for x in self.samples):
            raise ValueError("at least four positive, finite samples are required to estimate noise")

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def minimum(self) -> float:
        return min(self.samples)

    @property
    def p25(self) -> float:
        return statistics.quantiles(self.samples, n=4, method="inclusive")[0]

    @property
    def p75(self) -> float:
        return statistics.quantiles(self.samples, n=4, method="inclusive")[2]

    @property
    def iqr_ratio(self) -> float:
        return (self.p75 - self.p25) / self.median


@dataclass(frozen=True)
class AbResult:
    baseline: ArmStats
    candidate: ArmStats
    rounds: int

    @property
    def speedup(self) -> float:
        return self.baseline.median / self.candidate.median

    @property
    def noise_floor(self) -> float:
        """Fraction of the median that this machine moves for identical work."""
        return max(self.baseline.iqr_ratio, self.candidate.iqr_ratio, 0.03)

    @property
    def is_effect(self) -> bool:
        # Compare multiplicative changes in either direction on the same scale.
        return max(self.speedup, 1.0 / self.speedup) - 1.0 > self.noise_floor

    def verdict(self) -> str:
        return "effect" if self.is_effect else "within noise"


def _now() -> float:
    return time.perf_counter()


def sample_ms(
        fn: Callable[[], object],
        *,
        iters: int = DEFAULT_ITERS,
        warmup: int = DEFAULT_WARMUP,
        sync: Callable[[], None] | None = None,
) -> float:
    """Best of `iters` calls, in milliseconds.

    The minimum is used per round: the fastest run is the one least polluted by
    a scheduling or clock excursion, and rounds supply the spread estimate.
    """
    if iters < 1 or warmup < 0:
        raise ValueError("iters must be positive and warmup must be nonnegative")
    synchronize = _sync if sync is None else sync
    for _ in range(warmup):
        fn()
    synchronize()
    best = float("inf")
    for _ in range(iters):
        start = _now()
        fn()
        synchronize()
        best = min(best, (_now() - start) * 1e3)
    return best


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _warm_gpu(seconds: float = 1.5, device: torch.device | None = None) -> None:
    """Hold the GPU at its boost clock before the first sample.

    An idle GPU starts at its lowest clock, so the first arm measured would be
    the one that pays for the ramp.
    """
    if not torch.cuda.is_available() or seconds <= 0:
        return
    device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    b = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    out = torch.empty_like(a)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        torch.mm(a, b, out=out)
        torch.cuda.synchronize(device)
    del a, b, out
    torch.cuda.empty_cache()


def ab(
        baseline: Callable[[], object],
        candidate: Callable[[], object],
        *,
        rounds: int = DEFAULT_ROUNDS,
        iters: int = DEFAULT_ITERS,
        warmup: int = DEFAULT_WARMUP,
        warm_gpu_seconds: float = 1.5,
        warm_gpu_device: torch.device | None = None,
        sync: Callable[[], None] | None = None,
) -> AbResult:
    """Time both arms round by round, alternating which one runs first."""
    if rounds < 4 or iters < 1 or warmup < 0 or warm_gpu_seconds < 0:
        raise ValueError("rounds must be at least four; iters must be positive; warmup and GPU warmup must be nonnegative")
    _warm_gpu(warm_gpu_seconds, warm_gpu_device)
    base_samples: list[float] = []
    cand_samples: list[float] = []
    for round_index in range(rounds):
        order: Sequence[str] = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for arm in order:
            if arm == "baseline":
                base_samples.append(sample_ms(baseline, iters=iters, warmup=warmup, sync=sync))
            else:
                cand_samples.append(sample_ms(candidate, iters=iters, warmup=warmup, sync=sync))
    return AbResult(baseline=ArmStats(tuple(base_samples)), candidate=ArmStats(tuple(cand_samples)), rounds=rounds)


def self_check(
        work: Callable[[], object],
        *,
        rounds: int = DEFAULT_ROUNDS,
        iters: int = DEFAULT_ITERS,
        warmup: int = DEFAULT_WARMUP,
        warm_gpu_seconds: float = 1.5,
        warm_gpu_device: torch.device | None = None,
        sync: Callable[[], None] | None = None,
) -> AbResult:
    """Run identical work in both arms to check for measurement bias."""
    return ab(work, work, rounds=rounds, iters=iters, warmup=warmup,
              warm_gpu_seconds=warm_gpu_seconds, warm_gpu_device=warm_gpu_device, sync=sync)


def format_result(label: str, result: AbResult) -> str:
    return (
        f"{label}: baseline {result.baseline.median * 1000:9.1f}us  "
        f"candidate {result.candidate.median * 1000:9.1f}us  "
        f"{result.speedup:6.2f}x  noise {result.noise_floor * 100:5.1f}%  "
        f"-> {result.verdict()}"
    )


def _self_check_work(size: int = 2048) -> Callable[[], object]:
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        a = torch.randn(size, size, dtype=torch.float16, device=device)
        b = torch.randn(size, size, dtype=torch.float16, device=device)
        return lambda: a @ b

    a = torch.randn(size, size)
    b = torch.randn(size, size)
    return lambda: a @ b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-check", action="store_true",
                        help="time identical work in both arms to measure this machine's bias")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--size", type=int, default=2048, help="matmul size for --self-check")
    args = parser.parse_args(argv)

    if not args.self_check:
        parser.error("only --self-check is available from the command line; import ab() for a comparison")
    if args.rounds < 4 or args.iters < 1 or args.size < 1:
        parser.error("--rounds must be at least 4; --iters and --size must be positive")

    result = self_check(_self_check_work(args.size), rounds=args.rounds, iters=args.iters)
    print(format_result("A/A control", result))
    if result.is_effect:
        print("The protocol is biased on this machine; treat any single comparison here as unreliable.")
        return 1
    print("Within the noise floor: a change must exceed this spread to be believable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
