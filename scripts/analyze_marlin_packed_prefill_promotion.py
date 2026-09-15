#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Evaluate repeated packed-prefill matrix results with a fail-closed promotion gate.

Each input must be produced by ``benchmark_marlin_packed_prefill_matrix.py``
and must include paired per-round speedups plus a physical GPU UUID. Candidate
points are kept separate by compute capability, SM count, dtype, M, K, N, and
packed-prefill config; results from unlike hardware are never pooled.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable


_CONFIG_ROUTE = re.compile(r"config_(?P<config>[1-4])")
_T95_ONE_SIDED = (
    6.314,
    2.920,
    2.353,
    2.132,
    2.015,
    1.943,
    1.895,
    1.860,
    1.833,
    1.812,
    1.796,
    1.782,
    1.771,
    1.761,
    1.753,
    1.746,
    1.740,
    1.734,
    1.729,
    1.725,
    1.721,
    1.717,
    1.714,
    1.711,
    1.708,
    1.706,
    1.703,
    1.701,
    1.699,
    1.697,
)


@dataclass(frozen=True, order=True)
class CandidateKey:
    major: int
    minor: int
    sms: int
    dtype: str
    m: int
    k: int
    n: int
    config: int


@dataclass(frozen=True)
class PromotionObservation:
    gpu_uuid: str
    paired_round_speedups: tuple[float, ...]
    raw_speedup: float
    finite: bool
    source: str


@dataclass(frozen=True)
class PromotionThresholds:
    min_gpus: int = 2
    min_rounds_per_gpu: int = 3
    confidence_speedup: float = 1.05
    raw_speedup: float = 1.07


def _one_sided_t95(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    if df <= len(_T95_ONE_SIDED):
        return _T95_ONE_SIDED[df - 1]
    if df <= 40:
        return _T95_ONE_SIDED[-1]
    if df <= 60:
        return 1.684
    if df <= 120:
        return 1.671
    return 1.658


def _geometric_mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return math.exp(statistics.mean(math.log(value) for value in values))


def _log_speedup_lcb95(values: Iterable[float]) -> float | None:
    values = tuple(values)
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        return None
    logs = [math.log(value) for value in values]
    standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(statistics.mean(logs) - _one_sided_t95(len(logs) - 1) * standard_error)


def evaluate_candidate(
    observations: Iterable[PromotionObservation],
    *,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> dict[str, object]:
    """Return auditable gate details for one hardware/dtype/shape/config point."""
    observations = tuple(observations)
    rounds_by_gpu: dict[str, list[float]] = defaultdict(list)
    raw_speedups = []
    data_valid = bool(observations)
    correctness_pass = bool(observations)
    sources = set()
    for observation in observations:
        sources.add(observation.source)
        correctness_pass = correctness_pass and observation.finite
        data_valid = data_valid and bool(observation.gpu_uuid)
        data_valid = data_valid and math.isfinite(observation.raw_speedup) and observation.raw_speedup > 0
        data_valid = data_valid and bool(observation.paired_round_speedups)
        data_valid = data_valid and all(
            math.isfinite(speedup) and speedup > 0 for speedup in observation.paired_round_speedups
        )
        rounds_by_gpu[observation.gpu_uuid].extend(observation.paired_round_speedups)
        raw_speedups.append(observation.raw_speedup)

    round_counts = {gpu_uuid: len(values) for gpu_uuid, values in sorted(rounds_by_gpu.items())}
    gpu_count = len(rounds_by_gpu)
    resource_pass = (
        data_valid
        and gpu_count >= thresholds.min_gpus
        and all(count >= thresholds.min_rounds_per_gpu for count in round_counts.values())
    )

    paired_lcb95 = None
    min_gpu_geomean = None
    balanced_rounds_per_gpu = 0
    if resource_pass:
        # Balance the pool so rerunning one GPU cannot give it more statistical
        # weight than the others. Input order is preserved within each GPU.
        balanced_rounds_per_gpu = min(round_counts.values())
        balanced = {
            gpu_uuid: values[:balanced_rounds_per_gpu]
            for gpu_uuid, values in rounds_by_gpu.items()
        }
        pooled = [speedup for values in balanced.values() for speedup in values]
        paired_lcb95 = _log_speedup_lcb95(pooled)
        min_gpu_geomean = min(_geometric_mean(values) for values in balanced.values())

    confidence_pass = (
        resource_pass
        and paired_lcb95 is not None
        and min_gpu_geomean is not None
        and paired_lcb95 >= thresholds.confidence_speedup
        and min_gpu_geomean >= thresholds.confidence_speedup
    )
    min_raw_speedup = min(raw_speedups) if raw_speedups else None
    # The margin path must hold for every input artifact.
    raw_margin_pass = (
        resource_pass
        and min_raw_speedup is not None
        and min_raw_speedup >= thresholds.raw_speedup
    )
    eligible = correctness_pass and (confidence_pass or raw_margin_pass)
    return {
        "eligible": eligible,
        "confidence_pass": confidence_pass,
        "raw_margin_pass": raw_margin_pass,
        "correctness_pass": correctness_pass,
        "resource_pass": resource_pass,
        "data_valid": data_valid,
        "gpu_count": gpu_count,
        "gpu_uuids": sorted(rounds_by_gpu),
        "rounds_by_gpu": round_counts,
        "balanced_rounds_per_gpu": balanced_rounds_per_gpu,
        "paired_log_speedup_lcb95": paired_lcb95,
        "minimum_gpu_paired_geomean": min_gpu_geomean,
        "minimum_raw_speedup": min_raw_speedup,
        "sources": sorted(sources),
        "thresholds": asdict(thresholds),
    }


def _gpu_uuid(payload: dict[str, object], source: Path) -> str:
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError(f"{source}: missing hardware metadata")
    gpu_uuid = hardware.get("gpu_uuid")
    if isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"):
        return gpu_uuid
    preflight = payload.get("gpu_idle_preflight")
    if isinstance(preflight, dict):
        gpu_uuid = preflight.get("uuid")
        if isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"):
            return gpu_uuid
    visible_devices = hardware.get("cuda_visible_devices")
    if isinstance(visible_devices, str) and visible_devices.startswith("GPU-") and "," not in visible_devices:
        return visible_devices
    raise ValueError(f"{source}: a physical GPU UUID is required for cross-GPU promotion")


def _payload_observations(
    payload: dict[str, object],
    *,
    source: Path,
) -> list[tuple[CandidateKey, PromotionObservation]]:
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError(f"{source}: missing hardware metadata")
    capability = hardware.get("compute_capability")
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError(f"{source}: invalid compute capability")
    gpu_uuid = _gpu_uuid(payload, source)
    sms = int(hardware["multiprocessor_count"])
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{source}: missing result rows")

    extracted = []
    for row in results:
        if not isinstance(row, dict):
            raise ValueError(f"{source}: invalid result row")
        route = row.get("route")
        match = _CONFIG_ROUTE.fullmatch(str(route))
        if match is None:
            continue
        paired = row.get("paired_round_speedups_vs_marlin")
        if not isinstance(paired, list):
            raise ValueError(f"{source}: {route} is missing paired per-round speedups; rerun the matrix scanner")
        key = CandidateKey(
            major=int(capability[0]),
            minor=int(capability[1]),
            sms=sms,
            dtype=str(row["dtype"]),
            m=int(row["m"]),
            k=int(row["k"]),
            n=int(row["n"]),
            config=int(match.group("config")),
        )
        extracted.append(
            (
                key,
                PromotionObservation(
                    gpu_uuid=gpu_uuid,
                    paired_round_speedups=tuple(float(value) for value in paired),
                    raw_speedup=float(row["speedup_vs_marlin"]),
                    finite=bool(row.get("finite", False)),
                    source=str(source),
                ),
            )
        )
    return extracted


def analyze_files(paths: Iterable[Path], *, thresholds: PromotionThresholds) -> list[dict[str, object]]:
    grouped: dict[CandidateKey, list[PromotionObservation]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object")
        for key, observation in _payload_observations(payload, source=path):
            grouped[key].append(observation)
    return [
        {**asdict(key), **evaluate_candidate(observations, thresholds=thresholds)}
        for key, observations in sorted(grouped.items())
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Matrix JSON artifacts from distinct physical GPUs.")
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--min-rounds-per-gpu", type=int, default=3)
    parser.add_argument("--confidence-speedup", type=float, default=1.05)
    parser.add_argument("--raw-speedup", type=float, default=1.07)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.min_gpus < 2:
        parser.error("--min-gpus must be at least 2")
    if args.min_rounds_per_gpu < 2:
        parser.error("--min-rounds-per-gpu must be at least 2")
    if args.confidence_speedup <= 1 or args.raw_speedup < args.confidence_speedup:
        parser.error("speedup thresholds must satisfy raw >= confidence > 1")
    return args


def main() -> None:
    args = _parse_args()
    paths = tuple(dict.fromkeys(path.resolve() for path in args.results))
    thresholds = PromotionThresholds(
        min_gpus=args.min_gpus,
        min_rounds_per_gpu=args.min_rounds_per_gpu,
        confidence_speedup=args.confidence_speedup,
        raw_speedup=args.raw_speedup,
    )
    results = analyze_files(paths, thresholds=thresholds)
    payload = {"inputs": [str(path) for path in paths], "thresholds": asdict(thresholds), "results": results}
    print("cc    SMs dtype M     K     N     cfg LCB95   min_raw gate")
    for row in results:
        lcb = row["paired_log_speedup_lcb95"]
        min_raw = row["minimum_raw_speedup"]
        print(
            f"{row['major']}.{row['minor']}  {row['sms']:>3} {row['dtype']:<5} {row['m']:>5} "
            f"{row['k']:>5} {row['n']:>5} {row['config']:>3} "
            f"{lcb if lcb is not None else float('nan'):>7.4f} "
            f"{min_raw if min_raw is not None else float('nan'):>7.4f} "
            f"{'PASS' if row['eligible'] else 'FAIL'}"
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
