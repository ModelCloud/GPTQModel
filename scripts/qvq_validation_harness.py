#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Deterministic validation harness for QVQ P32 runtime/refactor work.

This is intentionally a thin orchestration and verification layer around the
full-model packed CUDA runner.  It freezes the experiment contract before the
runtime/checkpoint refactor begins.

Two initial profiles are supported:

* ``a31-a41-runtime`` fits A31 once and requires A41 to reuse byte-identical
  payloads.  This isolates runtime compilation/grouping from quantization.
* ``a0-a41-production`` independently fits A0 and A41 with the same data/seed.
  This measures the current production-control versus optimized-candidate
  tradeoff.

The harness never accepts a benchmark artifact merely because the subprocess
returned zero.  It validates provenance, dataset identity, dense parity,
packed/reconstructed coverage, timing sample counts, and payload identity when
that is part of the profile contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "scripts/benchmark_qvq_rotation_full_model_cuda.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/qvq_validation"


@dataclass(frozen=True)
class HarnessProfile:
    name: str
    reference_arm: str
    candidate_arm: str
    reuse_identical_payloads: bool
    expected_hadamards: dict[str, int]
    require_equal_ebpw: bool = True


PROFILES = {
    "a31-a41-runtime": HarnessProfile(
        name="a31-a41-runtime",
        reference_arm="A31",
        candidate_arm="A41",
        reuse_identical_payloads=True,
        expected_hadamards={"A31": 9, "A41": 9},
    ),
    "a0-a41-production": HarnessProfile(
        name="a0-a41-production",
        reference_arm="A0",
        candidate_arm="A41",
        reuse_identical_payloads=False,
        expected_hadamards={"A0": 14, "A41": 9},
    ),
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _assert_clean_tree(*, allow_dirty: bool) -> None:
    dirty = _git("status", "--porcelain")
    if dirty and not allow_dirty:
        raise RuntimeError(
            "validation harness requires a clean git tree; commit/stash changes "
            "or pass --allow-dirty for a non-promotable diagnostic run"
        )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{path}: root JSON value must be an object")
    return payload


def _sample_keys(rows: list[dict[str, Any]]) -> set[tuple[int, str]]:
    return {(int(row["row"]), str(row["sha256"])) for row in rows}


def _validate_dataset_contract(payload: dict[str, Any]) -> dict[str, Any]:
    calibration = payload.get("calibration", [])
    streams = payload.get("validation_streams", [])
    _require(len(calibration) == 16, "expected exactly 16 calibration samples")
    _require(len(streams) == 3, "expected exactly three validation streams")
    _require(
        all(len(stream) == 16 for stream in streams),
        "expected exactly 16 samples per validation stream",
    )

    calibration_keys = _sample_keys(calibration)
    _require(
        len(calibration_keys) == len(calibration),
        "duplicate calibration rows/hashes detected",
    )
    validation_rows = [row for stream in streams for row in stream]
    validation_keys = _sample_keys(validation_rows)
    _require(
        len(validation_keys) == len(validation_rows),
        "validation streams are not disjoint",
    )
    _require(
        "test split" in str(payload.get("final_evaluation_source", "")).lower()
        and "not read" in str(payload.get("final_evaluation_source", "")).lower(),
        "reserved final evaluation split must remain unread",
    )
    return {
        "calibration_samples": len(calibration),
        "validation_streams": len(streams),
        "validation_samples": len(validation_rows),
        "validation_tokens": sum(int(row["tokens"]) for row in validation_rows),
    }


def _module_hashes(arm: dict[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for module in arm.get("modules", []):
        name = str(module["module"])
        _require(name not in result, f"duplicate module record {name!r}")
        hashes = module.get("tensor_sha256")
        _require(isinstance(hashes, dict) and hashes, f"{name}: missing tensor hashes")
        result[name] = {str(key): str(value) for key, value in hashes.items()}
    return result


def _validate_quality_block(name: str, block: dict[str, Any]) -> dict[str, Any]:
    aggregate = block.get("aggregate")
    _require(isinstance(aggregate, dict), f"{name}: missing aggregate quality")
    required = (
        "tokens",
        "final_kl",
        "logits_relative_l2",
        "max_abs_logits_delta",
        "top1",
        "top5",
        "top10",
    )
    for key in required:
        _require(_finite_number(aggregate.get(key)), f"{name}: invalid {key}")
    per_text = aggregate.get("per_text")
    _require(isinstance(per_text, list) and len(per_text) == 48, f"{name}: expected 48 per-text rows")
    _require(int(aggregate["tokens"]) > 0, f"{name}: zero validation tokens")
    return {
        key: aggregate[key]
        for key in required
    }


def _validate_timing_arm(
    arm_name: str,
    arm: dict[str, Any],
    *,
    timing_cycles: int,
    prefill_iterations: int,
    decode_iterations: int,
    suite_iterations: int,
) -> dict[int, dict[str, float]]:
    benchmark = arm.get("decode_benchmark")
    _require(isinstance(benchmark, dict), f"{arm_name}: missing decode benchmark")
    cycles = benchmark.get("cycles")
    _require(isinstance(cycles, list) and len(cycles) == timing_cycles, f"{arm_name}: timing cycle count mismatch")
    aggregate = benchmark.get("aggregate")
    _require(isinstance(aggregate, list), f"{arm_name}: missing aggregate decode timing")

    result = {}
    for row in aggregate:
        batch = int(row["batch_size"])
        prefill = row["prefill"]
        decode = row["decode"]
        _require(
            len(prefill["wall_samples_ms"]) == timing_cycles * prefill_iterations,
            f"{arm_name} B{batch}: prefill raw-sample count mismatch",
        )
        _require(
            len(decode["wall_samples_ms"]) == timing_cycles * decode_iterations,
            f"{arm_name} B{batch}: decode raw-sample count mismatch",
        )
        _require(
            all(_finite_number(value) and value > 0 for value in decode["wall_samples_ms"]),
            f"{arm_name} B{batch}: invalid decode wall samples",
        )
        result[batch] = {
            "decode_wall_median_ms": float(decode["wall_median_ms"]),
            "decode_wall_p95_ms": float(decode["wall_p95_ms"]),
            "tokens_per_second": float(decode["tokens_per_second"]),
            "prefill_wall_median_ms": float(prefill["wall_median_ms"]),
        }

    suite = arm.get("quant_linear_suite_benchmark")
    _require(isinstance(suite, dict), f"{arm_name}: missing QuantLinear suite timing")
    _require(
        len(suite.get("cycles", [])) == timing_cycles,
        f"{arm_name}: QuantLinear suite cycle count mismatch",
    )
    for row in suite.get("aggregate", []):
        batch = int(row["batch_size"])
        samples = row["timing"]["wall_samples_ms"]
        _require(
            len(samples) == timing_cycles * suite_iterations,
            f"{arm_name} B{batch}: QuantLinear raw-sample count mismatch",
        )
    return result


def _validate_bootstrap(payload: dict[str, Any], profile: HarnessProfile) -> dict[str, Any]:
    key = f"{profile.candidate_arm}_minus_{profile.reference_arm}"
    comparison = payload.get("comparisons", {}).get(key)
    _require(isinstance(comparison, dict), f"missing paired comparison {key}")
    packed = comparison.get("packed")
    _require(isinstance(packed, dict), f"{key}: missing packed bootstrap")
    kl = packed["final_kl_delta_candidate_minus_reference"]
    top1 = packed["top1_delta_candidate_minus_reference"]
    for metric_name, metric in (("KL", kl), ("Top-1", top1)):
        ci = metric.get("ci95")
        _require(
            isinstance(ci, list)
            and len(ci) == 2
            and all(_finite_number(value) for value in ci),
            f"{key}: invalid {metric_name} confidence interval",
        )
    return {
        "kl_delta_median": float(kl["median"]),
        "kl_ci95": [float(value) for value in kl["ci95"]],
        "top1_delta_median": float(top1["median"]),
        "top1_ci95": [float(value) for value in top1["ci95"]],
        "kl_ci_crosses_zero": kl["ci95"][0] <= 0 <= kl["ci95"][1],
        "top1_ci_crosses_zero": top1["ci95"][0] <= 0 <= top1["ci95"][1],
    }


def validate_artifact(
    path: Path,
    profile: HarnessProfile,
    *,
    expected_revision: str,
    timing_cycles: int,
    prefill_iterations: int,
    decode_iterations: int,
    suite_iterations: int,
) -> dict[str, Any]:
    payload = _load_artifact(path)
    _require(payload.get("schema") == "qvq.rotation-folding.packed-cuda-full-model.v1", "unexpected engine artifact schema")
    _require(payload.get("status") == "complete", "benchmark artifact is not complete")
    _require(payload.get("repository_revision") == expected_revision, "artifact git revision does not match harness revision")
    _require(float(payload.get("bits")) == 2.0, "harness requires W2")
    _require(int(payload.get("quantized_layers")) == 16, "harness requires all 16 layers")
    _require(int(payload.get("sequence_length")) == 128, "harness requires sequence length 128")
    _require(payload.get("startup_compute_processes") == [], "benchmark did not start on an exclusive GPU")

    dataset = _validate_dataset_contract(payload)
    arms = payload.get("arms", {})
    for arm_name in (profile.reference_arm, profile.candidate_arm):
        _require(arm_name in arms, f"artifact is missing {arm_name}")
        arm = arms[arm_name]
        _require(arm.get("status") == "complete", f"{arm_name}: fit/runtime is incomplete")
        _require(len(arm.get("modules", [])) == 112, f"{arm_name}: expected 112 quantized projections")
        _require(
            int(arm.get("online_hadamards_per_block")) == profile.expected_hadamards[arm_name],
            f"{arm_name}: unexpected transform count",
        )
        dense = arm.get("dense_parity")
        _require(isinstance(dense, dict), f"{arm_name}: missing dense parity")
        _require(
            float(dense["logits_relative_l2"]) <= 2e-5,
            f"{arm_name}: dense logits relative L2 gate failed",
        )
        _require(float(dense["top1"]) == 1.0, f"{arm_name}: dense Top-1 identity gate failed")
        _validate_quality_block(f"{arm_name}.reconstructed", arm["reconstructed_quality"])
        _validate_quality_block(f"{arm_name}.packed", arm["packed_quality"])

    reference = arms[profile.reference_arm]
    candidate = arms[profile.candidate_arm]
    reference_hashes = _module_hashes(reference)
    candidate_hashes = _module_hashes(candidate)
    if profile.reuse_identical_payloads:
        _require(
            candidate.get("fit_reused_from") == profile.reference_arm,
            f"{profile.candidate_arm}: expected fit reuse from {profile.reference_arm}",
        )
        _require(
            reference_hashes == candidate_hashes,
            "runtime-equivalence profile requires byte-identical per-module payload hashes",
        )
    if profile.require_equal_ebpw:
        _require(
            abs(float(reference["effective_bpw"]) - float(candidate["effective_bpw"])) <= 1e-9,
            "profile requires identical effective BPW",
        )

    reference_timing = _validate_timing_arm(
        profile.reference_arm,
        reference,
        timing_cycles=timing_cycles,
        prefill_iterations=prefill_iterations,
        decode_iterations=decode_iterations,
        suite_iterations=suite_iterations,
    )
    candidate_timing = _validate_timing_arm(
        profile.candidate_arm,
        candidate,
        timing_cycles=timing_cycles,
        prefill_iterations=prefill_iterations,
        decode_iterations=decode_iterations,
        suite_iterations=suite_iterations,
    )
    _require(reference_timing.keys() == candidate_timing.keys(), "timed batch sets differ")

    timing_delta = {}
    for batch in reference_timing:
        reference_ms = reference_timing[batch]["decode_wall_median_ms"]
        candidate_ms = candidate_timing[batch]["decode_wall_median_ms"]
        timing_delta[str(batch)] = {
            "reference_ms": reference_ms,
            "candidate_ms": candidate_ms,
            "candidate_latency_delta_fraction": candidate_ms / reference_ms - 1.0,
            "candidate_speedup": reference_ms / candidate_ms,
        }

    return {
        "artifact": str(path),
        "status": "pass",
        "dataset": dataset,
        "payload_identity_required": profile.reuse_identical_payloads,
        "payload_identity": reference_hashes == candidate_hashes,
        "reference_effective_bpw": float(reference["effective_bpw"]),
        "candidate_effective_bpw": float(candidate["effective_bpw"]),
        "bootstrap": _validate_bootstrap(payload, profile),
        "decode": timing_delta,
    }


def _engine_command(
    profile: HarnessProfile,
    *,
    seed: int,
    output: Path,
    local_files_only: bool,
    timing_cycles: int,
    prefill_iterations: int,
    decode_iterations: int,
    suite_iterations: int,
) -> list[str]:
    command = [
        sys.executable,
        str(ENGINE),
        "--arms",
        f"{profile.reference_arm},{profile.candidate_arm}",
        "--bits",
        "2",
        "--seed",
        str(seed),
        "--layers",
        "16",
        "--calibration-samples",
        "16",
        "--validation-streams",
        "3",
        "--validation-samples",
        "16",
        "--sequence-length",
        "128",
        "--decode-batches",
        "1",
        "2",
        "4",
        "8",
        "--prefill-warmup",
        "2",
        "--prefill-iterations",
        str(prefill_iterations),
        "--decode-warmup",
        "5",
        "--decode-iterations",
        str(decode_iterations),
        "--module-suite-warmup",
        "10",
        "--module-suite-iterations",
        str(suite_iterations),
        "--timing-cycles",
        str(timing_cycles),
        "--json",
        str(output),
    ]
    if profile.reuse_identical_payloads:
        command.append("--reuse-identical-a31-payloads")
    if local_files_only:
        command.append("--local-files-only")
    return command


def _engine_environment(*, local_files_only: bool) -> dict[str, str]:
    environment = os.environ.copy()
    if local_files_only:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["HF_DATASETS_OFFLINE"] = "1"
    return environment


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="a31-a41-runtime")
    parser.add_argument("--seeds", type=_parse_seeds, default=(20260831,))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timing-cycles", type=int, default=5)
    parser.add_argument("--prefill-iterations", type=int, default=5)
    parser.add_argument("--decode-iterations", type=int, default=30)
    parser.add_argument("--suite-iterations", type=int, default=30)
    args = parser.parse_args()

    _assert_clean_tree(allow_dirty=args.allow_dirty)
    revision = _git("rev-parse", "HEAD")
    profile = PROFILES[args.profile]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        artifact = args.output_dir / f"{profile.name}_seed{seed}_{revision[:8]}.json"
        command = _engine_command(
            profile,
            seed=seed,
            output=artifact,
            local_files_only=args.local_files_only,
            timing_cycles=args.timing_cycles,
            prefill_iterations=args.prefill_iterations,
            decode_iterations=args.decode_iterations,
            suite_iterations=args.suite_iterations,
        )
        if not args.validate_only:
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                env=_engine_environment(local_files_only=args.local_files_only),
            )
        _require(artifact.is_file(), f"benchmark artifact does not exist: {artifact}")
        run = validate_artifact(
            artifact,
            profile,
            expected_revision=revision,
            timing_cycles=args.timing_cycles,
            prefill_iterations=args.prefill_iterations,
            decode_iterations=args.decode_iterations,
            suite_iterations=args.suite_iterations,
        )
        run["seed"] = seed
        run["engine_command"] = command
        runs.append(run)

    summary = {
        "schema": "qvq.validation-harness.v1",
        "status": "pass",
        "profile": profile.name,
        "reference_arm": profile.reference_arm,
        "candidate_arm": profile.candidate_arm,
        "repository_revision": revision,
        "promotable": not args.allow_dirty,
        "runs": runs,
    }
    summary_path = args.output_dir / f"summary_{profile.name}_{revision[:8]}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"validation harness PASS: {profile.name}")
    print(f"revision: {revision}")
    print(f"summary: {summary_path}")
    for run in runs:
        b1 = run["decode"]["1"]
        print(
            f"seed {run['seed']}: B1 {b1['reference_ms']:.4f} -> "
            f"{b1['candidate_ms']:.4f} ms, speedup {b1['candidate_speedup']:.4f}x"
        )


if __name__ == "__main__":
    main()
