# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.qvq_validation_harness import (
    PROFILES,
    _engine_environment,
    validate_artifact,
)


def _quality_block():
    per_text = [
        {
            "tokens": 100,
            "final_kl": 0.1,
            "logits_relative_l2": 0.2,
            "top1": 0.8,
            "top5": 0.9,
            "top10": 0.95,
        }
        for _ in range(48)
    ]
    return {
        "aggregate": {
            "tokens": 4800,
            "final_kl": 0.1,
            "logits_relative_l2": 0.2,
            "max_abs_logits_delta": 1.0,
            "top1": 0.8,
            "top5": 0.9,
            "top10": 0.95,
            "per_text": per_text,
        },
        "streams": [],
    }


def _timing_block(cycles=2, prefill_iterations=2, decode_iterations=3):
    cycle = [
        {
            "batch_size": 1,
            "prompt_tokens": 128,
            "prefill": {
                "event_samples_ms": [1.0] * prefill_iterations,
                "wall_samples_ms": [1.0] * prefill_iterations,
            },
            "decode": {
                "event_samples_ms": [2.0] * decode_iterations,
                "wall_samples_ms": [2.0] * decode_iterations,
            },
        }
    ]
    return {
        "cycles": [copy.deepcopy(cycle) for _ in range(cycles)],
        "aggregate": [
            {
                "batch_size": 1,
                "prompt_tokens": 128,
                "prefill": {
                    "wall_median_ms": 1.0,
                    "wall_p95_ms": 1.0,
                    "tokens_per_second": 128000.0,
                    "wall_samples_ms": [1.0] * (cycles * prefill_iterations),
                },
                "decode": {
                    "wall_median_ms": 2.0,
                    "wall_p95_ms": 2.0,
                    "tokens_per_second": 500.0,
                    "wall_samples_ms": [2.0] * (cycles * decode_iterations),
                },
            }
        ],
    }


def _suite_block(cycles=2, iterations=3):
    cycle = [
        {
            "batch_size": 1,
            "quantized_layers": 16,
            "packed_projections": 112,
            "timing": {"wall_samples_ms": [1.0] * iterations},
        }
    ]
    return {
        "cycles": [copy.deepcopy(cycle) for _ in range(cycles)],
        "aggregate": [
            {
                "batch_size": 1,
                "quantized_layers": 16,
                "packed_projections": 112,
                "timing": {"wall_samples_ms": [1.0] * (cycles * iterations)},
            }
        ],
    }


def _arm(name: str, hadamards: int):
    modules = []
    for index in range(112):
        modules.append(
            {
                "module": f"model.layers.{index // 7}.fake.{index}",
                "tensor_sha256": {
                    "trellis": f"trellis-{index}",
                    "SU": f"su-{index}",
                    "SV": f"sv-{index}",
                    "bank_ids": f"bank-{index}",
                    "bank_alt_id": f"alt-{index}",
                },
            }
        )
    return {
        "status": "complete",
        "online_hadamards_per_block": hadamards,
        "dense_parity": {
            "logits_relative_l2": 1e-6,
            "top1": 1.0,
        },
        "modules": modules,
        "effective_bpw": 2.054419024,
        "reconstructed_quality": _quality_block(),
        "packed_quality": _quality_block(),
        "decode_benchmark": _timing_block(),
        "quant_linear_suite_benchmark": _suite_block(),
    }


def _artifact(profile_name: str, revision="deadbeef"):
    profile = PROFILES[profile_name]
    reference = _arm(profile.reference_arm, profile.expected_hadamards[profile.reference_arm])
    candidate = _arm(profile.candidate_arm, profile.expected_hadamards[profile.candidate_arm])
    if profile.reuse_identical_payloads:
        candidate["fit_reused_from"] = profile.reference_arm
    calibration = [
        {"row": index, "sha256": f"cal-{index}", "tokens": 100}
        for index in range(16)
    ]
    streams = [
        [
            {
                "row": 1000 + stream * 100 + index,
                "sha256": f"val-{stream}-{index}",
                "tokens": 100,
            }
            for index in range(16)
        ]
        for stream in range(3)
    ]
    comparison_key = f"{profile.candidate_arm}_minus_{profile.reference_arm}"
    return {
        "schema": "qvq.rotation-folding.packed-cuda-full-model.v1",
        "status": "complete",
        "repository_revision": revision,
        "bits": 2.0,
        "quantized_layers": 16,
        "sequence_length": 128,
        "startup_compute_processes": [],
        "final_evaluation_source": "reserved wikitext test split (not read)",
        "calibration": calibration,
        "validation_streams": streams,
        "arms": {
            profile.reference_arm: reference,
            profile.candidate_arm: candidate,
        },
        "comparisons": {
            comparison_key: {
                "packed": {
                    "final_kl_delta_candidate_minus_reference": {
                        "median": 0.0,
                        "ci95": [-0.01, 0.01],
                    },
                    "top1_delta_candidate_minus_reference": {
                        "median": 0.0,
                        "ci95": [-0.01, 0.01],
                    },
                }
            }
        },
    }


def _write(tmp_path: Path, payload):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _validate(path, profile_name, revision="deadbeef"):
    return validate_artifact(
        path,
        PROFILES[profile_name],
        expected_revision=revision,
        timing_cycles=2,
        prefill_iterations=2,
        decode_iterations=3,
        suite_iterations=3,
    )


def test_runtime_profile_accepts_identical_payloads(tmp_path):
    result = _validate(_write(tmp_path, _artifact("a31-a41-runtime")), "a31-a41-runtime")
    assert result["status"] == "pass"
    assert result["payload_identity"] is True
    assert result["bootstrap"]["kl_ci_crosses_zero"] is True


def test_runtime_profile_rejects_payload_drift(tmp_path):
    payload = _artifact("a31-a41-runtime")
    payload["arms"]["A41"]["modules"][0]["tensor_sha256"]["SU"] = "changed"
    with pytest.raises(RuntimeError, match="byte-identical"):
        _validate(_write(tmp_path, payload), "a31-a41-runtime")


def test_harness_rejects_dense_parity_regression(tmp_path):
    payload = _artifact("a0-a41-production")
    payload["arms"]["A41"]["dense_parity"]["logits_relative_l2"] = 1e-3
    with pytest.raises(RuntimeError, match="dense logits relative L2"):
        _validate(_write(tmp_path, payload), "a0-a41-production")


def test_harness_rejects_validation_overlap(tmp_path):
    payload = _artifact("a0-a41-production")
    payload["validation_streams"][1][0] = copy.deepcopy(payload["validation_streams"][0][0])
    with pytest.raises(RuntimeError, match="not disjoint"):
        _validate(_write(tmp_path, payload), "a0-a41-production")


def test_harness_rejects_missing_raw_timing_samples(tmp_path):
    payload = _artifact("a0-a41-production")
    payload["arms"]["A41"]["decode_benchmark"]["aggregate"][0]["decode"]["wall_samples_ms"].pop()
    with pytest.raises(RuntimeError, match="raw-sample count mismatch"):
        _validate(_write(tmp_path, payload), "a0-a41-production")


def test_harness_rejects_revision_mismatch(tmp_path):
    payload = _artifact("a0-a41-production", revision="old")
    with pytest.raises(RuntimeError, match="git revision"):
        _validate(_write(tmp_path, payload), "a0-a41-production", revision="new")


def test_local_files_only_applies_to_entire_engine_process(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")

    environment = _engine_environment(local_files_only=True)

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
