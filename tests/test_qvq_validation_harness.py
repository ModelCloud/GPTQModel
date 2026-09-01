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
    if profile.runtime_oracle:
        reference["shared_input_runtime"] = {
            "mode": "plain_per_module_p32",
            "group_count": 0,
            "plain_fallback_count": 0,
        }
        candidate["shared_input_runtime"] = {
            "mode": "refactored_grouped_or_plain_p32",
            "group_count": 32,
            "plain_fallback_count": 0,
        }
    if profile.correction_required:
        for module in reference["modules"]:
            before = dict(module["tensor_sha256"])
            before["SV"] = f"before-{before['SV']}"
            module["fixed_trellis_correction"] = {
                "kind": "fixed_trellis_output_channel_sv",
                "SU_trainable": False,
                "accepted_channels": 1,
                "proxy_loss_before": 2.0,
                "proxy_loss_after": 1.0,
                "changed_tensors": ["SV"],
                "tensor_sha256_before": before,
            }
        candidate["modules"] = copy.deepcopy(reference["modules"])
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
    payload = {
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
    if profile.runtime_oracle:
        payload["runtime_oracle"] = {
            "mode": profile.runtime_oracle,
            "reference_arm": profile.reference_arm,
            "candidate_arm": profile.candidate_arm,
            "canonical_transform_plan": "A31",
            "canonical_payload_materializations": 1,
            "packed_runtime_equivalence": _quality_block(),
            "fixed_trellis_correction": (
                {
                    "kind": "fixed_trellis_output_channel_sv",
                    "mutable_tensors": ["SV"],
                    "immutable_tensors": [
                        "trellis",
                        "SU",
                        "bank_ids",
                        "bank_alt_id",
                        "bias",
                    ],
                    "SU_trainable": False,
                }
                if profile.correction_required
                else None
            ),
        }
        equivalence = payload["runtime_oracle"]["packed_runtime_equivalence"][
            "aggregate"
        ]
        equivalence.update(
            {
                "final_kl": 1e-7,
                "logits_relative_l2": 1e-4,
                "max_abs_logits_delta": 0.01,
                "top1": 1.0,
                "top5": 1.0,
                "top10": 1.0,
            }
        )
    return payload


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


def test_plain_refactored_oracle_accepts_one_canonical_payload(tmp_path):
    result = _validate(_write(tmp_path, _artifact("p0-r0-runtime")), "p0-r0-runtime")

    assert result["payload_identity"] is True
    assert result["runtime_oracle"]["candidate_group_count"] == 32
    assert result["runtime_oracle"]["packed_runtime_equivalence"]["top1"] == 1.0


def test_plain_refactored_oracle_rejects_packed_output_drift(tmp_path):
    payload = _artifact("p0-r0-runtime")
    payload["runtime_oracle"]["packed_runtime_equivalence"]["aggregate"][
        "logits_relative_l2"
    ] = 0.011

    with pytest.raises(RuntimeError, match="packed logits relative L2"):
        _validate(_write(tmp_path, payload), "p0-r0-runtime")


def test_corrected_oracle_accepts_sv_only_fixed_trellis_recovery(tmp_path):
    result = _validate(
        _write(tmp_path, _artifact("p0c-r0c-correction")),
        "p0c-r0c-correction",
    )

    correction = result["fixed_trellis_correction"]
    assert correction["corrected_module_records"] == 224
    assert correction["accepted_channels_across_both_identical_arms"] == 224
    assert correction["mutable_tensors"] == ["SV"]


def test_corrected_oracle_rejects_immutable_payload_drift(tmp_path):
    payload = _artifact("p0c-r0c-correction")
    payload["arms"]["P0+C"]["modules"][0]["tensor_sha256"]["SU"] = "changed"
    payload["arms"]["R0+C"]["modules"][0]["tensor_sha256"]["SU"] = "changed"

    with pytest.raises(RuntimeError, match="immutable SU"):
        _validate(_write(tmp_path, payload), "p0c-r0c-correction")


def test_corrected_oracle_rejects_no_accepted_recovery(tmp_path):
    payload = _artifact("p0c-r0c-correction")
    for arm in payload["arms"].values():
        for module in arm["modules"]:
            module["fixed_trellis_correction"]["accepted_channels"] = 0

    with pytest.raises(RuntimeError, match="accepted no output channels"):
        _validate(_write(tmp_path, payload), "p0c-r0c-correction")


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
