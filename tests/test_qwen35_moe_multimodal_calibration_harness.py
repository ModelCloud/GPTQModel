"""Lightweight contract tests for the standalone Qwen3.5 calibration harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_qwen35_moe_multimodal_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("qwen35_calibration_harness", SCRIPT)
HARNESS = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(HARNESS)


def _args(**overrides):
    args = HARNESS.parse_args(["--model-path", "fixture", "--dry-run"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_protocol_hash_is_order_independent():
    assert HARNESS.protocol_hash({"b": 2, "a": [1, True]}) == HARNESS.protocol_hash(
        {"a": [1, True], "b": 2}
    )


def test_tensor_summary_is_finite_and_bounded():
    try:
        import torch
    except ImportError:
        return
    summary = HARNESS.tensor_summary(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), include_values=True, max_values=2
    )
    assert summary["shape"] == [2, 2]
    assert summary["finite"] is True
    assert len(summary["values"]) == 2
    assert summary["mean"] == 2.5


def test_manifest_validation_rejects_failed_or_partial_results():
    base = {
        "schema_version": 1,
        "success": True,
        "engine": "dense",
        "model": "fixture",
        "task": "task",
        "metric": "metric",
        "score": 1.0,
        "samples": 2,
        "expected_samples": 2,
        "protocol_hash": "abc",
        "timing": {},
        "versions": {"engine": "fixture-1"},
        "gpus": [],
        "command": "cmd",
        "config": "cfg",
        "raw_result": "raw",
        "log": "log",
    }
    assert HARNESS.validate_manifest(base) == []
    partial = dict(base, success=False)
    assert "success_not_true" in HARNESS.validate_manifest(partial)
    incomplete = dict(base, samples=1)
    assert "sample_count" in HARNESS.validate_manifest(incomplete)


def test_cell_plan_contains_required_matrix_and_protocol(tmp_path):
    args = _args()
    configs = HARNESS.make_cell_configs(args)
    assert [config["cell"] for config in configs] == list(HARNESS.CELL_ORDER)
    assert len({config["protocol_hash"] for config in configs}) == 1
    assert configs[0]["calibration_modality"] == "none"
    assert all(config["inference_modality"] == "multimodal" for config in configs)
    assert configs[1]["calibration_modality"] == "text"
    assert configs[2]["calibration_modality"] == "multimodal"
    HARNESS.prepare_artifacts(
        tmp_path, configs, SCRIPT.parents[1], "python harness --dry-run"
    )
    assert (tmp_path / "source_state.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    assert {path.stem for path in (tmp_path / "configs").glob("*.json")} == set(
        HARNESS.CELL_ORDER
    )


def test_dry_run_cli_writes_auditable_plan(tmp_path, capsys):
    assert (
        HARNESS.main(
            [
                "--model-path",
                "fixture",
                "--artifact-root",
                str(tmp_path),
                "--cells",
                "dense,gptq-text",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["cells"] == {"dense": "planned", "gptq-text": "planned"}
    assert not list((tmp_path / "manifests").glob("*.json"))
