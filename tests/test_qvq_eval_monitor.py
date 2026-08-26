# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from scripts.qvq_eval_monitor import (
    discover_checkpoints,
    excluded_checkpoint,
    load_state,
    report_path,
    requested_jobs,
)


def _checkpoint(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "qvq_quantize_run.json").write_text("{}", encoding="utf-8")
    return path


def test_discovery_requires_manifest_and_excludes_contaminated_artifact(tmp_path):
    good = _checkpoint(tmp_path, "good")
    _checkpoint(tmp_path, "llama32-div300-sources-500k-bad")
    (tmp_path / "incomplete").mkdir()
    assert discover_checkpoints(tmp_path) == [good]
    assert excluded_checkpoint(tmp_path / "div300-sources-500k")


def test_report_names_are_stable(tmp_path):
    checkpoint = tmp_path / "arm"
    assert report_path(checkpoint, "gsm8k_platinum_cot").name.endswith("gsm8k-platinum-v1.json")
    assert report_path(checkpoint, "divergence300").name.endswith("div300-dev-v1.json")


def test_requested_jobs_skip_published_and_state_owned_reports(tmp_path):
    checkpoint = _checkpoint(tmp_path, "arm")
    state = load_state(tmp_path / "missing.json")
    jobs = requested_jobs(state, [checkpoint])
    assert {job.task for job in jobs} == {"gsm8k_platinum_cot", "divergence300"}
    jobs[0].output = str(report_path(checkpoint, jobs[0].task))
    state["jobs"][f"{checkpoint}|{jobs[0].task}"] = jobs[0].__dict__
    remaining = requested_jobs(state, [checkpoint])
    assert len(remaining) == 1
