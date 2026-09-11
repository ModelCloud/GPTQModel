# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from scripts import zml_native_fast_abi_benchmark as benchmark

GPU_UUID = "GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2"


def _gpu_snapshot(*, used_mib: int = 401, utilization: int = 0, apps: str = "") -> dict[str, object]:
    return {
        "gpu": {
            "returncode": 0,
            "stdout": (
                f"1, 00000000:25:00.0, {GPU_UUID}, NVIDIA PG506-230, "
                f"98304, {used_mib}, {98304 - used_mib}, {utilization}"
            ),
            "stderr": "",
        },
        "compute_apps": {"returncode": 0, "stdout": apps, "stderr": ""},
    }


def _set_gpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)


def test_gpu_exclusivity_gate_accepts_three_uuid_pinned_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_environment(monkeypatch)
    pid = os.getpid()
    calls = []

    def snapshot(expected_uuid: str | None = None) -> dict[str, object]:
        calls.append(expected_uuid)
        return _gpu_snapshot(apps=f"{GPU_UUID}, {pid}, python, 400")

    monkeypatch.setattr(benchmark, "gpu_snapshot", snapshot)

    result = benchmark.gpu_exclusivity_gate(
        expected_uuid=GPU_UUID,
        samples=3,
        max_attempts=3,
        max_unexplained_memory_mib=16,
        allowed_pid=pid,
    )

    assert result["ok"] is True
    assert result["accepted_consecutive_idle_samples"] == 3
    assert calls == [GPU_UUID, GPU_UUID, GPU_UUID]
    assert all(sample["unexplained_memory_mib"] == 1 for sample in result["samples"])


def test_gpu_exclusivity_gate_rejects_foreign_process(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gpu_environment(monkeypatch)
    monkeypatch.setattr(
        benchmark,
        "gpu_snapshot",
        lambda expected_uuid=None: _gpu_snapshot(
            apps=f"{GPU_UUID}, 9999, python, 400",
        ),
    )
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)

    result = benchmark.gpu_exclusivity_gate(
        expected_uuid=GPU_UUID,
        samples=3,
        max_attempts=3,
        max_unexplained_memory_mib=16,
        allowed_pid=os.getpid(),
    )

    assert result["ok"] is False
    assert result["accepted_consecutive_idle_samples"] == 0
    assert result["samples"][0]["foreign_compute_apps"][0]["pid"] == 9999


def test_gpu_exclusivity_gate_requires_three_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gpu_environment(monkeypatch)

    with pytest.raises(ValueError, match="at least three"):
        benchmark.gpu_exclusivity_gate(
            expected_uuid=GPU_UUID,
            samples=2,
            max_attempts=2,
            max_unexplained_memory_mib=16,
            allowed_pid=None,
        )


def test_validate_trace_run_proves_p32_and_stable_addresses(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    stderr_path = tmp_path / "trace.stderr.log"
    stderr_path.write_text(
        """info: native phase=prefill input_addresses={ 1, 2, 3 }
info: native phase=prefill QVQ callbacks=.{ .standard = 80, .grouped = 0, .grouped_record_create = 0, .grouped_record_update = 0 }
info: native phase=prefill input_addresses={ 1, 2, 3 }
info: native phase=decode input_addresses={ 4, 5, 6 }
info: native phase=decode QVQ callbacks=.{ .standard = 80, .grouped = 0, .grouped_record_create = 0, .grouped_record_update = 0 }
info: native phase=decode input_addresses={ 4, 5, 6 }
""",
        encoding="utf-8",
    )

    result = benchmark.validate_trace_run(
        {"returncode": 0, "stderr_log": str(stderr_path)},
        trace_path,
    )

    assert result["status"] == "ok"
    assert result["p32_activation_proven"] is True
    assert result["stable_graph_input_addresses"] is True


def test_validate_trace_run_fails_closed_on_missing_callbacks(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    stderr_path = tmp_path / "trace.stderr.log"
    stderr_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="QVQ activation"):
        benchmark.validate_trace_run(
            {"returncode": 0, "stderr_log": str(stderr_path)},
            trace_path,
        )


def test_validate_timing_contract_requires_blocking_output_copy(tmp_path: Path) -> None:
    source = tmp_path / "examples/llm/llama_native_runtime.zig"
    source.parent.mkdir(parents=True)
    source.write_text(
        "runner.run();\noutput.toSlice();\nreturn result;\n",
        encoding="utf-8",
    )

    result = benchmark.validate_timing_contract(tmp_path)

    assert result["method"] == "host_wall_clock_around_synchronized_native_step"
    assert result["source_sha256"] == benchmark.sha256_file(source)


def test_unique_prompt_hashes_reject_duplicates() -> None:
    batches = [
        {
            "status": "ok",
            "batch": 2,
            "prompt_hashes_sha256_u32le_per_sequence": ["same", "same"],
        }
    ]

    with pytest.raises(RuntimeError, match="prompt uniqueness"):
        benchmark.validate_unique_prompt_hashes(batches)


def test_failure_classification_requires_oom_evidence_for_sigkill() -> None:
    assert benchmark.classify_failure(-signal.SIGKILL, "", "", {}) == "killed"
    assert benchmark.classify_failure(-signal.SIGKILL, "", "", {"oom_kill": 1}) == "oom"
    assert benchmark.classify_failure(1, "CUDA_ERROR_OUT_OF_MEMORY", "", {}) == "oom"
