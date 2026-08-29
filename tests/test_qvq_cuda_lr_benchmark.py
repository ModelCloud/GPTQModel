# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from scripts import benchmark_qvq_cuda_lr


def test_cuda_ordinals_follow_pci_order(monkeypatch):
    def fake_check_output(command, **kwargs):
        assert command[:2] == ["nvidia-smi", "--query-gpu=index,pci.bus_id"]
        return "5, 00000000:03:00.0\n7, 00000000:01:00.0\n2, 00000000:02:00.0\n"

    monkeypatch.setattr(benchmark_qvq_cuda_lr.subprocess, "check_output", fake_check_output)

    assert benchmark_qvq_cuda_lr._cuda_ordinals_by_physical_gpu() == {7: 0, 2: 1, 5: 2}


def _row(path: str, median_ms: float, *, max_abs: float = 0.0) -> dict:
    return {
        "physical_gpu": 0,
        "pci_bus_id": "00000000:01:00.0",
        "uuid": "GPU-test",
        "gpu_name": "test",
        "compute_capability": "8.9",
        "sm_count": 128,
        "shape": "m1_narrow",
        "k": 2048,
        "n": 256,
        "dtype": "float16",
        "bits": 2.0,
        "m": 1,
        "rows": 1,
        "path": path,
        "split_count": 4,
        "median_ms": median_ms,
        "max_abs": max_abs,
    }


def test_regression_gate_rejects_slow_pair():
    report = benchmark_qvq_cuda_lr._regression_report(
        [_row("lr_native_s2", 2.0), _row("non_lr_native", 1.0)]
    )

    assert report["regression_summary"]["cases"] == 1
    with pytest.raises(AssertionError, match="performance gate"):
        benchmark_qvq_cuda_lr._enforce_regression_report(report, min_geomean_speedup=1.5)


def test_regression_gate_rejects_missing_pairs():
    report = benchmark_qvq_cuda_lr._regression_report([])

    with pytest.raises(AssertionError, match="no paired LR/non-LR cases"):
        benchmark_qvq_cuda_lr._enforce_regression_report(report, min_geomean_speedup=1.5)


def test_explicit_only_regression_summary_prints_without_production_geomean(capsys):
    report = benchmark_qvq_cuda_lr._regression_report(
        [_row("lr_native_s2", 0.5), _row("non_lr_native", 1.0)]
    )

    benchmark_qvq_cuda_lr._print_regression_summary(report)

    output = capsys.readouterr().out
    assert "production_cases=0" in output
    assert "production_geomean=n/a" in output


def test_regression_gate_cannot_hide_slow_auto_behind_tuning_candidate():
    report = benchmark_qvq_cuda_lr._regression_report(
        [
            _row("lr_native_auto", 2.0),
            _row("lr_native_s2", 0.125, max_abs=0.0),
            _row("non_lr_native", 1.0),
        ]
    )

    assert report["regression_summary"]["geomean_speedup_vs_non_lr"] == pytest.approx(2.0)
    assert report["regression_summary"]["production_geomean_speedup_vs_non_lr"] == pytest.approx(0.5)
    with pytest.raises(AssertionError, match="performance gate"):
        benchmark_qvq_cuda_lr._enforce_regression_report(report, min_geomean_speedup=1.5)


def test_regression_gate_rejects_accuracy_failure():
    report = benchmark_qvq_cuda_lr._regression_report(
        [_row("lr_native_auto", 1.0, max_abs=3e-3), _row("non_lr_native", 2.0)]
    )

    with pytest.raises(AssertionError, match="accuracy gate"):
        benchmark_qvq_cuda_lr._enforce_regression_report(report, min_geomean_speedup=1.5)
