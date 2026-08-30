# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from scripts import benchmark_qvq_lr_vs_gptq_llama32_1b as benchmark


def test_llama32_1b_projection_inventory_covers_qkvo_and_mlp_shapes():
    actual = {
        projection.name: (projection.in_features, projection.out_features)
        for projection in benchmark.LLAMA32_1B_PROJECTIONS
    }

    assert actual == {
        "q_proj": (2048, 2048),
        "k_proj": (2048, 512),
        "v_proj": (2048, 512),
        "o_proj": (2048, 2048),
        "gate_proj": (2048, 8192),
        "up_proj": (2048, 8192),
        "down_proj": (8192, 2048),
    }


def test_llama32_1b_unique_shape_matrix_preserves_all_seven_roles():
    roles = [role for case in benchmark.LLAMA32_1B_SHAPES for role in case.roles]
    geometries = {(case.in_features, case.out_features) for case in benchmark.LLAMA32_1B_SHAPES}

    assert roles == [
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert geometries == {(2048, 2048), (2048, 512), (2048, 8192), (8192, 2048)}
    assert sum(case.modules_per_layer for case in benchmark.LLAMA32_1B_SHAPES) == 7


def test_default_matrix_has_requested_rates_and_lr_row_specializations():
    assert benchmark.DEFAULT_QVQ_BITS == (2.0, 2.5, 3.0, 3.5)
    assert benchmark.DEFAULT_M_VALUES == (1, 2, 4, 8, 16)


def test_common_gptq_contract_rejects_group32_because_machete_cannot_run_it():
    with pytest.raises(ValueError, match="common Marlin/Machete"):
        benchmark._validate_contract(
            shapes=list(benchmark.LLAMA32_1B_SHAPES),
            m_values=list(benchmark.DEFAULT_M_VALUES),
            qvq_bits=list(benchmark.DEFAULT_QVQ_BITS),
            gptq_group_size=32,
        )


def _timed_row(kernel: str, median_ms: float, *, bits: float) -> dict:
    return {
        "name": "attn_qo",
        "m": 16,
        "kernel": kernel,
        "bits": bits,
        "median_ms": median_ms,
        "state": "complete",
    }


def test_relative_metrics_use_matched_shape_and_m_baselines():
    rows = benchmark._with_relative_metrics([
        _timed_row("qvq_lr", 0.5, bits=2.0),
        _timed_row("gptq_marlin", 1.0, bits=4.0),
        _timed_row("gptq_machete", 0.75, bits=4.0),
    ])

    qvq = rows[0]
    assert qvq["speedup_vs_marlin"] == pytest.approx(2.0)
    assert qvq["speedup_vs_machete"] == pytest.approx(1.5)
    assert rows[1]["speedup_vs_marlin"] == pytest.approx(1.0)
    assert rows[2]["speedup_vs_machete"] == pytest.approx(1.0)


def test_expected_matrix_contains_every_candidate_for_each_mkn():
    rows = benchmark._expected_rows(
        list(benchmark.LLAMA32_1B_SHAPES),
        list(benchmark.DEFAULT_M_VALUES),
        list(benchmark.DEFAULT_QVQ_BITS),
        "float16",
    )

    expected = (
        len(benchmark.LLAMA32_1B_SHAPES)
        * len(benchmark.DEFAULT_M_VALUES)
        * (len(benchmark.DEFAULT_QVQ_BITS) + 2)
    )
    assert len(rows) == expected == 120
    assert {row["kernel"] for row in rows} == {"qvq_lr", "gptq_marlin", "gptq_machete"}


def test_pre_timing_gate_waits_for_consecutive_idle_samples(monkeypatch, capsys):
    readings = iter([2, 0, 0, 0])
    waits = []

    monkeypatch.setattr(benchmark, "_compute_processes_for_uuid", lambda _uuid: [])
    monkeypatch.setattr(
        benchmark.benchmark_utils,
        "_query_gpu",
        lambda _gpu: {
            "index": "1",
            "pci.bus_id": "0000:44:00.0",
            "uuid": "GPU-test",
            "memory.used": "527",
            "utilization.gpu": str(next(readings)),
        },
    )

    class FakeEvent:
        def wait(self, interval):
            waits.append(interval)

    monkeypatch.setattr(benchmark.threading, "Event", FakeEvent)

    accepted = benchmark._pre_timing_exclusivity_gate(
        physical_gpu=1,
        gpu_uuid="GPU-test",
        samples=3,
        interval=0.25,
    )

    assert accepted["utilization.gpu"] == "0"
    assert waits == [0.25, 0.25, 0.25]
    assert "consecutive_idle_samples=3 attempts=4" in capsys.readouterr().out


def test_markdown_report_contains_complete_comparison_columns():
    payload = {
        "commit": "deadbeef",
        "benchmark_sha256": "cafebabe",
        "physical_gpu": 1,
        "hardware": {"pci.bus_id": "0000:44:00.0", "uuid": "GPU-test"},
        "device": {"name": "NVIDIA H100", "compute_capability": "9.0", "sm_count": 132},
        "software": {"torch": "test", "cuda": "13.0"},
        "args": {"dtype": "float16", "m": [1], "warmup": 1, "iterations": 2},
        "qvq": {"format": "qvq_v2b2_p32_lr", "rates": [3.0], "note": "P32 geometry"},
        "gptq": {"bits": 4, "group_size": 128},
        "rows": [{
            "name": "attn_qo",
            "roles": ["q_proj", "o_proj"],
            "m": 1,
            "k": 2048,
            "n": 2048,
            "kernel": "qvq_lr",
            "bits": 3.0,
            "median_ms": 0.1,
            "p95_ms": 0.11,
            "logical_tflops": 0.08,
            "effective_payload_gbs": 10.0,
            "speedup_vs_marlin": 0.5,
            "speedup_vs_machete": 0.6,
            "max_abs": 1e-5,
        }],
    }

    report = benchmark._markdown_report(payload)
    assert "CUDA Graph replay" in report
    assert "CPU scheduling and host launch gaps are outside each timed interval" in report
    assert "| Shape | Roles | M | K | N | Kernel | W | Group |" in report
    assert "| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 |" in report
