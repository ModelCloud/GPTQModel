# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_deepseek_v4_quantization.py"
SPEC = importlib.util.spec_from_file_location("profile_deepseek_v4_quantization", SCRIPT)
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


def test_parse_physical_gpus_rejects_duplicates():
    assert PROFILE._parse_physical_gpus("6,7") == [6, 7]
    with pytest.raises(ValueError, match="Duplicate"):
        PROFILE._parse_physical_gpus("6,6")


def test_profiler_idle_utilization_override_is_bounded(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--idle-max-utilization", "5"])
    assert PROFILE._parse_args().idle_max_utilization == 5
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--idle-max-utilization", "101"])
    with pytest.raises(SystemExit):
        PROFILE._parse_args()


def test_profiler_moe_capture_stream_count_must_be_positive(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--moe-capture-streams", "0"])
    with pytest.raises(SystemExit):
        PROFILE._parse_args()


def test_profiler_can_disable_parallel_moe_output_replay(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--no-moe-parallel-output-replay"])
    assert PROFILE._parse_args().moe_parallel_output_replay is False


def test_region_period_snapshots_supports_current_and_legacy_timers():
    class CurrentTimer:
        @staticmethod
        def period_snapshots():
            return [{"label": "layer:0", "regions": {}}]

    class LegacyTimer:
        pass

    assert PROFILE._region_period_snapshots(CurrentTimer()) == [{"label": "layer:0", "regions": {}}]
    assert PROFILE._region_period_snapshots(LegacyTimer()) == []


def test_quant_log_summary_separates_timing_from_quality_fingerprint():
    baseline = [
        {"layer": 0, "module": "q_proj", "loss": 0.25, "samples": 128, "damp": 0.05, "time": 2.0},
        {"layer": 1, "module": "q_proj", "loss": 0.5, "samples": 128, "damp": 0.05, "time": 3.0},
    ]
    faster = [{**row, "time": row["time"] / 4} for row in baseline]
    faster.reverse()

    baseline_summary = PROFILE._summarize_quant_log(baseline)
    faster_summary = PROFILE._summarize_quant_log(faster)

    assert baseline_summary["quality_fingerprint_sha256"] == faster_summary["quality_fingerprint_sha256"]
    assert baseline_summary["layers"]["0"]["module_time_sum_s"] == 2.0
    assert faster_summary["layers"]["0"]["module_time_sum_s"] == 0.5


def test_subset_profiler_records_forward_and_quant_phases():
    profiler = PROFILE.SubsetProfiler(enable_nvtx=False)
    common = {
        "layer_idx": 0,
        "subset_index": 0,
        "subset_total": 2,
        "module_names": ["q_proj", "k_proj"],
        "processor": "gptq",
    }
    profiler.subset_event(stage="forward_start", **common)
    profiler.subset_event(stage="forward_end", **common)
    profiler.subset_event(stage="quant_start", **common)
    profiler.subset_event(stage="quant_complete", **common)

    snapshot = profiler.snapshot()
    assert [event["phase"] for event in snapshot["subsets"]] == ["forward", "quant"]
    assert all(event["module_count"] == 2 for event in snapshot["subsets"])
    assert snapshot["open_event_count"] == 0


def test_subset_profiler_normalizes_callable_processor_name():
    profiler = PROFILE.SubsetProfiler(enable_nvtx=False)

    class Processor:
        @staticmethod
        def name():
            return "gptq"

    common = {
        "layer_idx": 0,
        "subset_index": 0,
        "subset_total": 1,
        "module_names": ["q_proj"],
        "processor": Processor.name,
    }
    profiler.subset_event(stage="quant_start", **common)
    profiler.subset_event(stage="quant_complete", **common)

    assert profiler.snapshot()["subsets"][0]["processor"] == "gptq"


def test_prepare_calibration_rows_defaults_to_raw_message_content():
    records = [{"messages": [{"role": "user", "content": "alpha"}, {"role": "user", "content": "beta"}]}]

    assert PROFILE._prepare_calibration_rows(records, apply_chat_template=False) == [{"text": "alpha\nbeta"}]
    assert PROFILE._prepare_calibration_rows(records, apply_chat_template=True) == records
