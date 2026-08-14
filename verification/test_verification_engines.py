# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import json

import numpy as np
import pytest

from verification import engine_common
from verification import compare_logits
from verification.compare_logits import (
    _dense_from_sglang_top_logprobs,
    _dense_from_vllm_position,
    _even_prefix_lengths,
    _validate_reused_native_artifact,
    _role_tensor_parallel_sizes,
    compare_log_prob_matrices,
)
from verification.evalution_score import (
    EvalutionTimingTracker,
    MMLU_HISTORY_SUBSETS,
    _evalution_process_environment,
    build_evalution_spec,
    build_parser,
    extract_score_summary,
    summarize_gpu_usage,
)


def _idle_gate_gpu(utilization: int) -> dict[str, object]:
    return {
        "index": 0,
        "pci.bus_id": "00000000:01:00.0",
        "uuid": "GPU-test",
        "name": "Example GPU",
        "driver_version": "999.1",
        "compute_cap": "9.0",
        "memory.total": 81920,
        "memory.used": 0,
        "utilization.gpu": utilization,
    }


def test_engine_transition_waits_for_strict_idle_after_transient_utilization(monkeypatch):
    observed_utilization = iter((100, 0, 0))
    monkeypatch.setattr(
        engine_common,
        "query_gpu",
        lambda _target: _idle_gate_gpu(next(observed_utilization)),
    )
    monkeypatch.setattr(engine_common, "query_compute_processes", lambda _target: [])
    monkeypatch.setattr(engine_common.time, "sleep", lambda _seconds: None)

    result = engine_common.wait_for_strict_idle_gate(
        ["GPU-test"],
        timeout_seconds=10,
        poll_interval_seconds=0,
        sample_count=2,
        interval_seconds=0,
    )

    assert len(result["failed_attempts"]) == 1
    assert result["failed_attempts"][0]["sample"]["utilization_gpu_percent"] == 100
    assert [sample["utilization_gpu_percent"] for sample in result["strict_idle_preflight"]] == [0, 0]


def test_direct_strict_idle_gate_remains_fail_fast(monkeypatch):
    monkeypatch.setattr(engine_common, "query_gpu", lambda _target: _idle_gate_gpu(100))
    monkeypatch.setattr(engine_common, "query_compute_processes", lambda _target: [])

    with pytest.raises(engine_common.StrictIdleGateError, match="failed strict idle gate"):
        engine_common.strict_idle_gate(["GPU-test"], sample_count=3, interval_seconds=0)


def test_engine_transition_rejects_a_gpu_that_stays_busy(monkeypatch):
    monkeypatch.setattr(engine_common, "query_gpu", lambda _target: _idle_gate_gpu(100))
    monkeypatch.setattr(engine_common, "query_compute_processes", lambda _target: [])

    with pytest.raises(RuntimeError, match="strict-idle timeout"):
        engine_common.wait_for_strict_idle_gate(
            ["GPU-test"],
            timeout_seconds=0,
            poll_interval_seconds=0,
            sample_count=3,
            interval_seconds=0,
        )


def test_worker_admission_uses_bounded_strict_idle_wait(monkeypatch, tmp_path):
    sequences = [[1, 2]]
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "sequences": sequences,
                "sequence_sha256": engine_common.sha256_json(sequences),
            }
        )
    )
    artifact_path = tmp_path / "worker.npz"
    config_path = tmp_path / "worker.json"
    config_path.write_text(
        json.dumps(
            {
                "engine": "vllm",
                "role": "quantized",
                "model": "quant/model",
                "revision": None,
                "tokenizer": "dense/model",
                "tokenizer_revision": None,
                "quantization": None,
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "vocab_size": 3,
                "inputs_path": str(inputs_path),
                "artifact_path": str(artifact_path),
                "idle_samples": 3,
                "idle_interval_seconds": 0,
                "idle_memory_tolerance_mib": 700,
                "idle_wait_timeout_seconds": 120,
                "idle_wait_poll_seconds": 1,
            }
        )
    )
    idle_result = {
        "failed_attempts": [{"sample": {"utilization_gpu_percent": 3}}],
        "strict_idle_preflight": [{"utilization_gpu_percent": 0}],
    }
    observed = {}

    monkeypatch.setattr(engine_common, "visible_gpu_targets", lambda _tp: ["GPU-test"])

    def wait_for_idle(targets, **kwargs):
        observed["targets"] = targets
        observed["kwargs"] = kwargs
        return idle_result

    monkeypatch.setattr(engine_common, "wait_for_strict_idle_gate", wait_for_idle)
    monkeypatch.setattr(
        compare_logits,
        "_run_vllm_worker",
        lambda _config, _sequences, _targets: (
            np.asarray([[-1.0, -2.0, -3.0]], dtype=np.float32),
            {},
        ),
    )

    compare_logits.worker_main(["--config", str(config_path)])

    assert observed["targets"] == ["GPU-test"]
    assert observed["kwargs"]["timeout_seconds"] == 120
    with np.load(artifact_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["strict_idle_preflight_wait"] == idle_result
    assert metadata["strict_idle_preflight"] == idle_result["strict_idle_preflight"]


def test_reused_native_artifact_requires_matching_inputs_and_runtime_identity(tmp_path):
    artifact = tmp_path / "native.npz"
    metadata = {
        "role": "native",
        "engine": "vllm",
        "model": "dense/model",
        "revision": None,
        "tokenizer": "dense/model",
        "tokenizer_revision": None,
        "quantization": None,
        "dtype": "bfloat16",
        "tensor_parallel_size": 8,
        "input_sequence_sha256": "expected-inputs",
        "sample_count": 2,
        "vocab_size": 3,
        "matrix_dtype": "float32",
        "runtime_exclusivity": {"foreign_compute_processes": []},
    }
    np.savez(
        artifact,
        log_probs=np.asarray([[-1.0, -2.0, -3.0], [-3.0, -2.0, -1.0]], dtype=np.float32),
        metadata_json=json.dumps(metadata),
    )
    args = SimpleNamespace(
        engine="vllm",
        native_model="dense/model",
        native_revision=None,
        tokenizer=None,
        tokenizer_revision=None,
        native_quantization=None,
        dtype="bfloat16",
        tensor_parallel_size=8,
    )
    inputs = {"sequence_sha256": "expected-inputs", "sequences": [[1], [2]]}

    matrix, observed, provenance = _validate_reused_native_artifact(args, artifact, inputs, 3)

    assert matrix.shape == (2, 3)
    assert observed == metadata
    assert len(provenance["source_sha256"]) == 64

    inputs["sequence_sha256"] = "different-inputs"
    with pytest.raises(ValueError, match="input_sequence_sha256"):
        _validate_reused_native_artifact(args, artifact, inputs, 3)


def test_role_specific_tensor_parallel_sizes_fall_back_independently():
    assert _role_tensor_parallel_sizes(
        SimpleNamespace(
            tensor_parallel_size=2,
            native_tensor_parallel_size=8,
            quantized_tensor_parallel_size=None,
        )
    ) == (8, 2)


def _evalution_args(engine: str, *extra: str):
    return build_parser().parse_args(
        [
            "--engine",
            engine,
            "--model",
            "org/model",
            "--output",
            "result.json",
            *extra,
        ]
    )


def test_evalution_vllm_spec_contains_one_independently_started_suite():
    spec = build_evalution_spec(
        _evalution_args(
            "vllm",
            "--task",
            "arc_challenge",
            "--batch-size",
            "7",
            "--tensor-parallel-size",
            "2",
            "--max-rows",
            "3",
        )
    )

    assert spec["engine"]["type"] == "VLLM"
    assert spec["engine"]["tensor_parallel_size"] == 2
    assert spec["model"]["path"] == "org/model"
    assert [test["type"] for test in spec["tests"]] == ["arc_challenge"]
    assert all(test["batch_size"] == 7 for test in spec["tests"])
    assert all(test["max_rows"] == 3 for test in spec["tests"])


def test_evalution_spec_can_select_one_independently_timed_task():
    spec = build_evalution_spec(
        _evalution_args(
            "vllm",
            "--task",
            "gsm8k_platinum_cot",
            "--batch-size",
            "8",
        )
    )

    assert spec["engine"]["batch_size"] == 8
    assert spec["tests"] == [
        {
            "type": "gsm8k_platinum",
            "variant": "cot",
            "apply_chat_template": True,
            "max_new_tokens": 256,
            "batch_size": 8,
            "stream": True,
        }
    ]


def test_evalution_spec_rejects_more_than_one_task_per_engine_startup():
    args = _evalution_args(
        "vllm",
        "--task",
        "arc_challenge",
        "--task",
        "arc_challenge",
    )

    with pytest.raises(ValueError, match="exactly one --task"):
        build_evalution_spec(args)


def test_evalution_sglang_spec_uses_the_validated_deterministic_defaults():
    spec = build_evalution_spec(
        _evalution_args("sglang", "--task", "mmlu_history")
    )

    assert spec["engine"]["type"] == "SGLang"
    assert spec["engine"]["attention_backend"] == "flashinfer"
    assert spec["engine"]["tp_size"] == 1
    assert spec["model"]["model_kwargs"] == {
        "enable_deterministic_inference": True,
        "disable_cuda_graph": True,
        "prefill_attention_backend": "triton",
    }


def test_evalution_sglang_can_disable_fused_wqa_wkv_for_child_only(monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_FUSE_WQA_WKV", "1")
    args = _evalution_args(
        "sglang",
        "--task",
        "arc_challenge",
        "--sglang-disable-fused-wqa-wkv",
    )

    environment, overrides = _evalution_process_environment(args)

    assert environment["SGLANG_OPT_FUSE_WQA_WKV"] == "0"
    assert overrides == {"SGLANG_OPT_FUSE_WQA_WKV": "0"}


def _synthetic_evalution_test(name, metric, *, subset=None):
    metadata = {} if subset is None else {"subset": subset}
    sample = {"scores": {metric: 1.0}, "metadata": metadata}
    if metric == "acc,exam":
        sample = _synthetic_arc_exam_sample(gold_index=0, selected_indices=(0,))
    elif metric == "acc,ll":
        metadata["choice_logprobs"] = [-1.0, -2.0]
    return {
        "name": name,
        "metrics": {metric: 1.0},
        "samples": [sample],
    }


def _synthetic_arc_exam_sample(*, gold_index, selected_indices):
    selected_indices = tuple(selected_indices)
    score = 1.0 / len(selected_indices) if gold_index in selected_indices else 0.0
    return {
        "scores": {"acc,exam": score},
        "extracted": {
            "gold_index": str(gold_index),
            "selected_indices": ",".join(str(index) for index in selected_indices),
        },
        "metadata": {
            "choice_logprobs": [-1.0, -2.0, -3.0, -4.0],
            "selected_count": len(selected_indices),
        },
    }


def test_evalution_score_summary_accepts_arc_exam_tie_aware_partial_credit():
    samples = [
        _synthetic_arc_exam_sample(gold_index=0, selected_indices=(0,)),
        _synthetic_arc_exam_sample(gold_index=3, selected_indices=(0, 3)),
        _synthetic_arc_exam_sample(gold_index=3, selected_indices=(0, 1)),
    ]
    payload = {
        "tests": [
            {
                "name": "arc_challenge",
                "metrics": {"acc,exam": 0.5},
                "samples": samples,
            }
        ]
    }

    scores = extract_score_summary(
        payload,
        require_full_coverage=False,
        required_tasks=("arc_challenge",),
    )

    assert scores["arc_challenge"]["score"] == 0.5
    assert scores["arc_challenge"]["score_sum"] == 1.5
    assert scores["arc_challenge"]["correct"] == 1
    assert scores["arc_challenge"]["partial_credit_samples"] == 1


def test_evalution_score_summary_rejects_invalid_arc_exam_partial_credit():
    sample = _synthetic_arc_exam_sample(gold_index=3, selected_indices=(0, 3))
    sample["scores"]["acc,exam"] = 0.25
    payload = {
        "tests": [
            {
                "name": "arc_challenge",
                "metrics": {"acc,exam": 0.25},
                "samples": [sample],
            }
        ]
    }

    with pytest.raises(ValueError, match="invalid tie-aware"):
        extract_score_summary(
            payload,
            require_full_coverage=False,
            required_tasks=("arc_challenge",),
        )


def test_evalution_score_summary_maps_dynamic_history_name_and_recomputes_scores():
    history_name = "mmlu_" + "__".join(
        subset.replace(".", "_") for subset in MMLU_HISTORY_SUBSETS
    )
    payload = {
        "tests": [
            _synthetic_evalution_test("arc_challenge", "acc,exam"),
            _synthetic_evalution_test("gsm8k_platinum_cot", "acc,num"),
            _synthetic_evalution_test(
                "mmlu_stem", "acc,ll", subset="stem.abstract_algebra"
            ),
            _synthetic_evalution_test(
                history_name,
                "acc,ll",
                subset=MMLU_HISTORY_SUBSETS[0],
            ),
        ]
    }

    scores = extract_score_summary(payload, require_full_coverage=False)

    assert list(scores) == [
        "arc_challenge",
        "gsm8k_platinum_cot",
        "mmlu_stem",
        "mmlu_history",
    ]
    assert scores["arc_challenge"]["correct"] == 1
    assert scores["mmlu_history"]["score"] == 1.0


def test_evalution_score_summary_rejects_an_inconsistent_aggregate():
    history_name = "mmlu_" + "__".join(
        subset.replace(".", "_") for subset in MMLU_HISTORY_SUBSETS
    )
    payload = {
        "tests": [
            _synthetic_evalution_test("arc_challenge", "acc,exam"),
            _synthetic_evalution_test("gsm8k_platinum_cot", "acc,num"),
            _synthetic_evalution_test("mmlu_stem", "acc,ll"),
            _synthetic_evalution_test(history_name, "acc,ll"),
        ]
    }
    payload["tests"][0]["metrics"]["acc,exam"] = 0.0

    with pytest.raises(ValueError, match="recompute"):
        extract_score_summary(payload, require_full_coverage=False)


@pytest.mark.parametrize("invalid_value", [None, float("nan"), float("inf")])
def test_evalution_score_summary_rejects_nonfinite_choice_logprobs(invalid_value):
    payload = {
        "tests": [
            {
                "name": "arc_challenge",
                "metrics": {"acc,exam": 0.0},
                "samples": [
                    {
                        "extracted": {"gold_index": "0", "selected_indices": "1"},
                        "scores": {"acc,exam": 0.0},
                        "metadata": {
                            "choice_logprobs": [invalid_value] * 4,
                            "selected_count": 1,
                        },
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="non-finite choice_logprobs"):
        extract_score_summary(
            payload,
            require_full_coverage=False,
            required_tasks=("arc_challenge",),
        )


def test_evalution_score_summary_accepts_one_requested_task():
    payload = {
        "tests": [
            _synthetic_evalution_test("gsm8k_platinum_cot", "acc,num"),
        ]
    }

    scores = extract_score_summary(
        payload,
        require_full_coverage=False,
        required_tasks=("gsm8k_platinum_cot",),
    )

    assert list(scores) == ["gsm8k_platinum_cot"]
    assert scores["gsm8k_platinum_cot"]["correct"] == 1


def test_evalution_timing_tracker_records_startup_and_each_task_duration():
    tracker = EvalutionTimingTracker(
        process_started_monotonic=100.0,
        process_started_at_utc="2026-08-07T10:00:00+00:00",
    )
    events = (
        ("running test suite ARCChallenge", 110.0, "2026-08-07T10:00:10+00:00"),
        ("completed test arc_challenge", 112.0, "2026-08-07T10:00:12+00:00"),
        ("running test suite GSM8KPlatinum", 115.0, "2026-08-07T10:00:15+00:00"),
        ("completed test gsm8k_platinum_cot", 120.0, "2026-08-07T10:00:20+00:00"),
        ("running test suite MMLU", 122.0, "2026-08-07T10:00:22+00:00"),
        ("completed test mmlu_stem", 130.0, "2026-08-07T10:00:30+00:00"),
        ("running test suite MMLU", 131.0, "2026-08-07T10:00:31+00:00"),
        (
            "completed test mmlu_"
            "humanities_high_school_european_history__"
            "humanities_high_school_us_history__"
            "humanities_high_school_world_history__humanities_prehistory",
            140.0,
            "2026-08-07T10:00:40+00:00",
        ),
    )
    for line, monotonic, utc in events:
        tracker.observe_line(line, now_monotonic=monotonic, now_utc=utc)

    timing = tracker.finish(
        return_code=0,
        now_monotonic=142.0,
        now_utc="2026-08-07T10:00:42+00:00",
    )

    assert timing["complete"] is True
    assert timing["engine_startup_seconds"] == 10.0
    assert timing["evaluation_wall_time_seconds"] == 42.0
    assert {
        name: record["duration_seconds"] for name, record in timing["tasks"].items()
    } == {
        "arc_challenge": 2.0,
        "gsm8k_platinum_cot": 5.0,
        "mmlu_stem": 8.0,
        "mmlu_history": 9.0,
    }


def test_evalution_timing_tracker_completes_one_selected_task():
    tracker = EvalutionTimingTracker(
        process_started_monotonic=100.0,
        process_started_at_utc="2026-08-07T10:00:00+00:00",
        task_order=("mmlu_history",),
    )
    tracker.observe_line(
        "running test suite MMLU",
        now_monotonic=110.0,
        now_utc="2026-08-07T10:00:10+00:00",
    )
    tracker.observe_line(
        "completed test mmlu_"
        "humanities_high_school_european_history__"
        "humanities_high_school_us_history__"
        "humanities_high_school_world_history__humanities_prehistory",
        now_monotonic=120.0,
        now_utc="2026-08-07T10:00:20+00:00",
    )

    timing = tracker.finish(
        return_code=0,
        now_monotonic=121.0,
        now_utc="2026-08-07T10:00:21+00:00",
    )

    assert timing["complete"] is True
    assert timing["engine_startup_seconds"] == 10.0
    assert timing["tasks"] == {
        "mmlu_history": {
            "state": "completed",
            "evalution_suite_type": "MMLU",
            "evalution_name": (
                "mmlu_humanities_high_school_european_history__"
                "humanities_high_school_us_history__"
                "humanities_high_school_world_history__humanities_prehistory"
            ),
            "started_at_utc": "2026-08-07T10:00:10+00:00",
            "completed_at_utc": "2026-08-07T10:00:20+00:00",
            "duration_seconds": 10.0,
        }
    }


def test_evalution_gpu_summary_records_model_once_per_gpu():
    base = {
        "physical_id": 3,
        "pci_bus_id": "00000000:04:00.0",
        "uuid": "GPU-test",
        "name": "Example GPU",
        "driver_version": "999.1",
        "compute_capability": "9.0",
        "memory_total_mib": 81920,
    }
    preflight = [
        {**base, "sample": 1, "memory_used_mib": 0},
        {**base, "sample": 2, "memory_used_mib": 0},
    ]

    assert summarize_gpu_usage(preflight) == [
        {
            "physical_id": 3,
            "pci_bus_id": "00000000:04:00.0",
            "uuid": "GPU-test",
            "model": "Example GPU",
            "driver_version": "999.1",
            "compute_capability": "9.0",
            "memory_total_mib": 81920,
        }
    ]


def test_full_vocabulary_engine_output_parsers_preserve_token_id_order():
    vllm_position = {
        2: SimpleNamespace(logprob=-3.0),
        0: SimpleNamespace(logprob=-1.0),
        1: SimpleNamespace(logprob=-2.0),
    }
    sglang_position = [[-2.0, 1, None], [-3.0, 2, None], [-1.0, 0, None]]

    assert np.array_equal(
        _dense_from_vllm_position(vllm_position, 3, np),
        np.asarray([-1.0, -2.0, -3.0], dtype=np.float32),
    )
    assert np.array_equal(
        _dense_from_sglang_top_logprobs(sglang_position, 3, np),
        np.asarray([-1.0, -2.0, -3.0], dtype=np.float32),
    )


def test_full_vocabulary_parser_rejects_a_partial_distribution():
    with pytest.raises(RuntimeError, match="omitted 1/3"):
        _dense_from_sglang_top_logprobs([[-1.0, 0, None], [-2.0, 2, None]], 3, np)


def test_sglang_float32_min_sentinel_is_treated_as_probability_zero():
    sentinel = float(np.finfo(np.float32).min)

    dense = _dense_from_sglang_top_logprobs(
        [[-1.0, 0, None], [sentinel, 1, None]], 2, np
    )

    assert dense[0] == pytest.approx(-1.0)
    assert np.isneginf(dense[1])


def test_distribution_comparison_reports_exact_native_to_quantized_kld():
    native = np.log(np.asarray([[0.75, 0.25]], dtype=np.float64))
    quantized = np.log(np.asarray([[0.50, 0.50]], dtype=np.float64))
    positions = [
        {"sample_index": 0, "prompt_index": 0, "prefix_length": 3, "target_token_id": 1}
    ]

    result = compare_log_prob_matrices(native, quantized, positions, top_k=2)

    expected = 0.75 * np.log(0.75 / 0.50) + 0.25 * np.log(0.25 / 0.50)
    assert result["summary"]["kld"]["mean"] == pytest.approx(expected)
    assert result["summary"]["top1_agreement"] == 1.0
    assert result["summary"]["observed_next_token_nll"]["mean_delta"] == pytest.approx(
        -np.log(0.50) + np.log(0.25)
    )


def test_centered_logits_ignore_an_unidentifiable_additive_shift():
    native = np.asarray([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]], dtype=np.float64)
    quantized = native + np.asarray([[17.0], [-9.0]])
    positions = [
        {
            "sample_index": 0,
            "prompt_index": 0,
            "prefix_length": 2,
            "target_token_id": None,
        },
        {
            "sample_index": 1,
            "prompt_index": 1,
            "prefix_length": 4,
            "target_token_id": None,
        },
    ]

    result = compare_log_prob_matrices(native, quantized, positions, top_k=2)

    assert result["summary"]["kld"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert result["summary"]["centered_logit_rmse"]["max"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result["summary"]["top1_agreement"] == 1.0


def test_centered_logits_exclude_sglang_probability_zero_sentinels():
    sentinel = np.finfo(np.float32).min
    native = np.asarray([[-1.0, -2.0, sentinel]], dtype=np.float32)
    quantized = np.asarray([[-1.0, -3.0, sentinel]], dtype=np.float32)
    positions = [
        {
            "sample_index": 0,
            "prompt_index": 0,
            "prefix_length": 2,
            "target_token_id": None,
        }
    ]

    result = compare_log_prob_matrices(native, quantized, positions, top_k=1)

    assert result["summary"]["centered_logit_rmse"]["mean"] == pytest.approx(0.5)
    assert result["per_position"][0]["common_finite_vocab_entries"] == 2


@pytest.mark.parametrize(
    ("token_count", "requested", "expected"),
    (
        (1, 4, [1]),
        (4, 1, [4]),
        (4, 4, [1, 2, 3, 4]),
        (10, 3, [1, 5, 10]),
    ),
)
def test_even_prefix_lengths_are_bounded_and_include_the_final_position(
    token_count, requested, expected
):
    assert _even_prefix_lengths(token_count, requested) == expected
