# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.quantization import FORMAT
from scripts.qvq_evaluate import (
    TASKS as QVQ_EVALUATION_TASKS,
)
from scripts.qvq_evaluate import (
    _encode_prompt,
    _greedy_rollout,
    _mmlu_question_row_progress,
    _model_logits,
    _publish_snapshot_evaluation,
    _row_text,
    _resolve_cuda_graph_request,
    _wilson_interval,
    validate_evaluation_is_held_out,
)
from scripts.qvq_evaluate import build_parser as build_evaluate_parser
from scripts.qvq_quantize import (
    DatasetSlice,
    _automatic_bank_count,
    aggregate_qvq_process_telemetry,
    build_quantize_config,
    dataset_slice_evidence,
    validate_disjoint_slices,
)
from scripts.qvq_quantize import (
    build_parser as build_quantize_parser,
)


def test_qvq_quantize_parser_builds_nested_yaqa_configuration():
    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dense-model",
            "--output",
            "quantized-model",
            "--calibration-dataset",
            "dataset",
            "--format",
            FORMAT.QVQ_V2B2_P32.value,
            "--yaqa-chat-template-weighting",
        ]
    )

    config = build_quantize_config(args)

    assert config.format == FORMAT.QVQ_V2B2_P32
    assert config.bank_count == 2
    assert config.rounding == "yaqa"
    assert config.yaqa.batch_size == 8
    assert config.yaqa.chat_template.enabled is True
    assert config.yaqa.chat_template.content_weight == 0.97
    assert config.offload_to_disk is False
    assert args.qvq_telemetry is True


def test_qvq_quantize_parser_can_disable_nested_telemetry():
    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dense-model",
            "--output",
            "quantized-model",
            "--calibration-dataset",
            "dataset",
            "--no-qvq-telemetry",
        ]
    )

    assert args.qvq_telemetry is False


def test_qvq_quantize_parser_exposes_fail_closed_disjointness_gate():
    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dense-model",
            "--output",
            "quantized-model",
            "--calibration-dataset",
            "dataset",
            "--require-disjointness",
        ]
    )

    assert args.require_disjointness is True


def test_qvq_quantize_json_requires_explicit_rounding(tmp_path):
    config_path = tmp_path / "legacy.json"
    config_path.write_text('{"bits": 2, "format": "qvq"}', encoding="utf-8")
    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dense-model",
            "--output",
            "quantized-model",
            "--calibration-dataset",
            "dataset",
            "--quant-config",
            str(config_path),
        ]
    )

    with pytest.raises(ValueError, match="explicit.*rounding"):
        build_quantize_config(args)


def test_qvq_quantize_aggregates_nested_telemetry_by_shape_and_module():
    quant_log = {
        "qvq": [
            {
                "full_name": "model.layers.0.self_attn.q_proj",
                "time": "2.5",
                "qvq_telemetry": {
                    "phases": {
                        "yaqa_segmented_viterbi": {
                            "calls": 2,
                            "host_dispatch_ms": 2000.0,
                            "gpu_ms": 1900.0,
                        }
                    },
                    "counters": {"input_features": 16, "output_features": 32, "yaqa_tiles": 2},
                    "viterbi_pruning": {
                        "baseline_candidates_possible": 100,
                        "candidates_evaluated": 40,
                    },
                },
            },
            {
                "full_name": "model.layers.1.self_attn.q_proj",
                "time": "3.5",
                "qvq_telemetry": {
                    "phases": {
                        "yaqa_segmented_viterbi": {
                            "calls": 3,
                            "host_dispatch_ms": 3000.0,
                            "gpu_ms": 2800.0,
                        }
                    },
                    "counters": {"input_features": 16, "output_features": 32, "yaqa_tiles": 4},
                    "viterbi_pruning": {
                        "baseline_candidates_possible": 250,
                        "candidates_evaluated": 90,
                    },
                },
            },
        ]
    }

    telemetry = aggregate_qvq_process_telemetry(quant_log)

    assert telemetry is not None
    assert telemetry["process_quant_seconds"] == 6.0
    assert telemetry["phases"]["yaqa_segmented_viterbi"] == {
        "calls": 5,
        "host_dispatch_ms": 5000.0,
        "gpu_ms": 4700.0,
    }
    assert telemetry["counters"]["yaqa_tiles"] == 6
    assert telemetry["viterbi_pruning"] == {
        "baseline_candidates_possible": 250,
        "candidates_evaluated": 90,
    }
    assert telemetry["modules"][0]["viterbi_pruning"]["baseline_candidates_possible"] == 100
    assert telemetry["shapes"]["32x16"]["modules"] == 2
    assert telemetry["shapes"]["32x16"]["process_quant_seconds"] == 6.0
    assert len(telemetry["modules"]) == 2


@pytest.mark.parametrize(
    ("format_value", "expected"),
    [
        (FORMAT.QVQ.value, 1),
        (FORMAT.QVQ_V2B2_P32.value, 2),
        (FORMAT.QVQ_V2B4_P64.value, 4),
        (FORMAT.QVQ_V4.value, 4),
    ],
)
def test_qvq_quantize_automatic_bank_count(format_value, expected):
    assert _automatic_bank_count(format_value) == expected


def test_qvq_quantize_rejects_overlapping_preparation_slices():
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_slices(
            {
                "calibration": DatasetSlice("dataset", "config", "train", 0, 512),
                "yaqa": DatasetSlice("dataset", "config", "train", 500, 512),
            }
        )


def test_qvq_quantize_accepts_disjoint_and_different_dataset_slices():
    validate_disjoint_slices(
        {
            "calibration": DatasetSlice("dataset", "config", "train", 0, 512),
            "yaqa": DatasetSlice("dataset", "config", "train", 512, 512),
            "replay": DatasetSlice("other-dataset", None, "train", 0, 512),
        }
    )


def test_qvq_quantize_binds_local_dataset_and_manifest_hashes(tmp_path):
    source = tmp_path / "calibration.jsonl"
    manifest = tmp_path / "calibration.manifest.json"
    source.write_text('{"content":"sample"}\n', encoding="utf-8")
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")

    evidence = dataset_slice_evidence(DatasetSlice(str(source), None, "train", 0, 1))

    assert evidence["source"] == str(source.resolve())
    assert len(evidence["content_sha256"]) == 64
    assert evidence["identity_manifest"] == str(manifest.resolve())
    assert len(evidence["identity_manifest_sha256"]) == 64


def test_qvq_evaluate_rejects_overlap_with_recorded_preparation(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "qvq_quantize_run.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "calibration": {
                        "source": "dataset",
                        "config": "config",
                        "split": "train",
                        "row_start": 0,
                        "rows": 512,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps recorded"):
        validate_evaluation_is_held_out(
            checkpoint,
            DatasetSlice("dataset", "config", "train", 511, 512),
            allow_overlap=False,
        )

    validate_evaluation_is_held_out(
        checkpoint,
        DatasetSlice("dataset", "config", "train", 512, 512),
        allow_overlap=False,
    )


def test_qvq_evaluate_parser_keeps_quantization_out_of_evaluation():
    args = build_evaluate_parser().parse_args(
        [
            "diagnostics",
            "--dense-model",
            "dense-model",
            "--checkpoint",
            "quantized-model",
            "--dataset",
            "dataset",
            "--row-start",
            "1536",
            "--output",
            "result.json",
        ]
    )

    assert args.command == "diagnostics"
    assert args.rows == 512
    assert not hasattr(args, "bits")


def test_qvq_evaluate_parser_has_canonical_divergence300_contract():
    args = build_evaluate_parser().parse_args(
        [
            "divergence300",
            "--dense-model",
            "dense-model",
            "--checkpoint",
            "quantized-model",
            "--dataset",
            "locked.jsonl",
            "--output",
            "result.json",
        ]
    )

    assert args.command == "divergence300"
    assert args.max_prompt_tokens == 16384
    assert args.dtype == "float16"
    assert args.attn_implementation == "sdpa"


def test_qvq_evaluate_parser_exposes_prefix_cache_prewarm():
    args = build_evaluate_parser().parse_args(
        [
            "tasks",
            "--checkpoint",
            "quantized-model",
            "--output",
            "result.json",
            "--loglikelihood-prefix-cache",
            "--loglikelihood-prefix-cache-prewarm",
            "--loglikelihood-prefix-cache-prewarm-batch-size",
            "32",
            "--loglikelihood-prefix-cache-release-after-group",
        ]
    )

    assert args.loglikelihood_prefix_cache is True
    assert args.loglikelihood_prefix_cache_prewarm is True
    assert args.loglikelihood_prefix_cache_prewarm_batch_size == 32
    assert args.loglikelihood_prefix_cache_release_after_group is True


def test_qvq_evaluate_exposes_full_mmlu_humanities_category():
    assert QVQ_EVALUATION_TASKS["mmlu_humanities"] == (
        "mmlu",
        False,
        {"subsets": "humanities"},
    )


def test_qvq_evaluate_forces_incremental_evalution_progress():
    assert os.environ["LOGBAR_FORCE_PROGRESS"] == "1"


def test_qvq_evaluate_tasks_require_paged_continuous_batching_defaults():
    args = build_evaluate_parser().parse_args(
        [
            "tasks",
            "--checkpoint",
            "quantized-model",
            "--output",
            "result.json",
            "--task",
            "gsm8k_platinum_cot",
        ]
    )

    assert args.device == "cuda:0"
    assert args.attn_implementation == "paged|flash_attention_2"
    assert args.batch_size == 64
    assert args.use_cuda_graph is None
    assert args.cuda_graph_mode is None
    assert _resolve_cuda_graph_request(args) == ((False, True), "decode")
    assert args.resume is False


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("off", ((False, False), "off")),
        ("varlen", ((True, False), "varlen")),
        ("decode", ((False, True), "decode")),
        ("both", ((True, True), "both")),
        ("auto", (None, "auto")),
    ],
)
def test_qvq_evaluate_resolves_transformers_cuda_graph_modes(option, expected):
    args = build_evaluate_parser().parse_args(
        [
            "tasks",
            "--checkpoint",
            "quantized-model",
            "--output",
            "result.json",
            "--cuda-graph-mode",
            option,
        ]
    )
    assert _resolve_cuda_graph_request(args) == expected


def test_qvq_evaluate_legacy_cuda_graph_flag_maps_to_both_paths():
    args = build_evaluate_parser().parse_args(
        [
            "tasks",
            "--checkpoint",
            "quantized-model",
            "--output",
            "result.json",
            "--use-cuda-graph",
        ]
    )
    assert _resolve_cuda_graph_request(args) == ((True, True), "both")


def test_qvq_evaluate_rejects_conflicting_cuda_graph_flags():
    args = build_evaluate_parser().parse_args(
        [
            "tasks",
            "--checkpoint",
            "quantized-model",
            "--output",
            "result.json",
            "--use-cuda-graph",
            "--cuda-graph-mode",
            "decode",
        ]
    )
    with pytest.raises(ValueError, match="only one"):
        _resolve_cuda_graph_request(args)


def test_qvq_evaluate_reports_mmlu_choice_work_as_completed_rows(monkeypatch):
    mmlu_module = importlib.import_module("evalution.benchmarks.mmlu")

    class FakeProgress:
        def __init__(self):
            self.next_calls = 0
            self.draw_calls = 0

        def next(self):
            self.next_calls += 1
            return self

        def draw(self):
            self.draw_calls += 1
            return self

    captured = {}

    def fake_manual_progress(total, *, title, subtitle):
        captured.update(total=total, title=title, subtitle=subtitle, progress=FakeProgress())
        return captured["progress"]

    monkeypatch.setattr(mmlu_module, "manual_progress", fake_manual_progress)
    with _mmlu_question_row_progress(True):
        row_progress = mmlu_module.manual_progress(
            12,
            title="mmlu_stem: scoring answer choices",
            subtitle="batch_size=16",
        )
        for _ in range(7):
            row_progress.next().draw()

    assert captured["total"] == 3
    assert captured["title"] == "mmlu_stem: completed question rows"
    assert captured["subtitle"] == "batch_size=16 choices_per_row=4"
    assert captured["progress"].next_calls == 1
    assert captured["progress"].draw_calls == 1


class _BareDecoder(nn.Module):
    def forward(self, **_kwargs):
        return SimpleNamespace(last_hidden_state=torch.zeros(1, 2, 3))


class _CausalLMWrapper(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.model = _BareDecoder()
        self.output = output

    def forward(self, **_kwargs):
        return self.output


@pytest.mark.parametrize(
    "output",
    [
        SimpleNamespace(logits=torch.ones(1, 2, 3)),
        {"logits": torch.ones(1, 2, 3)},
        (torch.ones(1, 2, 3),),
    ],
)
def test_qvq_evaluate_calls_public_causal_lm_and_extracts_logits(output):
    logits = _model_logits(_CausalLMWrapper(output), {"input_ids": torch.ones(1, 2, dtype=torch.long)})

    torch.testing.assert_close(logits, torch.ones(1, 2, 3), rtol=0, atol=0)


def test_qvq_evaluate_rejects_output_without_logits():
    with pytest.raises(TypeError, match="SimpleNamespace"):
        _model_logits(
            _CausalLMWrapper(SimpleNamespace(last_hidden_state=torch.ones(1, 2, 3))),
            {"input_ids": torch.ones(1, 2, dtype=torch.long)},
        )


class _GreedyForwardRecorder:
    def __init__(self, continuation):
        self.continuation = list(continuation)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        step = len(self.calls) - 1
        logits = torch.full((1, kwargs["input_ids"].shape[1], 16), -100.0)
        logits[:, -1, self.continuation[step]] = 1.0
        # No cache is intentional: the rollout must remain correct for a
        # model/backend that declines to return past_key_values.
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_qvq_evaluate_greedy_rollout_uses_literal_fixed_horizon_argmax():
    model = _GreedyForwardRecorder([7, 8, 9])
    encoded = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }

    result = _greedy_rollout(model, encoded, token_count=3, pad_token_id=0)

    torch.testing.assert_close(result, torch.tensor([7, 8, 9]), rtol=0, atol=0)
    assert len(model.calls) == 3
    assert all("min_new_tokens" not in call for call in model.calls)
    assert all("max_new_tokens" not in call for call in model.calls)
    assert all(call.get("use_cache") is True for call in model.calls)


def test_qvq_evaluate_greedy_rollout_does_not_suppress_eos():
    # Token 0 is the model's EOS/PAD token in this synthetic setup.  It must
    # be retained at step one and the fixed horizon must continue afterward.
    model = _GreedyForwardRecorder([0, 8, 9])
    encoded = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }

    result = _greedy_rollout(model, encoded, token_count=3, pad_token_id=0)

    torch.testing.assert_close(result, torch.tensor([0, 8, 9]), rtol=0, atol=0)
    assert len(model.calls) == 3


def test_qvq_evaluate_snapshot_publication_is_append_only(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    first = {"metrics": {"top1": 0.1}}
    second = {"metrics": {"top1": 0.2}}

    _publish_snapshot_evaluation(checkpoint, tmp_path / "first.json", first, kind="d300")
    _publish_snapshot_evaluation(checkpoint, tmp_path / "second.json", second, kind="d300")

    legacy = checkpoint / "post_quant_eval_result_d300.json"
    assert json.loads(legacy.read_text(encoding="utf-8")) == first
    digest_files = sorted(checkpoint.glob("post_quant_eval_result_d300_*.json"))
    assert len(digest_files) == 2
    assert {json.loads(path.read_text(encoding="utf-8"))["metrics"]["top1"] for path in digest_files} == {0.1, 0.2}


def test_qvq_evaluate_chat_rows_end_with_generation_prompt():
    tokenizer = SimpleNamespace(apply_chat_template=lambda value, **kwargs: (value, kwargs))
    messages = [{"role": "user", "content": "hello"}]

    rendered_messages, kwargs = _row_text({"messages": messages}, tokenizer, None)

    assert rendered_messages == messages
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}


def test_qvq_evaluate_encodes_chat_prompt_once_with_left_truncation():
    class FakeTokenizer:
        truncation_side = "right"

        def apply_chat_template(self, messages, **kwargs):
            assert self.truncation_side == "left"
            assert messages == [{"role": "user", "content": "hello"}]
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
                "truncation": True,
                "max_length": 128,
            }
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    tokenizer = FakeTokenizer()
    encoded = _encode_prompt(
        {"messages": [{"role": "user", "content": "hello"}]},
        tokenizer,
        max_prompt_tokens=128,
    )

    torch.testing.assert_close(encoded["input_ids"], torch.tensor([[1, 2, 3]]), rtol=0, atol=0)
    assert tokenizer.truncation_side == "right"


def test_qvq_evaluate_wilson_interval_contains_observed_rate():
    low, high = _wilson_interval(246, 300)

    assert low < 0.82 < high
