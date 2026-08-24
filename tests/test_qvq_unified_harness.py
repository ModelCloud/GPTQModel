# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.quantization import FORMAT
from scripts.qvq_evaluate import (
    _greedy_rollout,
    _model_logits,
    _row_text,
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


class _GenerateRecorder:
    def __init__(self, continuation):
        self.continuation = continuation
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.cat((kwargs["input_ids"], self.continuation), dim=1)


def test_qvq_evaluate_greedy_rollout_uses_independent_fixed_horizon_generation():
    model = _GenerateRecorder(torch.tensor([[7, 8, 9]]))
    encoded = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }

    result = _greedy_rollout(model, encoded, token_count=3, pad_token_id=0)

    torch.testing.assert_close(result, torch.tensor([7, 8, 9]), rtol=0, atol=0)
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["min_new_tokens"] == 3
    assert model.kwargs["max_new_tokens"] == 3
    assert model.kwargs["use_cache"] is True


def test_qvq_evaluate_chat_rows_end_with_generation_prompt():
    tokenizer = SimpleNamespace(apply_chat_template=lambda value, **kwargs: (value, kwargs))
    messages = [{"role": "user", "content": "hello"}]

    rendered_messages, kwargs = _row_text({"messages": messages}, tokenizer, None)

    assert rendered_messages == messages
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}
