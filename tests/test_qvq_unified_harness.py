# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from gptqmodel.quantization import FORMAT
from scripts.qvq_evaluate import build_parser as build_evaluate_parser
from scripts.qvq_evaluate import validate_evaluation_is_held_out
from scripts.qvq_quantize import (
    DatasetSlice,
    _automatic_bank_count,
    build_parser as build_quantize_parser,
    build_quantize_config,
    validate_disjoint_slices,
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
    assert config.offload_to_disk is False


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
