# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentTypeError, Namespace

import pytest

from gptqmodel import BACKEND
from scripts.eval_qvq_checkpoint import _parse_backend
from scripts.validate_qvq_lifecycle import _yaqa_calibration_controls


def _args(**overrides) -> Namespace:
    values = {
        "dataset": "base",
        "dataset_config": "LLM",
        "rounding": "yaqa",
        "yaqa_rows": None,
        "yaqa_row_start": 0,
        "yaqa_dataset": None,
        "yaqa_dataset_config": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_yaqa_calibration_controls_reuse_base_stream_when_unspecified():
    assert _yaqa_calibration_controls(_args()) is None


def test_yaqa_calibration_controls_resolve_independent_exact_slice():
    controls = _yaqa_calibration_controls(_args(yaqa_rows=1024, yaqa_row_start=128))

    assert controls == ("base", "LLM", 128, 1024)


def test_yaqa_calibration_controls_allow_independent_dataset():
    controls = _yaqa_calibration_controls(
        _args(yaqa_rows=1024, yaqa_dataset="yaqa", yaqa_dataset_config="Fisher")
    )

    assert controls == ("yaqa", "Fisher", 0, 1024)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"yaqa_row_start": 1}, "require --yaqa-rows"),
        ({"yaqa_rows": 0}, "must be positive"),
        ({"yaqa_rows": 1, "yaqa_row_start": -1}, "must be nonnegative"),
        ({"yaqa_rows": 1, "rounding": "block_ldlq"}, "requires --rounding yaqa"),
    ),
)
def test_yaqa_calibration_controls_reject_invalid_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _yaqa_calibration_controls(_args(**overrides))


@pytest.mark.parametrize("backend", (BACKEND.QVQ, BACKEND.EXL3_EXLLAMA_V3))
def test_quantized_checkpoint_evaluator_accepts_native_low_bit_backends(backend):
    assert _parse_backend(backend.value) == backend


def test_quantized_checkpoint_evaluator_rejects_unrelated_backend():
    with pytest.raises(ArgumentTypeError, match="Unsupported checkpoint backend"):
        _parse_backend(BACKEND.GPTQ_TORCH.value)
