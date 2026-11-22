# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Calibration input-handling gaps after #4193/#4238 (issue #4311).

Covers five defects:

1. a columnar dict built from ``datasets`` ``Column`` objects was treated as a
   single row and failed tensor conversion;
2. ``calibration_dataset_concat_size`` packed pad tokens from padded rows that
   did not need trimming;
3. the concat tail padding raised ``TypeError`` when ``pad_token_id`` was None;
4. an all-zero additive/extended mask was read as "drop everything";
5. ``LoopProcessor`` counted columns, not rows, in its size warning.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from gptqmodel.looper import loop_processor as loop_processor_module
from gptqmodel.looper.loop_processor import (
    LoopProcessor,
    _calibration_dataset_length,
)
from gptqmodel.quantization.config import CalibrationConfig
from gptqmodel.utils.calibration import prepare_calibration_dataset


class _Recorder:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warn(self, message, *args):
        self.warnings.append(str(message) % args if args else str(message))

    def info(self, message, *args):
        self.infos.append(str(message) % args if args else str(message))


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = None
    padding_side = "right"
    truncation_side = "right"

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(token)) for token in ids)


def _qmodel(tokenizer=None, *, max_positions=128):
    return SimpleNamespace(
        tokenizer=tokenizer if tokenizer is not None else _Tokenizer(),
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=max_positions),
        ),
    )


def test_datasets_column_mapping_is_unpacked_into_rows():
    """Defect 1: ``datasets`` ``Column`` objects must be read as columns."""

    datasets = pytest.importorskip("datasets")
    tds = datasets.Dataset.from_dict(
        {
            "input_ids": [[11, 12, 13], [21, 22, 23]],
            "attention_mask": [[1, 1, 1], [1, 0, 1]],
        }
    )

    prepared = prepare_calibration_dataset(
        _qmodel(),
        {"input_ids": tds["input_ids"], "attention_mask": tds["attention_mask"]},
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert [batch["input_ids"].tolist() for batch in prepared] == [
        [[11, 12, 13]],
        [[21, 22, 23]],
    ]
    assert [batch["attention_mask"].tolist() for batch in prepared] == [
        [[1, 1, 1]],
        [[1, 0, 1]],
    ]

    # A plain single-row dict still describes one row, not a column.
    single = prepare_calibration_dataset(
        _qmodel(),
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
        batch_size=1,
        calibration_data_min_length=1,
    )
    assert len(single) == 1
    assert single[0]["input_ids"].tolist() == [[7, 8, 9]]


def test_concat_path_strips_pad_tokens_before_packing():
    """Defect 2: padded rows must not leak pad ids into concat windows."""

    dataset = [
        {"input_ids": [[1, 2, 3, 4, 0, 0]], "attention_mask": [[1, 1, 1, 1, 0, 0]]},
        {"input_ids": [[5, 6, 7, 8, 9, 0]], "attention_mask": [[1, 1, 1, 1, 1, 0]]},
    ]

    prepared = prepare_calibration_dataset(
        _qmodel(),
        dataset,
        calibration_dataset_concat_size=8,
        batch_size=1,
        # 0, not 1: `min_length` now filters the packed windows, and this
        # dataset's tail window carries a single real token. The assertion
        # below is about pad ids staying out of the packed window, not about
        # the minimum-length filter (#4892 covers the tail-window rule).
        calibration_data_min_length=0,
    )

    assert prepared[0]["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert prepared[0]["attention_mask"].tolist() == [[1] * 8]
    # Only the final window's real tail is padded.
    assert prepared[1]["input_ids"].tolist() == [[9, 0, 0, 0, 0, 0, 0, 0]]
    assert prepared[1]["attention_mask"].tolist() == [[1, 0, 0, 0, 0, 0, 0, 0]]


class _NoPadTokenizer(_Tokenizer):
    pad_token_id = None
    eos_token_id = 99


class _NoPadNoEosTokenizer(_Tokenizer):
    pad_token_id = None
    eos_token_id = None


def test_concat_tail_padding_without_pad_token_id():
    """Defect 3: a None ``pad_token_id`` must fall back instead of crashing."""

    dataset = [{"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}]

    eos_padded = prepare_calibration_dataset(
        _qmodel(_NoPadTokenizer()),
        dataset,
        calibration_dataset_concat_size=6,
        batch_size=1,
        calibration_data_min_length=1,
    )
    assert eos_padded[0]["input_ids"].tolist() == [[1, 2, 3, 99, 99, 99]]
    assert eos_padded[0]["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0]]

    zero_padded = prepare_calibration_dataset(
        _qmodel(_NoPadNoEosTokenizer()),
        dataset,
        calibration_dataset_concat_size=6,
        batch_size=1,
        calibration_data_min_length=1,
    )
    assert zero_padded[0]["input_ids"].tolist() == [[1, 2, 3, 0, 0, 0]]
    assert zero_padded[0]["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0]]


def test_all_zero_additive_or_extended_mask_keeps_every_token():
    """Defect 4: an unpadded all-zero additive/extended mask means keep all."""

    float_recorder = _Recorder()
    float_mask = prepare_calibration_dataset(
        _qmodel(),
        [{"input_ids": [1, 2, 3, 4, 5], "attention_mask": torch.zeros(1, 5)}],
        batch_size=1,
        calibration_data_min_length=1,
        logger=float_recorder,
    )
    assert float_mask[0]["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert float_mask[0]["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]

    extended_recorder = _Recorder()
    extended_mask = prepare_calibration_dataset(
        _qmodel(),
        [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": torch.zeros(1, 1, 1, 5, dtype=torch.long),
            }
        ],
        batch_size=1,
        calibration_data_min_length=1,
        logger=extended_recorder,
    )
    assert extended_mask[0]["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert not any("length <=" in message for message in extended_recorder.warnings)

    # Only genuinely masked positions of an extended mask are dropped.
    mixed = prepare_calibration_dataset(
        _qmodel(),
        [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": torch.tensor([[[[0.0, 0.0, 0.0, -1e9, 0.0]]]]),
            }
        ],
        batch_size=1,
        calibration_data_min_length=1,
    )
    assert mixed[0]["attention_mask"].tolist() == [[1, 1, 1, 0, 1]]

    # A genuine 2-D 0/1 all-padding row is still dropped, with an accurate
    # message (not the old "input_ids with length <= 0" text).
    padding_recorder = _Recorder()
    dropped = prepare_calibration_dataset(
        _qmodel(),
        [{"input_ids": [1, 2, 3], "attention_mask": [0, 0, 0]}],
        batch_size=1,
        calibration_data_min_length=0,
        logger=padding_recorder,
    )
    assert dropped == []
    assert any("all-padding" in message for message in padding_recorder.warnings)
    assert not any("input_ids with length" in message for message in padding_recorder.warnings)


def test_loop_processor_size_warning_counts_rows_not_columns(monkeypatch, tmp_path):
    """Defect 5: a columnar dict must be measured by rows, not columns."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LoopProcessor, "_init_device_smi_handles", lambda _self: {})
    monkeypatch.setattr(LoopProcessor, "_init_cpu_device_handle", lambda _self: None)

    warnings = []
    monkeypatch.setattr(
        loop_processor_module.log,
        "warn",
        lambda message, *args, **kwargs: warnings.append(str(message)),
    )

    rows = 100
    columns = {
        "input_ids": [[1, 2, 3] for _ in range(rows)],
        "attention_mask": [[1, 1, 1] for _ in range(rows)],
    }
    assert _calibration_dataset_length(columns) == rows

    prepared = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        for _ in range(rows)
    ]
    LoopProcessor(
        tokenizer=None,
        qcfg=MagicMock(),
        calibration=CalibrationConfig(dataset=columns).bind(lambda **kwargs: prepared),
    )

    size_warnings = [message for message in warnings if "size should be more than" in message]
    assert size_warnings, warnings
    assert "Current: 100" in size_warnings[0]
