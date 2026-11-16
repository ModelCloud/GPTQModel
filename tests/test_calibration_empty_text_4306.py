# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Empty-text calibration rows must be filtered, not fatal (issue #4306).

Tokenizers that add no BOS (Qwen and others) encode ``""`` as a zero-width
``(1, 0)`` row. #4194's zero-row guard also rejected that shape, so one blank
line in the calibration data aborted ``quantize()``. A zero-ROW item
(``torch.empty(0, N)``) is still a structural error.
"""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.calibration import prepare_calibration_dataset


MAX_POSITIONS = 128


class _RecordingLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def info(self, message, *args, **kwargs):
        self.infos.append(message % args if args else message)

    def warn(self, message, *args, **kwargs):
        self.warnings.append(message % args if args else message)

    # `logbar` exposes both spellings; the code base uses `warn`.
    warning = warn


class _NoBosTokenizer:
    """Encodes text without adding a BOS: ``""`` becomes shape ``(1, 0)``."""

    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, add_special_tokens=True, return_tensors="pt"):
        token_count = len(text.split()) if text.strip() else 0
        input_ids = torch.tensor([list(range(1, token_count + 1))], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{int(token)}" for token in ids)


def _qmodel():
    return SimpleNamespace(
        tokenizer=_NoBosTokenizer(),
        support_batch_quantize=True,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=MAX_POSITIONS)),
    )


def _prepare(dataset, logger=None, **kwargs):
    kwargs.setdefault("batch_size", 1)
    kwargs.setdefault("calibration_data_min_length", 10)
    return prepare_calibration_dataset(_qmodel(), dataset, logger=logger or _RecordingLogger(), **kwargs)


def _sentence(index):
    return " ".join(f"w{index}_{token}" for token in range(12))


def test_blank_text_row_in_list_is_dropped():
    # Mirrors the issue repro: one blank entry among valid calibration texts.
    texts = [_sentence(index) for index in range(32)]
    texts.insert(5, "")
    logger = _RecordingLogger()

    prepared = _prepare(texts, logger=logger)

    assert len(prepared) == 32
    assert all(item["input_ids"].shape == (1, 12) for item in prepared)
    assert any("were removed" in message for message in logger.warnings)


def test_blank_text_dict_is_dropped():
    logger = _RecordingLogger()

    prepared = _prepare([{"text": ""}], logger=logger)

    assert prepared == []
    assert any("were removed" in message for message in logger.warnings)


def test_empty_nested_list_gets_short_row_handling():
    # `torch.as_tensor([[]])` is float32; that must not surface as a misleading
    # "must contain integers" dtype error.
    logger = _RecordingLogger()

    prepared = _prepare([[]], logger=logger)

    assert prepared == []
    assert any("were removed" in message for message in logger.warnings)


def test_empty_row_with_explicit_mask_is_dropped():
    empty = torch.empty(1, 0, dtype=torch.long)

    prepared = _prepare([{"input_ids": empty, "attention_mask": empty.clone()}])

    assert prepared == []


def test_zero_token_1d_tensor_is_dropped():
    # A 1-D empty tensor becomes `(1, 0)`: a zero-token row, not a zero-row item.
    prepared = _prepare([torch.empty(0, dtype=torch.long)])

    assert prepared == []


@pytest.mark.parametrize(
    "sample",
    [
        torch.empty(0, 8, dtype=torch.long),
        {"input_ids": torch.empty(0, 8, dtype=torch.long)},
    ],
)
def test_zero_row_item_still_raises_a_clear_error(sample):
    with pytest.raises(ValueError, match=r"empty `input_ids`.*shape \(0, 8\)"):
        _prepare([sample])


def test_float_token_ids_are_still_rejected():
    with pytest.raises(ValueError, match="integers"):
        _prepare([torch.tensor([[1.0, 2.0, 3.0]])])
