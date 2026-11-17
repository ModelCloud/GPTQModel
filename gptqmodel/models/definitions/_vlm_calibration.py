# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Shared calibration-dataset handling for vision-language definitions.

``quantize()`` accepts both pre-tokenized rows (``input_ids``) and chat
conversations. Vision-language processors only understand conversations, so
definitions that override ``prepare_dataset`` use these helpers to keep the
base text path for pre-tokenized rows instead of handing token ids to
``processor.apply_chat_template`` (issue #4228).

The base text path keeps only ``input_ids``/``attention_mask``. A row that also
carries multimodal inputs (``pixel_values``, ``image_grid_thw``, ...) cannot use
that path without silently dropping them, so the splitter rejects it with an
actionable error instead of routing it to either path (issue #4371). Scalar or
string metadata columns that ``datasets.map(tokenize)`` keeps (``id``, ``url``,
``source``, ...) are not model inputs, so they do not block the text path
(issue #4455).
"""

from __future__ import annotations

import itertools
import numbers
import random
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import numpy as np
import torch

from ...utils.calibration import (
    _columnar_mapping_examples,
    _is_columnar_mapping,
    _is_dataset_split_mapping,
    _is_integer_sequence,
)
from ...utils.logger import setup_logger


log = setup_logger()

_MISSING = object()

# Columns the base text path can consume or safely ignore. Any other column must
# hold plain metadata (see `_is_metadata_value`); it may otherwise carry
# multimodal model inputs that must not be dropped silently.
_TEXT_ONLY_COLUMNS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "text",
        "messages",
        "token_type_ids",
        "labels",
        "special_tokens_mask",
        "position_ids",
        "offset_mapping",
        "overflow_to_sample_mapping",
        "num_truncated_tokens",
    }
)


# Column-name parts that identify multimodal model inputs or references to
# them (``pixel_values``, ``image_grid_thw``, ``video_grid_thw``, ``image_url``,
# ``input_features``, ...). Such a column is rejected even when it holds a
# string or scalar: a path, URL or size for an image is still an image input.
_MULTIMODAL_COLUMN_NAME_PARTS = frozenset(
    {
        "pixel",
        "pixels",
        "image",
        "images",
        "img",
        "video",
        "videos",
        "audio",
        "audios",
        "vision",
        "visual",
        "thw",
        "features",
    }
)


def _is_multimodal_column_name(name: Any) -> bool:
    parts = str(name).lower().replace("-", "_").split("_")
    return any(part in _MULTIMODAL_COLUMN_NAME_PARTS for part in parts)


def _is_metadata_value(value: Any) -> bool:
    """Return whether ``value`` is plain row metadata, not a model input.

    Scalars (``None``, bool, numbers, strings) and flat string lists are
    metadata. Tensors, arrays, numeric sequences, bytes, mappings and any other
    object may hold multimodal inputs and are not.
    """

    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, np.generic):
        return value.ndim == 0
    if isinstance(value, numbers.Number):
        return True
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, str) for item in value)
    return False


def _non_text_columns(sample: Mapping, *, columnar: bool = False) -> list[str]:
    """Return mapping columns the text calibration path would drop as model inputs.

    ``columnar`` checks every row value of a column-major mapping.
    """

    columns = []
    for name, value in sample.items():
        if name in _TEXT_ONLY_COLUMNS:
            continue
        if _is_multimodal_column_name(name):
            columns.append(str(name))
            continue
        if columnar and isinstance(value, (list, tuple)):
            is_metadata = all(_is_metadata_value(item) for item in value)
        else:
            is_metadata = _is_metadata_value(value)
        if not is_metadata:
            columns.append(str(name))
    return columns


def _raise_non_text_columns(columns: list[str], *, label: str) -> None:
    raise ValueError(
        f"{label}: calibration row provides non-text column(s) {columns}. The VLM "
        "text calibration path keeps only `input_ids`/`attention_mask`, so these "
        "values would be silently dropped. Pass the source conversations and let "
        "the model processor build the multimodal inputs, or remove the extra "
        "columns for text-only calibration."
    )


def is_pretokenized_sample(sample: Any) -> bool:
    """Return whether ``sample`` is a token-id row the base text path accepts.

    The base path keeps only ``input_ids``/``attention_mask`` from a mapping, so
    extra text-only tokenizer columns (``token_type_ids``, ``labels``, ...) must
    not divert the row to the processor path, where it would be wrapped as a
    chat message. Rows with multimodal columns are not pre-tokenized text: the
    splitter rejects them so their tensors are never dropped. NumPy arrays and
    rows built from NumPy/0-D-tensor token ids are token ids too.
    """

    if isinstance(sample, Mapping):
        return "input_ids" in sample and not _non_text_columns(sample)
    if isinstance(sample, (list, tuple)):
        return bool(sample) and _is_integer_sequence(sample)
    if isinstance(sample, np.ndarray):
        return True
    return torch.is_tensor(sample)


def _mapping_rows(calibration_dataset: Mapping, *, label: str) -> list[Any]:
    """Normalize a mapping calibration input the same way the base helper does."""

    if len(calibration_dataset) == 0:
        return []
    if _is_columnar_mapping(calibration_dataset):
        if "input_ids" in calibration_dataset:
            columns = _non_text_columns(calibration_dataset, columnar=True)
            if columns:
                _raise_non_text_columns(columns, label=label)
        return _columnar_mapping_examples(calibration_dataset)
    if _is_dataset_split_mapping(calibration_dataset):
        splits = ", ".join(repr(key) for key in calibration_dataset.keys())
        raise ValueError(
            f"{label}: calibration dataset is a mapping of dataset splits "
            f"({splits}); select one split, e.g. `calibration_dataset['train']`."
        )
    if any(name in calibration_dataset for name in ("input_ids", "text", "messages")):
        return [calibration_dataset]
    raise ValueError(
        f"{label}: calibration dataset mapping must provide `input_ids`, `text`, "
        f"or `messages` columns, got keys={list(calibration_dataset.keys())}."
    )


def _reject_non_text_row(sample: Any, *, label: str) -> None:
    """Reject a token-id row whose extra columns may hold multimodal tensors."""

    if not isinstance(sample, Mapping) or "input_ids" not in sample:
        return
    columns = _non_text_columns(sample)
    if columns:
        _raise_non_text_columns(columns, label=label)


def split_calibration_dataset(
    calibration_dataset: Iterable[Any],
    *,
    label: str,
) -> tuple[Optional[list[Any]], Optional[Iterable[Any]]]:
    """Split calibration rows into pre-tokenized rows and conversations.

    Returns ``(pretokenized_rows, conversations)``; exactly one of the two is
    ``None``. Mixing both row kinds in one dataset is rejected because the
    processor path cannot consume token ids and silently dropping them would
    change the calibration set. A row that carries multimodal tensors is
    rejected outright: neither path can forward them, and the text path would
    drop them silently.
    """

    if isinstance(calibration_dataset, Mapping):
        calibration_dataset = _mapping_rows(calibration_dataset, label=label)

    iterator = iter(calibration_dataset)
    first = next(iterator, _MISSING)
    if first is _MISSING:
        # Let the base text path report the empty dataset.
        return [], None
    _reject_non_text_row(first, label=label)
    if not is_pretokenized_sample(first):
        return None, itertools.chain((first,), iterator)

    rows = [first]
    for sample in iterator:
        _reject_non_text_row(sample, label=label)
        if not is_pretokenized_sample(sample):
            raise ValueError(
                f"{label}: calibration dataset mixes pre-tokenized rows with chat "
                "conversations. Pass either token ids or conversations, not both."
            )
        rows.append(sample)
    return rows, None


def pop_conversation_calibration_options(
    kwargs: dict,
    *,
    label: str,
) -> tuple[Optional[str], Optional[int]]:
    """Extract the calibration options the conversation path can honour.

    ``calibration_concat_size`` is rejected: concatenating token ids would break
    the image/token alignment the processor produced. ``calibration_sort`` and
    ``calibration_data_min_length`` are returned so the conversation path can
    apply them to the prepared batches.
    """

    concat_size = kwargs.pop("calibration_dataset_concat_size", None)
    sort = kwargs.pop("calibration_dataset_sort", None)
    min_length = kwargs.pop("calibration_data_min_length", None)
    kwargs.pop("calibration_concat_separator", None)

    if concat_size is not None:
        raise ValueError(
            f"{label}: `calibration_concat_size` is not supported for conversation "
            f"calibration data (got {concat_size}) because concatenating token ids "
            "would break the image/token alignment. Pass pre-tokenized `input_ids` "
            "rows to use it."
        )

    return sort, min_length


def _batch_input_ids_length(batch: Any) -> Optional[int]:
    if not isinstance(batch, Mapping):
        return None
    input_ids = batch.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.shape[-1]) if input_ids.dim() > 0 else 1
    if isinstance(input_ids, (list, tuple)) and input_ids:
        first = input_ids[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(input_ids)
    return None


def finalize_conversation_batches(
    batches: Iterable[Any],
    *,
    sort: Optional[str] = None,
    min_length: Optional[int] = None,
    label: str = "VLM",
) -> list[Any]:
    """Apply the base calibration options to processor-prepared batches."""

    batches = list(batches)

    if min_length is not None:
        kept = []
        dropped = 0
        for batch in batches:
            length = _batch_input_ids_length(batch)
            if length is not None and length <= min_length:
                dropped += 1
                continue
            kept.append(batch)
        if dropped:
            log.warn(
                f"{label}: {dropped} calibration conversation(s) with input_ids length "
                f"<= {min_length} were removed. Use "
                "quantize(calibration_data_min_length=...) to set a custom minimum length."
            )
        batches = kept

    if sort == "asc":
        batches.sort(key=lambda item: _batch_input_ids_length(item) or 0)
    elif sort == "desc":
        batches.sort(key=lambda item: _batch_input_ids_length(item) or 0, reverse=True)
    elif sort == "shuffle":
        random.shuffle(batches)

    return batches
