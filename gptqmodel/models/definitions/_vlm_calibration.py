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
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import torch

from ...utils.logger import setup_logger


log = setup_logger()

_MISSING = object()


def is_pretokenized_sample(sample: Any) -> bool:
    """Return whether ``sample`` is a token-id row the base text path accepts."""

    if isinstance(sample, Mapping):
        return "input_ids" in sample and set(sample.keys()) <= {"input_ids", "attention_mask"}
    if isinstance(sample, (list, tuple)):
        return bool(sample) and all(isinstance(token, int) for token in sample)
    return torch.is_tensor(sample)


def split_calibration_dataset(
    calibration_dataset: Iterable[Any],
    *,
    label: str,
) -> tuple[Optional[list[Any]], Optional[Iterable[Any]]]:
    """Split calibration rows into pre-tokenized rows and conversations.

    Returns ``(pretokenized_rows, conversations)``; exactly one of the two is
    ``None``. Mixing both row kinds in one dataset is rejected because the
    processor path cannot consume token ids and silently dropping them would
    change the calibration set.
    """

    iterator = iter(calibration_dataset)
    first = next(iterator, _MISSING)
    if first is _MISSING:
        # Let the base text path report the empty dataset.
        return [], None
    if not is_pretokenized_sample(first):
        return None, itertools.chain((first,), iterator)

    rows = [first]
    for sample in iterator:
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
