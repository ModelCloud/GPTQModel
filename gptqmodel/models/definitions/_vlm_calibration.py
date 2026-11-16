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

import numpy as np
import torch

from ...utils.calibration import (
    _columnar_mapping_examples,
    _is_batched_column,
    _is_columnar_mapping,
    _is_dataset_split_mapping,
    _is_integer_sequence,
)
from ...utils.logger import setup_logger


log = setup_logger()

_MISSING = object()

# Extra columns a tokenizer may attach to a text row. The base text path keeps
# only ``input_ids``/``attention_mask`` and drops these, which is harmless: they
# never describe an image/video/audio input, so diverting the row to the
# processor path (where it would be wrapped as a chat message) would be worse.
_TEXT_ONLY_SAMPLE_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
        "special_tokens_mask",
        "offset_mapping",
        "overflow_to_sample_mapping",
    }
)

# Processor output columns that carry multimodal model inputs. These are matched
# by name as well as by value shape so a PIL image or a nested list column is
# not mistaken for harmless row metadata.
_MULTIMODAL_SAMPLE_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_sizes",
        "aspect_ratio_ids",
        "aspect_ratio_mask",
        "cross_attention_mask",
        "image_embeds",
        "video_embeds",
        "second_per_grid_ts",
        "images",
        "image",
        "videos",
        "video",
        "audios",
        "audio",
    }
)


def _non_text_model_input_keys(sample: Mapping) -> list[str]:
    """Return columns that carry model inputs the base text path would discard.

    The base path rebuilds every row from ``input_ids``/``attention_mask`` only,
    so a tensor/array payload such as ``pixel_values`` would be silently dropped
    and the calibration set would change without any warning. Scalar metadata
    columns (an ``id``, a ``source`` string) and text-only tokenizer columns are
    not model inputs and are ignored here.
    """

    keys: list[str] = []
    for key, value in sample.items():
        if key in _TEXT_ONLY_SAMPLE_KEYS:
            continue
        if key in _MULTIMODAL_SAMPLE_KEYS or torch.is_tensor(value) or isinstance(value, np.ndarray):
            keys.append(str(key))
        elif isinstance(value, (list, tuple)) and _is_batched_column(value):
            keys.append(str(key))
    return keys


def _reject_dropped_model_inputs(sample: Any, *, label: str, where: str) -> None:
    """Fail loudly instead of letting the text path strip multimodal tensors."""

    if not isinstance(sample, Mapping) or "input_ids" not in sample:
        return
    keys = _non_text_model_input_keys(sample)
    if not keys:
        return
    raise ValueError(
        f"{label}: {where} contains `input_ids` together with non-text model "
        f"inputs {keys}. The pre-tokenized text path keeps only `input_ids` and "
        "`attention_mask`, so these tensors would be silently dropped and the "
        "calibration data would change. Pass chat conversations instead (the "
        "processor then builds `pixel_values`/`image_grid_thw` from the images), "
        "or remove the extra columns to calibrate on token ids only."
    )


def is_pretokenized_sample(sample: Any) -> bool:
    """Return whether ``sample`` is a token-id row the base text path accepts.

    The base path keeps only ``input_ids``/``attention_mask`` from a mapping, so
    extra tokenizer columns (``token_type_ids``, ...) must not divert the row to
    the processor path, where it would be wrapped as a chat message. NumPy
    arrays and rows built from NumPy/0-D-tensor token ids are token ids too.

    A row that also carries multimodal model inputs (``pixel_values``, ...) is
    *not* a pre-tokenized text row: the text path cannot forward those tensors,
    and ``split_calibration_dataset`` rejects such a row explicitly.
    """

    if isinstance(sample, Mapping):
        return "input_ids" in sample and not _non_text_model_input_keys(sample)
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
        # Columnar mappings are unpacked into rows, so the extra columns would be
        # gone before the per-row check below can see them.
        _reject_dropped_model_inputs(
            calibration_dataset, label=label, where="calibration mapping columns"
        )
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

    if isinstance(calibration_dataset, Mapping):
        calibration_dataset = _mapping_rows(calibration_dataset, label=label)

    iterator = iter(calibration_dataset)
    first = next(iterator, _MISSING)
    if first is _MISSING:
        # Let the base text path report the empty dataset.
        return [], None
    _reject_dropped_model_inputs(first, label=label, where="calibration row 0")
    if not is_pretokenized_sample(first):
        return None, itertools.chain((first,), iterator)

    rows = [first]
    for index, sample in enumerate(iterator, start=1):
        _reject_dropped_model_inputs(sample, label=label, where=f"calibration row {index}")
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
