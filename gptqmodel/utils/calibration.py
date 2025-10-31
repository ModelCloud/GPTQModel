# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utilities for preparing calibration datasets used during quantization."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .data import collate_data
from .logger import setup_logger


try:  # pragma: no cover - optional dependency
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
except Exception:  # pragma: no cover - handled dynamically
    HFDataset = HFIterableDataset = None


CalibrationInputType = Union[
    List[Dict[str, Union[List[int], torch.LongTensor]]],
    List[str],
    List[List[int]],
    "HFDataset",  # type: ignore[type-arg]
    "HFIterableDataset",  # type: ignore[type-arg]
]


def batched(iterable, batch_size: int, process_func=None):
    """Yield fixed-size batches from ``iterable`` after optional processing."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batch = []
    for item in iterable:
        processed = process_func(item) if process_func is not None else item
        batch.append(processed)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def prepare_calibration_dataset(
    qmodel,
    calibration_dataset: CalibrationInputType,
    calibration_dataset_concat_size: Optional[int] = None,
    calibration_dataset_sort: Optional[str] = None,
    batch_size: int = 1,
    calibration_data_min_length: int = 10,
    calibration_concat_separator: Optional[str] = None,
    logger=None,
):
    """Normalize, validate, and batch calibration samples for quantization.

    Parameters mirror ``BaseQModel.prepare_dataset`` so existing code paths can
    delegate directly to this helper.
    """

    log = logger or setup_logger()

    tokenizer = getattr(qmodel, "tokenizer", None)
    support_batch_quantize = getattr(qmodel, "support_batch_quantize", True)

    hf_dataset_types: tuple = ()
    if HFDataset is not None:
        hf_dataset_types += (HFDataset,)
    if HFIterableDataset is not None:
        hf_dataset_types += (HFIterableDataset,)

    if isinstance(calibration_dataset, str):
        raise ValueError("Quantize: calibration dataset must be iterable, not a single string.")

    if hf_dataset_types and isinstance(calibration_dataset, hf_dataset_types):
        raw_examples = list(calibration_dataset)
    elif isinstance(calibration_dataset, list):
        raw_examples = calibration_dataset
    elif isinstance(calibration_dataset, Sequence) and not isinstance(calibration_dataset, (bytes, bytearray)):
        raw_examples = list(calibration_dataset)
    else:
        raw_examples = list(calibration_dataset)

    if len(raw_examples) == 0:
        raise ValueError("Quantize: calibration dataset is empty.")

    def _require_tokenizer(reason: str) -> None:
        if tokenizer is None:
            raise ValueError(f"tokenizer must be provided when {reason}.")

    def _to_2d_long_tensor(value: Any, name: str, idx: int) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value, dtype=torch.long)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Quantize: failed to convert `{name}` to tensor for calibration item {idx}.") from exc

        if tensor.ndim == 0:
            raise ValueError(f"Quantize: `{name}` for calibration item {idx} must be 1D or 2D, got scalar.")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"Quantize: `{name}` for calibration item {idx} must be rank 1 or 2, got rank {tensor.ndim}."
            )
        return tensor

    def _pack_ids(ids_value: Any, mask_value: Any, idx: int) -> Dict[str, torch.Tensor]:
        ids_tensor = _to_2d_long_tensor(ids_value, "input_ids", idx)

        if mask_value is None:
            mask_tensor = torch.ones_like(ids_tensor, dtype=torch.long)
        else:
            mask_tensor = _to_2d_long_tensor(mask_value, "attention_mask", idx)
            if mask_tensor.shape != ids_tensor.shape:
                if mask_tensor.numel() == ids_tensor.numel():
                    mask_tensor = mask_tensor.reshape(ids_tensor.shape)
                else:
                    raise ValueError(
                        f"Quantize: attention_mask shape {tuple(mask_tensor.shape)} does not match input_ids shape "
                        f"{tuple(ids_tensor.shape)} for calibration item {idx}."
                    )

        return {
            "input_ids": ids_tensor.detach(),
            "attention_mask": mask_tensor.detach(),
        }

    def _tokenize_text_value(text_value: Any, idx: int) -> Dict[str, torch.Tensor]:
        _require_tokenizer("calibration data contains raw text")
        tokenized = tokenizer(  # type: ignore[call-arg]
            text_value,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
        return _pack_ids(input_ids, attention_mask, idx)

    def _tokenize_messages_value(messages_value: Any, idx: int) -> Dict[str, torch.Tensor]:
        _require_tokenizer("calibration data uses the `messages` feature")
        apply_fn = getattr(tokenizer, "apply_template", None)
        if apply_fn is None:
            raise ValueError("tokenizer must expose `apply_template` to handle `messages` calibration data.")
        try:
            templated = apply_fn(messages_value, tokenize=False)
        except TypeError:
            templated = apply_fn(messages_value)

        if templated is None:
            raise ValueError(f"tokenizer.apply_template returned None for calibration item {idx}.")

        if hasattr(templated, "get"):
            ids_value = templated.get("input_ids")
            mask_value = templated.get("attention_mask")
            text_value = templated.get("text")
            if ids_value is not None:
                return _pack_ids(ids_value, mask_value, idx)
            if text_value is not None:
                return _tokenize_text_value(text_value, idx)

        if isinstance(templated, (list, tuple)):
            if len(templated) > 0 and isinstance(templated[0], int):
                return _pack_ids(list(templated), None, idx)
            raise ValueError(
                "tokenizer.apply_template returned an unsupported sequence type for calibration item {idx}."
            )

        if torch.is_tensor(templated):
            return _pack_ids(templated, None, idx)

        if isinstance(templated, str):
            return _tokenize_text_value(templated, idx)

        raise ValueError(
            f"tokenizer.apply_template returned unsupported type {type(templated)} for calibration item {idx}."
        )

    processed_examples: List[Dict[str, torch.Tensor]] = []
    for idx, example in enumerate(raw_examples):
        if isinstance(example, dict):
            if "messages" in example:
                apply_fn = getattr(tokenizer, "apply_template", None) if tokenizer else None
                if apply_fn is None:
                    if "text" in example:
                        processed_examples.append(_tokenize_text_value(example["text"], idx))
                        continue
                    raise ValueError(
                        "tokenizer must expose `apply_template` or calibration data must provide `text` when using `messages`."
                    )
                processed_examples.append(_tokenize_messages_value(example["messages"], idx))
                continue
            if "text" in example:
                processed_examples.append(_tokenize_text_value(example["text"], idx))
                continue
            if "input_ids" in example:
                processed_examples.append(_pack_ids(example["input_ids"], example.get("attention_mask"), idx))
                continue
            raise ValueError(
                f"Quantize: unsupported calibration example structure at index {idx}: keys={list(example.keys())}"
            )

        if isinstance(example, str):
            processed_examples.append(_tokenize_text_value(example, idx))
            continue

        if isinstance(example, (list, tuple)):
            if all(isinstance(x, int) for x in example):
                processed_examples.append(_pack_ids(list(example), None, idx))
                continue
            raise ValueError(
                f"Quantize: list-based calibration example at index {idx} must contain only integers."
            )

        if torch.is_tensor(example):
            processed_examples.append(_pack_ids(example, None, idx))
            continue

        try:
            processed_examples.append(_pack_ids(example, None, idx))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Quantize: unsupported calibration example type {type(example)} at index {idx}."
            ) from exc

    calibration_dataset = processed_examples

    def _convert_tensor_to_list(tensor):
        if isinstance(tensor, torch.Tensor):
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.long()
            return tensor.cpu().numpy().tolist()
        return [tensor]

    new_calibration_dataset = []
    too_short_calibration_data_count = 0

    max_positions = None
    max_positions_source = None
    trimmed_row_count = 0
    longest_trimmed_row = 0

    def _maybe_resolve_length(value, source_name):
        nonlocal max_positions, max_positions_source
        try:
            if value is None:
                return False
            limit = int(value)
        except Exception:
            return False
        if limit <= 0:
            return False
        if max_positions is None or limit < max_positions:
            max_positions = limit
            max_positions_source = source_name
        return True

    model_config = getattr(getattr(qmodel, "model", None), "config", None)
    if model_config is not None:
        primary_names = ("max_position_embeddings",)
        fallback_names = (
            "max_sequence_length",
            "max_seq_len",
            "n_positions",
            "seq_length",
        )

        for attr_name in primary_names:
            if _maybe_resolve_length(getattr(model_config, attr_name, None), attr_name):
                break
        if max_positions is None:
            for attr_name in fallback_names:
                if _maybe_resolve_length(getattr(model_config, attr_name, None), attr_name):
                    break

    for example in calibration_dataset:
        input_ids = _convert_tensor_to_list(example["input_ids"])
        attention_mask = _convert_tensor_to_list(example["attention_mask"])

        if max_positions is not None:
            trimmed = False
            trimmed_input_ids = []
            trimmed_attention_mask = []

            for row_ids, row_mask in zip(input_ids, attention_mask):
                row_len = len(row_ids)
                if row_len > max_positions:
                    trimmed = True
                    trimmed_row_count += 1
                    longest_trimmed_row = max(longest_trimmed_row, row_len)
                    trimmed_input_ids.append(row_ids[:max_positions])
                    trimmed_attention_mask.append(row_mask[:max_positions])
                else:
                    trimmed_input_ids.append(row_ids)
                    trimmed_attention_mask.append(row_mask)

            if trimmed:
                input_ids = trimmed_input_ids
                attention_mask = trimmed_attention_mask

        if len(input_ids[0]) <= calibration_data_min_length:
            too_short_calibration_data_count += 1
            continue

        new_calibration_dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    if too_short_calibration_data_count > 0:
        log.warn(
            f"Quantize: {too_short_calibration_data_count} input_ids with length <= {calibration_data_min_length} were removed. "
            f"Use quantize(calibration_data_min_length={calibration_data_min_length}) to set a custom minimum length."
        )

    if trimmed_row_count > 0:
        log.info(
            "Quantize: trimmed %s calibration rows above %s=%s (longest original length=%s)",
            trimmed_row_count,
            max_positions_source,
            max_positions,
            longest_trimmed_row,
        )

    if calibration_dataset_concat_size:
        _require_tokenizer("`calibration_dataset_concat_size` is specified")
        concatenated_data = []
        input_ids_buff = []
        attention_mask_buff = []
        current_length = 0

        separator = calibration_concat_separator if calibration_concat_separator is not None else ""
        if separator:
            new_line = tokenizer(separator, return_tensors="pt")  # type: ignore[call-arg]
            new_line_input_ids = _convert_tensor_to_list(new_line["input_ids"])[0]
            new_line_attention_mask = _convert_tensor_to_list(new_line["attention_mask"])[0]
        else:
            new_line_input_ids = []
            new_line_attention_mask = []
        new_line_input_ids_len = len(new_line_input_ids)

        def flush_buffer():
            nonlocal input_ids_buff, attention_mask_buff, current_length
            concatenated_data.append(
                {
                    "input_ids": [input_ids_buff],
                    "attention_mask": [attention_mask_buff],
                }
            )
            input_ids_buff = []
            attention_mask_buff = []
            current_length = 0

        for example in new_calibration_dataset:
            row_ids = example["input_ids"][0]
            row_mask = example["attention_mask"][0]
            position = 0
            row_length = len(row_ids)

            while position < row_length:
                if input_ids_buff:
                    if new_line_input_ids_len:
                        if current_length + new_line_input_ids_len > calibration_dataset_concat_size:
                            flush_buffer()
                            continue
                        input_ids_buff.extend(new_line_input_ids)
                        attention_mask_buff.extend(new_line_attention_mask)
                        current_length += new_line_input_ids_len

                available = calibration_dataset_concat_size - current_length
                if available == 0:
                    flush_buffer()
                    continue

                chunk_len = min(available, row_length - position)
                if chunk_len == 0:
                    flush_buffer()
                    continue

                end = position + chunk_len
                input_ids_buff.extend(row_ids[position:end])
                attention_mask_buff.extend(row_mask[position:end])
                current_length += chunk_len
                position = end

                if current_length == calibration_dataset_concat_size:
                    flush_buffer()

        if input_ids_buff:
            padding_length = calibration_dataset_concat_size - len(input_ids_buff)
            if padding_length > 0:
                pad_id = getattr(tokenizer, "pad_token_id", 0)
                input_ids_buff.extend([pad_id] * padding_length)
                attention_mask_buff.extend([0] * padding_length)
            concatenated_data.append(
                {
                    "input_ids": [input_ids_buff],
                    "attention_mask": [attention_mask_buff],
                }
            )

        new_calibration_dataset = concatenated_data

    if calibration_dataset_sort == "asc":
        log.info("Calibration: Sort in ascending order by length")
        sorted_dataset = sorted(
            new_calibration_dataset,
            key=lambda item: len(item["input_ids"][0]),
        )
    elif calibration_dataset_sort == "desc":
        log.info("Calibration: Sort in descending order by length")
        sorted_dataset = sorted(
            new_calibration_dataset,
            key=lambda item: len(item["input_ids"][0]),
            reverse=True,
        )
    elif calibration_dataset_sort == "shuffle":
        log.info("Calibration: Sort by random shuffle")
        sorted_dataset = new_calibration_dataset[:]
        random.shuffle(sorted_dataset)
    else:
        log.info("Calibration: Native order")
        sorted_dataset = new_calibration_dataset

    if support_batch_quantize:
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
        new_calibration_dataset_batched = [
            collate_data(sorted_dataset[start : start + batch_size], pad_token_id)
            for start in range(0, len(sorted_dataset), batch_size)
        ]

        total_padded = 0
        total_non_padded = 0

        for batch in new_calibration_dataset_batched:
            mask = batch["attention_mask"]
            total_padded += (mask == 0).sum().item()
            total_non_padded += (mask == 1).sum().item()

        log.info(f"Calibration: Total padded tokens: {total_padded}")
        log.info(f"Calibration: Total non-padded tokens: {total_non_padded}")
        log.info(f"Calibration: Total tokens: {total_non_padded + total_padded}")
    else:
        new_calibration_dataset_batched = [
            {
                "input_ids": torch.tensor(block["input_ids"], dtype=torch.long),
            }
            for block in sorted_dataset
        ]

    return new_calibration_dataset_batched


__all__ = ["batched", "prepare_calibration_dataset"]
