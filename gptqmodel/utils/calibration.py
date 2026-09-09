# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utilities for preparing calibration datasets used during quantization."""

from __future__ import annotations

import math
import os
import random
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlsplit, urlunsplit

import torch

from ..quantization.config import META_FIELD_CALIBRATION_PATHS
from .attn_mask import normalize_seq_mask
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


def normalize_chat_calibration_sample(sample: Any) -> Any:
    """Normalize text-like calibration rows to processor chat conversations."""

    if isinstance(sample, dict):
        if "messages" in sample:
            return sample["messages"]
        if isinstance(sample.get("text"), str):
            sample = sample["text"]

    if isinstance(sample, str):
        return [{"role": "user", "content": [{"type": "text", "text": sample}]}]

    return sample


def batched_conversations(iterable, batch_size: int, process_func=None):
    """Yield chat-normalized calibration samples in fixed-size batches."""

    def normalize_and_process(sample):
        sample = normalize_chat_calibration_sample(sample)
        return process_func(sample) if process_func is not None else sample

    yield from batched(iterable, batch_size, process_func=normalize_and_process)


# Remote dataset identifiers (https://..., hf://..., s3://..., etc.) that are not arbitrary text.
_CALIBRATION_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+$")
_CALIBRATION_URI_PREFIX_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

# Filesystem path heuristics used before we pay for an os.path.exists() syscall.
_KNOWN_FILE_EXTS = (
    ".json",
    ".jsonl",
    ".parquet",
    ".csv",
    ".txt",
    ".gz",
    ".zip",
    ".bin",
    ".safetensors",
    ".arrow",
    ".hf",
)
_MAX_PATH_LEN = 4096


def _looks_like_path(value: str) -> bool:
    """Cheap syntactic check: does ``value`` plausibly name a file or directory?"""
    if value.startswith(("/", "./", "../", "~")) or (len(value) >= 2 and value[1] == ":" and value[0].isalpha()):
        return True
    if "/" in value or "\\" in value:
        return True
    if any(value.lower().endswith(ext) for ext in _KNOWN_FILE_EXTS):
        return True
    return False


def _resolve_calibration_path(value: Any) -> Optional[str]:
    """Return a normalized calibration source string, or None if ``value`` is not a path/URI.

    Existing local filesystem paths are reduced to their basename so that saved
    model configs do not leak the user's local directory layout or username.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > _MAX_PATH_LEN:
        return None
    # Calibration text rows almost always contain whitespace; skip the syscall.
    if re.search(r"\s", value):
        return None
    if _CALIBRATION_URI_RE.match(value):
        try:
            parsed = urlsplit(value)
            if parsed.hostname is None:
                return None
            # Accessing ``port`` validates malformed values that ``urlsplit``
            # otherwise leaves unchecked until the attribute is read.
            _ = parsed.port
            # Dataset URLs can contain basic-auth credentials, signed query
            # parameters, or private fragments. They are not needed for saved
            # provenance and must never be persisted in model metadata.
            safe_netloc = parsed.netloc.rsplit("@", 1)[-1]
            return urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))
        except ValueError:
            return None
    if not _looks_like_path(value):
        return None

    try:
        if os.path.exists(value):
            return os.path.basename(value.rstrip(os.sep)) or value
    except (OSError, ValueError):
        pass

    return None


def _extract_calibration_paths(calibration_dataset: Any) -> List[str]:
    """Extract path-like dataset identifiers from the raw calibration input."""
    if isinstance(calibration_dataset, str):
        resolved = _resolve_calibration_path(calibration_dataset)
        return [resolved] if resolved else []
    if isinstance(calibration_dataset, (list, tuple)):
        return [r for r in (_resolve_calibration_path(item) for item in calibration_dataset) if r]

    paths: List[str] = []
    seen: set = set()

    cache_files = getattr(calibration_dataset, "cache_files", None)
    if cache_files:
        for cf in cache_files:
            if isinstance(cf, dict):
                path = cf.get("filename") or cf.get("path")
            elif isinstance(cf, str):
                path = cf
            else:
                path = None
            if path:
                if isinstance(path, str) and _CALIBRATION_URI_PREFIX_RE.match(path.strip()):
                    path = _resolve_calibration_path(path)
                else:
                    # Cache paths are absolute and may contain local usernames/dirs;
                    # keep only the filename to avoid leaking local filesystem layout.
                    path = os.path.basename(path)
                if path and path not in seen:
                    seen.add(path)
                    paths.append(path)

    for info_attr in ("info", "_info"):
        info = getattr(calibration_dataset, info_attr, None)
        if info is not None:
            for name_attr in ("dataset_name", "builder_name"):
                val = getattr(info, name_attr, None)
                resolved = _resolve_calibration_path(val)
                # URI-looking metadata is security-sensitive provenance.  If
                # parsing fails, drop it instead of falling back to a raw value
                # that may contain credentials or signed query parameters.
                if (
                    resolved is None
                    and isinstance(val, str)
                    and _CALIBRATION_URI_PREFIX_RE.match(val.strip())
                ):
                    continue
                normalized = resolved if resolved is not None else val
                if (
                    isinstance(val, str)
                    and val
                    and normalized not in seen
                    and ("/" in val or resolved is not None)
                ):
                    seen.add(normalized)
                    paths.append(normalized)
            break

    return paths


def _record_calibration_source(qmodel, calibration_dataset: Any) -> None:
    """Append unique path-like calibration sources to the quantize config meta."""
    qcfg = getattr(qmodel, "quantize_config", None)
    if qcfg is None:
        return

    paths = _extract_calibration_paths(calibration_dataset)
    if not paths:
        return

    existing = qcfg.meta_get(META_FIELD_CALIBRATION_PATHS) or []
    if not isinstance(existing, list):
        existing = [existing]

    merged = existing[:]
    for path in paths:
        if path not in merged:
            merged.append(path)
    qcfg.meta_set(META_FIELD_CALIBRATION_PATHS, merged)


def prepare_calibration_dataset(
    qmodel,
    calibration_dataset: CalibrationInputType,
    calibration_dataset_concat_size: Optional[int] = None,
    calibration_dataset_sort: Optional[str] = None,
    batch_size: int = 1,
    calibration_data_min_length: int = 10,
    calibration_concat_separator: Optional[str] = None,
    chat_template_config=None,
    source_weight_column: Optional[str] = None,
    source_weights: Optional[Sequence[Sequence[Union[str, float]]]] = None,
    logger=None,
):
    """Normalize, validate, and batch calibration samples for quantization.

    Parameters mirror ``BaseQModel.prepare_dataset`` so existing code paths can
    delegate directly to this helper.
    """

    log = logger or setup_logger()
    chat_template_weighting = bool(getattr(chat_template_config, "enabled", False))
    source_weight_map = dict(source_weights or ())
    if bool(source_weight_column) != bool(source_weight_map):
        raise ValueError(
            "source_weight_column and non-empty source_weights must be configured together"
        )
    for source_name, source_weight in source_weight_map.items():
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("source-weight names must be non-empty strings")
        if (
            isinstance(source_weight, bool)
            or not isinstance(source_weight, (int, float))
            or not math.isfinite(float(source_weight))
            or float(source_weight) <= 0
        ):
            raise ValueError("source weights must be finite and positive")

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

    _record_calibration_source(qmodel, calibration_dataset)

    message_examples = 0
    message_template_name = None
    message_text_fallback_examples = 0

    def _require_tokenizer(reason: str) -> None:
        if tokenizer is None:
            raise ValueError(f"tokenizer must be provided when {reason}.")

    message_apply_fn = None
    message_apply_name = None
    message_template_checked = False

    def _get_message_template():
        # Prefer the model's native chat formatter when calibration rows carry
        # `messages`, but only when the tokenizer has an actual template to use.
        # Some HF tokenizers expose `apply_chat_template()` while leaving
        # `chat_template=None`, which raises at runtime.
        nonlocal message_apply_fn, message_apply_name, message_template_checked
        if message_template_checked:
            return message_apply_fn, message_apply_name

        message_template_checked = True

        if tokenizer is None:
            return None, None

        apply_fn = getattr(tokenizer, "apply_template", None)
        if callable(apply_fn):
            message_apply_fn = apply_fn
            message_apply_name = "apply_template"
            return message_apply_fn, message_apply_name

        apply_chat_fn = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_fn):
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template is None:
                get_chat_template = getattr(tokenizer, "get_chat_template", None)
                if callable(get_chat_template):
                    try:
                        chat_template = get_chat_template(None, None)
                    except Exception:
                        chat_template = None

            if chat_template is not None:
                message_apply_fn = apply_chat_fn
                message_apply_name = "apply_chat_template"
                return message_apply_fn, message_apply_name

        return None, None

    def _to_2d_long_tensor(value: Any, name: str, idx: int) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value, dtype=torch.long)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Quantize: failed to convert `{name}` to tensor for calibration item {idx}.") from exc

        if tensor.ndim == 0:
            raise ValueError(f"Quantize: `{name}` for calibration item {idx} must be 1D or 2D, got scalar.")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2 and name == "attention_mask":
            # Some tokenizers emit causal masks shaped like [B, 1, T, T] or [B, T, T].
            # Collapse those higher-rank masks back to the token presence mask expected here.
            tensor = tensor.ne(0)
            for dim in range(tensor.ndim - 2, 0, -1):
                tensor = tensor.any(dim=dim)
            tensor = tensor.to(torch.long)
        elif tensor.ndim != 2:
            raise ValueError(
                f"Quantize: `{name}` for calibration item {idx} must be rank 1 or 2, got rank {tensor.ndim}."
            )
        return tensor

    def _normalize_attention_mask(mask_value: Any, ids_tensor: torch.Tensor, idx: int) -> torch.Tensor:
        try:
            mask_tensor = torch.as_tensor(mask_value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Quantize: failed to convert `attention_mask` to tensor for calibration item {idx}."
            ) from exc

        if mask_tensor.ndim == 0:
            raise ValueError(
                f"Quantize: `attention_mask` for calibration item {idx} must be rank 1 or higher, got scalar."
            )
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)

        try:
            keep_mask = normalize_seq_mask(mask_tensor, seq_len=ids_tensor.shape[-1])
        except ValueError as exc:
            raise ValueError(
                f"Quantize: failed to normalize `attention_mask` for calibration item {idx}: {exc}"
            ) from exc

        mask_tensor = keep_mask.to(dtype=torch.long)
        if mask_tensor.shape != ids_tensor.shape:
            if mask_tensor.numel() == ids_tensor.numel():
                mask_tensor = mask_tensor.reshape(ids_tensor.shape)
            else:
                raise ValueError(
                    f"Quantize: attention_mask shape {tuple(mask_tensor.shape)} does not match input_ids shape "
                    f"{tuple(ids_tensor.shape)} for calibration item {idx}."
                )
        return mask_tensor

    def _pack_ids(
        ids_value: Any,
        mask_value: Any,
        idx: int,
        chat_template_mask_value: Any = None,
    ) -> Dict[str, torch.Tensor]:
        ids_tensor = _to_2d_long_tensor(ids_value, "input_ids", idx)

        if mask_value is None:
            mask_tensor = torch.ones_like(ids_tensor, dtype=torch.long)
        else:
            mask_tensor = _normalize_attention_mask(mask_value, ids_tensor, idx)

        packed = {
            "input_ids": ids_tensor.detach(),
            "attention_mask": mask_tensor.detach(),
        }
        if chat_template_mask_value is not None:
            template_mask = _normalize_attention_mask(chat_template_mask_value, ids_tensor, idx).bool()
            packed["chat_template_mask"] = template_mask.detach()
        return packed

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
        nonlocal message_examples, message_template_name
        _require_tokenizer("calibration data uses the `messages` feature")
        apply_fn, template_name = _get_message_template()
        if apply_fn is None:
            raise ValueError(
                "tokenizer must expose `apply_template` or `apply_chat_template` to handle `messages` calibration data."
            )
        try:
            if template_name == "apply_chat_template":
                templated = apply_fn(messages_value, tokenize=False, add_generation_prompt=False)
            else:
                templated = apply_fn(messages_value, tokenize=False)
        except TypeError:
            templated = apply_fn(messages_value)

        if templated is None:
            raise ValueError(f"tokenizer.apply_template returned None for calibration item {idx}.")

        message_examples += 1
        message_template_name = template_name

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
            if not chat_template_weighting:
                return _tokenize_text_value(templated, idx)
            content_spans = []
            search_start = 0
            for message in messages_value:
                content = str(message.get("content", ""))
                if not content:
                    continue
                start = templated.find(content, search_start)
                if start < 0:
                    matcher = SequenceMatcher(None, content, templated[search_start:], autojunk=False)
                    blocks = [block for block in matcher.get_matching_blocks() if block.size >= 3]
                    if not blocks:
                        raise ValueError(
                            "chat-template weighting could not align message content to the rendered template"
                        )
                    for block in blocks:
                        content_spans.append((search_start + block.b, search_start + block.b + block.size))
                    stop = max(end for _, end in content_spans)
                else:
                    content_spans.append((start, start + len(content)))
                    stop = start + len(content)
                search_start = stop
            if not content_spans:
                raise ValueError("chat-template weighting requires at least one non-empty message content span")
            encoded = tokenizer(
                templated,
                # Match the unweighted chat-template path exactly. Weighting
                # may add provenance metadata, but must never alter the token
                # sequence being calibrated.
                add_special_tokens=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping")
            if offsets is None:
                raise ValueError("chat-template weighting requires tokenizer offset provenance")
            offset_rows = offsets.tolist() if isinstance(offsets, torch.Tensor) else offsets
            offset_row = offset_rows[0] if offset_rows and isinstance(offset_rows[0], list) else offset_rows
            content_mask = [
                any(start < int(end) and int(begin) < stop for start, stop in content_spans)
                for begin, end in offset_row
            ]
            template_mask = [[not value for value in content_mask]]
            return _pack_ids(encoded["input_ids"], encoded.get("attention_mask"), idx, template_mask)

        raise ValueError(
            f"tokenizer.apply_template returned unsupported type {type(templated)} for calibration item {idx}."
        )

    processed_examples: List[Dict[str, torch.Tensor]] = []

    def _append_processed(packed: Dict[str, torch.Tensor], raw_example: Any, idx: int) -> None:
        if source_weight_map:
            if not isinstance(raw_example, dict) or source_weight_column not in raw_example:
                raise ValueError(
                    f"Quantize: calibration item {idx} must provide source column {source_weight_column!r}"
                )
            source_name = str(raw_example[source_weight_column])
            if source_name not in source_weight_map:
                raise ValueError(
                    f"Quantize: calibration item {idx} has unmapped source {source_name!r}; "
                    f"configured sources are {sorted(source_weight_map)}"
                )
            rows = int(packed["input_ids"].shape[0])
            packed["fisher_sequence_weight"] = torch.full(
                (rows,),
                float(source_weight_map[source_name]),
                dtype=torch.float64,
            )
        elif isinstance(raw_example, dict) and "fisher_sequence_weight" in raw_example:
            weights = torch.as_tensor(raw_example["fisher_sequence_weight"], dtype=torch.float64).reshape(-1)
            if weights.numel() != packed["input_ids"].shape[0]:
                raise ValueError("fisher_sequence_weight must provide one scalar per calibration row")
            if not torch.isfinite(weights).all() or not weights.gt(0).all():
                raise ValueError("fisher_sequence_weight must be finite and positive")
            packed["fisher_sequence_weight"] = weights.detach().clone()
        processed_examples.append(packed)

    for idx, example in enumerate(raw_examples):
        if isinstance(example, dict):
            if "messages" in example:
                apply_fn, _ = _get_message_template()
                if apply_fn is None:
                    if "text" in example:
                        message_text_fallback_examples += 1
                        _append_processed(_tokenize_text_value(example["text"], idx), example, idx)
                        continue
                    raise ValueError(
                        "tokenizer must expose `apply_template` or `apply_chat_template`, or calibration data must "
                        "provide `text` when using `messages`."
                    )
                _append_processed(_tokenize_messages_value(example["messages"], idx), example, idx)
                continue
            if "text" in example:
                _append_processed(_tokenize_text_value(example["text"], idx), example, idx)
                continue
            if "input_ids" in example:
                _append_processed(
                    _pack_ids(example["input_ids"], example.get("attention_mask"), idx),
                    example,
                    idx,
                )
                continue
            raise ValueError(
                f"Quantize: unsupported calibration example structure at index {idx}: keys={list(example.keys())}"
            )

        if isinstance(example, str):
            _append_processed(_tokenize_text_value(example, idx), example, idx)
            continue

        if isinstance(example, (list, tuple)):
            if all(isinstance(x, int) for x in example):
                _append_processed(_pack_ids(list(example), None, idx), example, idx)
                continue
            raise ValueError(
                f"Quantize: list-based calibration example at index {idx} must contain only integers."
            )

        if torch.is_tensor(example):
            _append_processed(_pack_ids(example, None, idx), example, idx)
            continue

        try:
            _append_processed(_pack_ids(example, None, idx), example, idx)
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

    padding_side = getattr(tokenizer, "padding_side", "right") if tokenizer is not None else "right"
    if padding_side not in ("left", "right"):
        raise ValueError(
            f"Unsupported tokenizer.padding_side `{padding_side}`. Expected `left` or `right`."
        )

    if tokenizer is None:
        pad_token_id = 0
    else:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            # Reusing EOS storage is safe because the attention mask remains the
            # sole source of padding truth. Do not mutate tokenizer state here;
            # reusable tokenizer normalization remains Tokenicer's ownership.
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(pad_token_id, int):
            raise ValueError("Quantize: tokenizer must define an integer pad_token_id or eos_token_id.")

    for example in calibration_dataset:
        input_ids = _convert_tensor_to_list(example["input_ids"])
        attention_mask = _convert_tensor_to_list(example["attention_mask"])
        fisher_weights = example.get("fisher_sequence_weight")
        if fisher_weights is not None:
            fisher_weights = torch.as_tensor(fisher_weights, dtype=torch.float64).reshape(-1).tolist()
            if len(fisher_weights) != len(input_ids):
                raise ValueError("fisher_sequence_weight must provide one scalar per calibration row")

        # Normalize every logical sequence into its own calibration row. Later
        # sorting, filtering, concatenation, and batching must never use only the
        # first row of a packed tokenizer result or treat masked width as data.
        for row_index, (row_ids, row_mask) in enumerate(zip(input_ids, attention_mask)):
            row_len = len(row_ids)
            if max_positions is not None and row_len > max_positions:
                trimmed_row_count += 1
                longest_trimmed_row = max(longest_trimmed_row, row_len)
                if padding_side == "left":
                    row_ids = row_ids[-max_positions:]
                    row_mask = row_mask[-max_positions:]
                else:
                    row_ids = row_ids[:max_positions]
                    row_mask = row_mask[:max_positions]

            valid_token_count = sum(bool(value) for value in row_mask)
            if valid_token_count <= calibration_data_min_length:
                too_short_calibration_data_count += 1
                continue

            normalized = {
                "input_ids": [row_ids],
                "attention_mask": [row_mask],
            }
            if fisher_weights is not None:
                normalized["fisher_sequence_weight"] = [float(fisher_weights[row_index])]
            if "chat_template_mask" in example:
                template_row = example["chat_template_mask"]
                if isinstance(template_row, torch.Tensor):
                    template_row = template_row.tolist()
                template_row = template_row[0] if template_row and isinstance(template_row[0], list) else template_row
                if max_positions is not None and len(template_row) > max_positions:
                    if padding_side == "left":
                        template_row = template_row[-max_positions:]
                    else:
                        template_row = template_row[:max_positions]
                if len(template_row) != len(row_ids):
                    raise ValueError("chat_template_mask must remain aligned with input_ids during preparation")
                normalized["chat_template_mask"] = [template_row]
            new_calibration_dataset.append(normalized)

    if too_short_calibration_data_count > 0:
        log.warn(
            f"Quantize: {too_short_calibration_data_count} input rows with valid-token count <= "
            f"{calibration_data_min_length} were removed. "
            f"Use quantize(calibration_data_min_length={calibration_data_min_length}) to set a custom minimum length."
        )

    if not new_calibration_dataset:
        raise ValueError("Quantize: calibration dataset has no usable rows after valid-token filtering.")

    if message_examples > 0 and message_template_name is not None:
        log.info(
            "Calibration: tokenized %s `messages` examples via tokenizer.%s",
            message_examples,
            message_template_name,
        )
    if message_text_fallback_examples > 0:
        log.warn(
            "Calibration: fell back to raw `text` for %s `messages` examples because the tokenizer has no message "
            "template configured.",
            message_text_fallback_examples,
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
        if source_weight_map:
            raise ValueError("source-weighted Fisher calibration requires concat_size=0")
        if chat_template_weighting:
            raise ValueError("chat-template weighting currently requires concat_size=0 to preserve provenance")
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
            source_ids = example["input_ids"][0]
            source_mask = example["attention_mask"][0]
            # Concatenation changes sequence geometry deliberately. Compact only
            # positions selected by the mask so pre-existing padding cannot
            # consume the concat budget or separate otherwise adjacent tokens.
            row_ids = [token_id for token_id, keep in zip(source_ids, source_mask) if keep]
            row_mask = [1] * len(row_ids)
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
                if padding_side == "left":
                    input_ids_buff = ([pad_token_id] * padding_length) + input_ids_buff
                    attention_mask_buff = ([0] * padding_length) + attention_mask_buff
                else:
                    input_ids_buff.extend([pad_token_id] * padding_length)
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
            key=lambda item: sum(bool(value) for value in item["attention_mask"][0]),
        )
    elif calibration_dataset_sort == "desc":
        log.info("Calibration: Sort in descending order by length")
        sorted_dataset = sorted(
            new_calibration_dataset,
            key=lambda item: sum(bool(value) for value in item["attention_mask"][0]),
            reverse=True,
        )
    elif calibration_dataset_sort == "shuffle":
        log.info("Calibration: Sort by random shuffle")
        sorted_dataset = new_calibration_dataset[:]
        random.shuffle(sorted_dataset)
    else:
        log.info("Calibration: Native order")
        sorted_dataset = new_calibration_dataset

    preview_count = max(0, int(os.getenv("GPTQMODEL_LOG_CALIBRATION_SAMPLES", "0") or 0))
    if preview_count > 0:
        # Preview the exact token rows that will be batched for quantization.
        for idx, example in enumerate(sorted_dataset[:preview_count], start=1):
            row_ids = example["input_ids"][0]
            preview = ""
            if tokenizer is not None:
                try:
                    preview = tokenizer.decode(row_ids[:128], skip_special_tokens=False).replace("\n", " ")
                except Exception:
                    preview = ""
            log.info(
                "Calibration sample %s/%s: tokens=%s preview=%r",
                idx,
                min(preview_count, len(sorted_dataset)),
                len(row_ids),
                preview[:240],
            )

    if support_batch_quantize:
        new_calibration_dataset_batched = [
            collate_data(
                sorted_dataset[start : start + batch_size],
                pad_token_id,
                padding_side=getattr(tokenizer, "padding_side", "right"),
            )
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
        new_calibration_dataset_batched = []
        for block in sorted_dataset:
            row_ids = block["input_ids"][0]
            row_mask = block["attention_mask"][0]
            compact_ids = [token_id for token_id, keep in zip(row_ids, row_mask) if keep]
            new_calibration_dataset_batched.append(
                {
                    "input_ids": torch.tensor([compact_ids], dtype=torch.long),
                }
            )

    return new_calibration_dataset_batched


__all__ = ["batched", "prepare_calibration_dataset"]
