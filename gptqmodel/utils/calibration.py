# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utilities for preparing calibration datasets used during quantization."""

from __future__ import annotations

import json
import numbers
import os
import random
from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import torch

from .attn_mask import normalize_seq_mask
from .data import collate_data
from .logger import setup_logger


_HF_DATASET_TYPES: Optional[tuple] = None
_HF_DATASET_ATTRS = ("HFDataset", "HFIterableDataset")


def _hf_dataset_types() -> tuple:
    """Return the HF ``datasets`` classes, or ``()`` when it is not installed.

    Resolved on first use: importing ``datasets`` drags in ``pyarrow`` and
    ``pandas`` (about 0.4 s of process startup) and the classes are only needed
    to recognise HF dataset objects handed in as calibration input.
    """

    global _HF_DATASET_TYPES
    if _HF_DATASET_TYPES is None:
        try:
            from datasets import Dataset, IterableDataset
        except Exception:  # pragma: no cover - optional dependency
            _HF_DATASET_TYPES = ()
        else:
            _HF_DATASET_TYPES = (Dataset, IterableDataset)
    return _HF_DATASET_TYPES


def __getattr__(name: str):
    """Keep ``HFDataset``/``HFIterableDataset`` importable without eager import."""

    # Answer unknown names (the import system probes `__path__` on every
    # submodule import) before touching the optional dependency.
    if name not in _HF_DATASET_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    types = _hf_dataset_types()
    if not types:
        return None
    return types[0] if name == "HFDataset" else types[1]


# `HFDataset`/`HFIterableDataset` are forward references resolved lazily by
# `__getattr__` above, so annotating does not import the optional dependency.
CalibrationInputType = Union[
    List[Dict[str, Union[List[int], torch.LongTensor]]],
    List[str],
    List[List[int]],
    Mapping[str, Any],
    "HFDataset",  # type: ignore[type-arg]  # noqa: F821
    "HFIterableDataset",  # type: ignore[type-arg]  # noqa: F821
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
    """Normalize text-like calibration rows to processor chat conversations.

    Multimodal processors expect one conversation per calibration sample, while
    the public quantization API also accepts raw text and common dataset rows.
    Already structured conversations are returned unchanged.
    """

    if isinstance(sample, dict):
        if "messages" in sample:
            return sample["messages"]
        if isinstance(sample.get("text"), str):
            sample = sample["text"]

    if isinstance(sample, str):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": sample}],
            }
        ]

    return sample


def batched_conversations(iterable, batch_size: int, process_func=None):
    """Yield chat-normalized calibration samples in fixed-size batches."""

    def normalize_and_process(sample):
        sample = normalize_chat_calibration_sample(sample)
        return process_func(sample) if process_func is not None else sample

    yield from batched(iterable, batch_size, process_func=normalize_and_process)


def _iter_calibration_rows(dataset):
    """Yield every ``(input_ids, attention_mask)`` row of a normalized dataset.

    Items may hold one row or several (2-D ``input_ids``). The concatenation
    path must consume all of them, otherwise every row after the first is
    silently dropped from the Hessian.
    """

    for example in dataset:
        yield from zip(example["input_ids"], example["attention_mask"])


_CALIBRATION_COLUMNS = ("input_ids", "attention_mask", "text", "messages")

# Extra chat-template inputs a `messages` row may carry. The standard HF layout
# keeps the tool schemas in `tools` (and RAG templates read `documents`); a
# columnar mapping must carry them into every row like a plain row does (#4858).
_TEMPLATE_INPUT_COLUMNS = ("tools", "documents")

# Calibration input kinds a model definition may declare. The default is the
# token/text path every existing definition uses; `tensor_kwargs` is the opt-in
# path for models whose forward takes named tensors (Decision Transformer).
CALIBRATION_INPUT_TOKEN_IDS = "token_ids"
CALIBRATION_INPUT_TENSOR_KWARGS = "tensor_kwargs"
_CALIBRATION_INPUT_KINDS = (CALIBRATION_INPUT_TOKEN_IDS, CALIBRATION_INPUT_TENSOR_KWARGS)


def calibration_input_kind(qmodel) -> str:
    """Return the calibration input kind declared by a model definition.

    Only a definition that opts in with ``calibration_inputs = "tensor_kwargs"``
    changes behavior; every other model keeps the token-id path.
    """

    kind = getattr(qmodel, "calibration_inputs", CALIBRATION_INPUT_TOKEN_IDS)
    if kind not in _CALIBRATION_INPUT_KINDS:
        raise ValueError(
            f"Unsupported `calibration_inputs` {kind!r} on {type(qmodel).__name__}. "
            f"Expected one of {_CALIBRATION_INPUT_KINDS}."
        )
    return kind


def _is_tensor_kwargs_row(value: Any) -> bool:
    """Return whether ``value`` is a non-empty mapping of tensor values."""

    return (
        isinstance(value, Mapping)
        and len(value) > 0
        and all(torch.is_tensor(item) for item in value.values())
    )


def _prepare_tensor_kwargs_dataset(
    raw_examples: List[Any],
    *,
    calibration_dataset_concat_size: Optional[int],
    batch_size: int,
    log,
) -> List[Dict[str, torch.Tensor]]:
    """Validate tensor-only calibration rows and return them as batches.

    Each row already carries the model's own batch dimension, so it is one
    calibration batch. Token-level policies (minimum length, concatenation,
    length sorting, pad collation) do not apply to named tensors.
    """

    if calibration_dataset_concat_size:
        raise ValueError(
            "Quantize: `calibration_dataset_concat_size` is token-based and cannot be used "
            "with a model that declares `calibration_inputs = 'tensor_kwargs'`."
        )

    batches: List[Dict[str, torch.Tensor]] = []
    for idx, example in enumerate(raw_examples):
        if not _is_tensor_kwargs_row(example):
            if not isinstance(example, Mapping) or len(example) == 0:
                raise ValueError(
                    "Quantize: tensor calibration example at index "
                    f"{idx} must be a non-empty mapping of named tensors, got "
                    f"{type(example).__name__}."
                )
            invalid = [
                key for key, item in example.items() if not torch.is_tensor(item)
            ]
            raise ValueError(
                f"Quantize: tensor calibration example at index {idx} must hold only "
                f"torch tensors; keys {invalid} are not tensors."
            )

        batch: Dict[str, torch.Tensor] = {}
        for key, value in example.items():
            if value.ndim == 0:
                raise ValueError(
                    f"Quantize: tensor calibration example at index {idx} key `{key}` "
                    "must be at least 1-D (include the batch dimension)."
                )
            batch[key] = value.detach()
        batches.append(batch)

    if not batches:
        raise ValueError("Quantize: calibration dataset is empty.")

    if batch_size != 1:
        log.info(
            "Calibration: tensor-kwargs rows are passed through one row per batch; "
            "`batch_size=%s` is ignored.",
            batch_size,
        )

    return batches


def _column_entries(value: Any) -> Optional[List[Any]]:
    """Return a column's per-row entries as a Python list, else ``None``.

    ``Dataset.with_format("numpy")[:]`` stores a ragged token column as a 1-D
    ``ndarray(dtype=object)`` whose elements are arrays, and a pandas
    ``Series`` is not a ``Sequence`` and does not index positionally through
    ``[]``. Both hold one entry per calibration row, so resolve their entries
    here instead of classifying them by ``ndim`` or ``isinstance`` alone.

    ``None`` means the value is not a container of rows (a string, a scalar, a
    0-D array, ...). A 1-D numeric array/tensor is returned as its scalar
    entries; the caller decides whether those scalars make one token row or a
    column of rows.
    """

    if isinstance(value, (str, bytes, bytearray)):
        return None
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        if ndim == 0:
            return None
        if ndim >= 2:
            # Row-major 2-D array/tensor: one entry per row.
            return [value[index] for index in range(len(value))]
    if isinstance(value, (list, tuple)):
        return list(value)
    if ndim == 1:
        # A 1-D ``ndarray``/``Series``: ``tolist()`` unwraps an object column
        # element-wise without converting the elements themselves.
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            entries = tolist()
            if isinstance(entries, list):
                return entries
    if isinstance(value, Sequence):
        # ``datasets.arrow_dataset.Column`` is a ``Sequence`` without
        # ``ndim``/``tolist``.
        return list(value)
    return None


def _hf_dataset_rows(dataset: Any) -> Optional[List[Any]]:
    """Return an HF dataset's rows as python mappings, or ``None`` if not HF.

    A caller often leaves a display format on the dataset for another pipeline
    step. Iteration then yields one-row ``pandas.DataFrame`` or
    ``pyarrow.Table`` objects instead of the documented python rows, and every
    row was rejected purely because of that format (issue #4882). Clearing the
    format keeps the same rows, so the rejection depends on the content only.
    """

    hf_dataset_types = _hf_dataset_types()
    if not hf_dataset_types or not isinstance(dataset, hf_dataset_types):
        return None
    reset_format = getattr(dataset, "with_format", None)
    return list(reset_format(None) if callable(reset_format) else dataset)


def _frame_columns(dataset: Any) -> Optional[Mapping[str, Any]]:
    """Return a DataFrame-like input as a columnar mapping, else ``None``.

    A ``pandas.DataFrame`` is not a ``Mapping``, so iterating it yields its
    column *labels*: a text-only frame calibrated on the literal string
    ``"text"``, and an ``input_ids`` frame failed with a misleading
    minimum-length error. A table that stores one entry per row per column is
    therefore converted to the columnar mapping the rest of this module
    understands. ``None`` means the object is not table-like, so the existing
    handling applies unchanged.
    """

    if isinstance(dataset, Mapping):
        return None
    if getattr(dataset, "columns", None) is None:
        return None
    to_dict = getattr(dataset, "to_dict", None)
    if callable(to_dict):
        # pandas spells the orientation as a positional argument or ``orient=``;
        # Polars uses ``as_series=False``. A `to_dict` that understands none of
        # them is not a row-wise table, so let the iterable handling take it.
        for args, kwargs in (
            (("list",), {}),
            ((), {"orient": "list"}),
            ((), {"as_series": False}),
        ):
            try:
                mapped = to_dict(*args, **kwargs)
            except TypeError:
                continue
            if isinstance(mapped, Mapping):
                break
        else:
            return None
    else:
        # ``pyarrow.Table`` (what ``Dataset.with_format("arrow")[:]`` returns)
        # has ``columns`` but no ``to_dict``, so it used to fall through to the
        # iterable handling, where iteration yields one ``ChunkedArray`` per
        # column and every row is rejected.
        to_pydict = getattr(dataset, "to_pydict", None)
        if not callable(to_pydict):
            return None
        try:
            mapped = to_pydict()
        except Exception:
            return None
        if not isinstance(mapped, Mapping):
            return None

    columns: Dict[str, Any] = {}
    for name, value in mapped.items():
        if callable(getattr(value, "to_list", None)):
            value = value.to_list()
        if not isinstance(value, (list, tuple)):
            # A column that is not a per-row sequence means this is not a
            # row-wise table after all.
            return None
        columns[name] = value
    return columns


def _unwrap_object_array(value: Any) -> Any:
    """Return an object-dtype ``ndarray`` as a Python list.

    ``torch.as_tensor`` refuses ``dtype=object`` arrays even when every element
    is an integer token id, which is exactly what a NumPy-formatted HF dataset
    yields per row. Unwrapping the elements lets the existing integer/dtype
    checks classify them; anything else (nested arrays, strings) keeps failing
    with the same structural error as before.
    """

    if getattr(getattr(value, "dtype", None), "kind", None) != "O":
        return value
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        return value
    return tolist()


def _is_single_tool_definition(value: Any) -> bool:
    """Return True for one tool schema rather than a list of them.

    The marker keys below belong to a tool definition, not to a parameters
    schema (`{"type": "object", ...}`), so a parameters object is not mistaken
    for a tool.
    """

    return isinstance(value, Mapping) and (
        "function" in value or value.get("type") == "function"
    )


def _normalize_template_input_value(value: Any) -> Any:
    """Return a ``tools``/``documents`` value a chat template can render.

    Function-calling datasets on the Hub store these columns in two layouts
    that ``apply_chat_template`` cannot consume verbatim (#4934):

    * a JSON string, because Arrow cannot hold a different struct schema per
      row. The template rejects the string with ``ValueError``, and the row is
      then rendered without the schema -- the #4858 symptom;
    * a list of dicts read from an Arrow ``Dataset``. Arrow unifies the struct
      schema across rows, so every row's schema gains the other rows'
      parameter names with a ``None`` value, e.g.
      ``"properties": {"city": {...}, "tz": null}``. The rendered tool block
      then silently differs from what inference sends.

    Decode a JSON string and drop dict entries whose value is ``None``. A real
    JSON-schema ``null`` is the string ``"null"`` (``{"type": "null"}``), so it
    survives; a ``None`` element inside a list is a legitimate JSON value and
    is kept as well. The walk copies every container, so the caller's rows are
    never mutated.
    """

    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:  # pragma: no cover - non-UTF-8 binary column
            return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            return value
        # Only a container is a schema or a document list. A document whose
        # text happens to read as a JSON scalar stays text.
        if not isinstance(decoded, (Mapping, list, tuple)):
            return value
        return _normalize_template_input_value(decoded)
    # An object-dtype ndarray (`dataset.with_format("numpy")[:]`) holds one
    # entry per row; unwrap it so the walk sees python containers.
    value = _unwrap_object_array(value)
    if isinstance(value, Mapping):
        return {
            key: _normalize_template_input_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_template_input_value(item) for item in value]
    return value


def _normalize_template_input_payload(value: Any, *, key: str) -> Any:
    """Normalize a template input and present `tools` as the list templates take.

    `apply_chat_template(tools=...)` iterates its argument, so a one-tool
    dataset whose column holds a single schema object (the layout a JSON-string
    round trip yields) is rejected outright. `documents` is passed through.
    """

    normalized = _normalize_template_input_value(value)
    if key == "tools" and _is_single_tool_definition(normalized):
        return [normalized]
    return normalized


def _normalize_chat_messages(value: Any) -> Any:
    """Return a conversation a chat template renders like the plain Python rows.

    The `messages` column of the same function-calling rows #4934 normalized
    keeps Arrow's unified struct schema (#5082). Arrow stores one struct type
    per column, so every turn gains a ``None`` value for each key another turn
    uses -- ``tool_calls`` on user/tool/plain-assistant turns, ``name`` or
    ``tool_call_id`` where only some rows carry them -- and every tool call's
    ``arguments`` gains the other rows' argument names. The template then sees
    a different conversation than the plain Python rows do:

    * Qwen2.5/Qwen3 render ``{"city": "Paris", "query": null}`` instead of
      ``{"city": "Paris"}``, so every tool-call row is calibrated on tokens
      inference never emits;
    * Llama-3.x tests ``'tool_calls' in message``, so the injected
      ``tool_calls: None`` turns a user turn into a tool call and the template
      raises ``TypeError: object of type 'NoneType' has no len()``.

    Drop dict entries whose value is ``None``, recursively. Two values are not
    Arrow padding and are kept:

    * ``content: None`` is how an OpenAI-style tool-call turn spells "no text",
      and templates such as DeepSeek-R1 test it with ``message.content is
      none``;
    * ``arguments`` stored as a JSON string is inserted verbatim by raw-text
      templates (DeepSeek-R1 writes it into a ``json`` block), so it stays a
      string rather than being decoded into a mapping.

    A ``None`` element inside a list is a legitimate JSON value and is kept as
    well. An explicit ``None`` argument value is indistinguishable from Arrow's
    padding for the same key, so it is dropped too -- the same rule #4934
    applies to a ``tools`` schema's ``properties``. The walk copies every
    container, so the caller's rows are never mutated.
    """

    # An object-dtype ndarray (`dataset.with_format("numpy")[:]`) holds one
    # entry per row; unwrap it so the walk sees python containers.
    value = _unwrap_object_array(value)
    if isinstance(value, Mapping):
        return {
            key: _normalize_chat_messages(item)
            for key, item in value.items()
            if item is not None or key == "content"
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_chat_messages(item) for item in value]
    return value


def _is_batched_column(value: Any) -> bool:
    """Return True when ``value`` holds one entry per calibration row."""

    if isinstance(value, (str, bytes, bytearray)):
        return False
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim >= 2:
        return True
    entries = _column_entries(value)
    if not entries:
        return False
    first = entries[0]
    first_ndim = getattr(first, "ndim", None)
    if first_ndim is not None:
        # A 1-D object column of arrays/lists is a column of rows; a column of
        # 0-D scalars is one token row.
        return first_ndim >= 1
    # A nested sequence is a column of rows; a sequence of scalars is one token
    # row. This also covers ``datasets.arrow_dataset.Column``.
    return isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray))


def _is_conversation_entry(value: Any) -> bool:
    """Return True when a ``messages`` column entry is itself a conversation.

    A single row stores one conversation as a list of role mappings, so its
    entries start with a mapping. A column stores one conversation per row, so
    its entries start with a container: a list/tuple, or the ``ndarray``/
    ``Series`` row that ``with_format("numpy"|"pandas")[:]`` produces (#4936).
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)) or torch.is_tensor(value):
        return False
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim >= 1:
        return True
    return isinstance(value, Sequence)


def _is_columnar_mapping(mapping: Mapping) -> bool:
    """Return True when a top-level mapping stores columns instead of one row.

    ``tokenizer(texts)`` returns a ``BatchEncoding`` whose ``input_ids`` is 2-D;
    a dict of columns behaves the same way. A mapping describing a single
    example instead carries a 1-D ``input_ids``, one ``text`` string, or one
    ``messages`` conversation.
    """

    if "input_ids" in mapping:
        return _is_batched_column(mapping["input_ids"])
    if "text" in mapping:
        return _column_entries(mapping["text"]) is not None
    if "messages" in mapping:
        entries = _column_entries(mapping["messages"])
        if not entries:
            return False
        return _is_conversation_entry(entries[0])
    return False


def calibration_input_row_count(dataset: Any) -> Optional[int]:
    """Return the number of calibration rows, or ``None`` when unknown.

    A columnar mapping stores one entry per *column*, so ``len(mapping)`` is the
    number of columns (2 for ``input_ids``/``attention_mask``), not the number
    of calibration rows. Report the primary column's length instead so callers
    such as the size warning count rows.
    """

    if isinstance(dataset, Mapping):
        if _is_columnar_mapping(dataset):
            for name in _CALIBRATION_COLUMNS:
                if name in dataset and dataset[name] is not None:
                    try:
                        return len(dataset[name])
                    except TypeError:
                        return None
            return None
        if any(name in dataset for name in _CALIBRATION_COLUMNS):
            # A mapping that is not columnar describes one calibration row.
            return 1
    try:
        return len(dataset)
    except TypeError:
        return None


def _is_dataset_split_mapping(mapping: Mapping) -> bool:
    """Return True for a ``DatasetDict``-style mapping of splits to datasets."""

    hf_dataset_types = _hf_dataset_types()
    if not hf_dataset_types or not mapping:
        return False
    return all(isinstance(value, hf_dataset_types) for value in mapping.values())


def _to_python_containers(value: Any) -> Any:
    """Return ``value`` with NumPy/pandas containers unwrapped to Python ones.

    A ``Dataset.with_format("numpy")[:]`` row stores ``messages``/``tools`` as
    ``ndarray`` objects (a 1-D object array of conversations, or a 2-D object
    array of turns when every conversation has the same length) and pandas
    stores them as a ``Series`` of arrays. A chat template needs plain lists of
    dicts: handed an ``ndarray`` it renders the template's default system turn
    and silently drops the conversation (#4936). Copy the row into plain
    ``list``/``dict``/``str``/number containers so every input shape renders
    identically.

    Torch tensors are returned unchanged: the token-id path consumes them and
    converting them to lists would change how they are classified. The copy is
    recursive and never mutates the caller's dataset.
    """

    if torch.is_tensor(value) or isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, Mapping):
        return {key: _to_python_containers(item) for key, item in value.items()}

    # A NumPy structured array (record array) or one of its ``np.void`` rows
    # stores the columns by name; ``tolist()`` would turn each row into a bare
    # tuple and lose the column names.
    dtype_names = getattr(getattr(value, "dtype", None), "names", None)
    if dtype_names:
        if getattr(value, "ndim", 0) == 0:
            return {
                name: _to_python_containers(item)
                for name, item in zip(dtype_names, value.tolist())
            }
        return [_to_python_containers(row) for row in value]

    # A pandas ``DataFrame`` row (iterating a pandas-formatted dataset) maps
    # column names to one-element lists; keep it columnar so the caller's
    # row/column logic sees the same shape as ``to_dict``.
    if getattr(value, "columns", None) is not None:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                mapped = to_dict("list")
            except TypeError:
                mapped = None
            if isinstance(mapped, Mapping):
                return {key: _to_python_containers(item) for key, item in mapped.items()}

    # ``ndarray`` and pandas ``Series`` expose ``tolist``; a 0-D scalar's
    # ``tolist()`` returns a Python scalar. A Series of arrays keeps the arrays
    # in its list, hence the recursion.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            listed = tolist()
        except Exception:  # pragma: no cover - defensive
            listed = None
        if isinstance(listed, list):
            return [_to_python_containers(item) for item in listed]
        if listed is not None and not isinstance(listed, (tuple, Mapping)):
            return listed

    if isinstance(value, tuple):
        return tuple(_to_python_containers(item) for item in value)
    if isinstance(value, list):
        return [_to_python_containers(item) for item in value]

    # A ``pyarrow`` scalar (e.g. from a table that was not routed through
    # ``_frame_columns``) converts itself with ``as_py``.
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        try:
            return _to_python_containers(as_py())
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _columnar_mapping_examples(mapping: Mapping) -> List[Dict[str, Any]]:
    """Split a columnar calibration mapping into one mapping per row.

    Object-dtype NumPy columns and pandas ``Series`` columns are indexed through
    ``_column_entries`` so each row receives one element, not the whole column.
    ``tools``/``documents`` travel with the row so the chat template sees the
    same schema a plain row would pass (#4858, #4936).
    """

    primary = next(name for name in _CALIBRATION_COLUMNS if mapping.get(name) is not None)

    columns: Dict[str, Any] = {}
    for name in (*_CALIBRATION_COLUMNS, *_TEMPLATE_INPUT_COLUMNS):
        if name not in mapping or mapping[name] is None:
            continue
        entries = _column_entries(mapping[name])
        columns[name] = mapping[name] if entries is None else entries

    row_count = len(columns[primary])
    for name, column in columns.items():
        try:
            column_length = len(column)
        except TypeError:
            continue
        if column_length != row_count:
            raise ValueError(
                f"Quantize: calibration mapping column `{name}` has {column_length} rows, "
                f"expected {row_count} (same as `{primary}`)."
            )

    return [
        {name: column[row_index] for name, column in columns.items()}
        for row_index in range(row_count)
    ]


def _is_integer_scalar(value: Any) -> bool:
    """Return True for a scalar token id (Python, NumPy, or 0-D torch)."""

    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Integral):
        return True
    if torch.is_tensor(value):
        return (
            value.numel() == 1
            and value.dtype != torch.bool
            and not value.is_floating_point()
            and not value.is_complex()
        )
    return False


def _is_integer_sequence(value: Any) -> bool:
    """Return True when a nested list/tuple holds only scalar token ids."""

    if isinstance(value, (list, tuple)):
        return all(_is_integer_sequence(item) for item in value)
    if torch.is_tensor(value):
        return value.dtype != torch.bool and not value.is_floating_point() and not value.is_complex()
    return _is_integer_scalar(value)


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
    # Only a definition that opts into tensor kwargs changes the row handling
    # below; every text model keeps the token-id path unchanged.
    tensor_kwargs_inputs = calibration_input_kind(qmodel) == CALIBRATION_INPUT_TENSOR_KWARGS

    hf_dataset_types: tuple = _hf_dataset_types()

    if isinstance(calibration_dataset, str):
        raise ValueError("Quantize: calibration dataset must be iterable, not a single string.")

    frame_columns = _frame_columns(calibration_dataset)
    if frame_columns is not None:
        calibration_dataset = frame_columns

    if isinstance(calibration_dataset, Mapping):
        if len(calibration_dataset) == 0:
            raw_examples = []
        elif _is_columnar_mapping(calibration_dataset):
            raw_examples = _columnar_mapping_examples(calibration_dataset)
        elif _is_dataset_split_mapping(calibration_dataset):
            splits = ", ".join(repr(key) for key in calibration_dataset.keys())
            raise ValueError(
                "Quantize: calibration dataset is a mapping of dataset splits "
                f"({splits}); select one split, e.g. `calibration_dataset['train']`."
            )
        elif any(name in calibration_dataset for name in ("input_ids", "text", "messages")):
            raw_examples = [calibration_dataset]
        elif tensor_kwargs_inputs and _is_tensor_kwargs_row(calibration_dataset):
            # A bare mapping of named tensors is one tensor-kwargs example.
            raw_examples = [calibration_dataset]
        else:
            raise ValueError(
                "Quantize: calibration dataset mapping must provide `input_ids`, `text`, "
                f"or `messages` columns, got keys={list(calibration_dataset.keys())}."
            )
    elif hf_dataset_types and isinstance(calibration_dataset, hf_dataset_types):
        raw_examples = _hf_dataset_rows(calibration_dataset)
    elif isinstance(calibration_dataset, list):
        raw_examples = calibration_dataset
    elif isinstance(calibration_dataset, Sequence) and not isinstance(calibration_dataset, (bytes, bytearray)):
        raw_examples = list(calibration_dataset)
    else:
        raw_examples = list(calibration_dataset)

    if len(raw_examples) == 0:
        raise ValueError("Quantize: calibration dataset is empty.")

    if calibration_dataset_sort not in (None, "asc", "desc", "shuffle"):
        raise ValueError(
            f"Quantize: unsupported `calibration_dataset_sort` {calibration_dataset_sort!r}. "
            "Expected `asc`, `desc`, `shuffle`, or None."
        )

    if tensor_kwargs_inputs:
        # Named-tensor rows already carry the model's batch dimension and have no
        # token length, so the token-level normalization below does not apply.
        return _prepare_tensor_kwargs_dataset(
            raw_examples,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
            batch_size=batch_size,
            log=log,
        )

    # A numpy/pandas-formatted row keeps ``messages``/``tools`` in ``ndarray``/
    # ``Series`` objects that a chat template cannot render as a conversation
    # (#4936). Unwrap them once, for every input shape, before templating.
    raw_examples = [_to_python_containers(example) for example in raw_examples]

    message_examples = 0
    message_template_name = None
    dropped_template_inputs = False
    message_text_fallback_examples = 0
    ignored_source_examples = 0

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
            tensor = torch.as_tensor(_unwrap_object_array(value))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Quantize: failed to convert `{name}` to tensor for calibration item {idx}.") from exc

        # `torch.as_tensor([])` / `torch.as_tensor([[]])` infers float32 from the
        # empty Python container. That is an absent value, not a float token id,
        # so normalize empty containers to long before the integer check instead
        # of reporting a misleading dtype error.
        if tensor.numel() == 0 and not torch.is_tensor(value):
            tensor = tensor.to(dtype=torch.long)

        if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(
                f"Quantize: `{name}` for calibration item {idx} must contain integers, got dtype {tensor.dtype}."
            )
        tensor = tensor.to(dtype=torch.long)

        if tensor.ndim == 0:
            raise ValueError(f"Quantize: `{name}` for calibration item {idx} must be 1D or 2D, got scalar.")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"Quantize: `{name}` for calibration item {idx} must be rank 1 or 2, got rank {tensor.ndim}."
            )
        return tensor

    def _normalize_attention_mask(mask_value: Any, ids_tensor: torch.Tensor, idx: int) -> torch.Tensor:
        try:
            mask_tensor = torch.as_tensor(_unwrap_object_array(mask_value))
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

        # An additive/extended mask encodes "attend" as 0 (or a positive bias)
        # and "masked" as a negative bias. When such a mask is all zeros there
        # is no padding at all, but ``normalize_seq_mask`` reads 0 as "drop" for
        # a 0/1 mask and would erase every token. Resolve it here first so only
        # genuinely masked positions are dropped. A 2-D integer 0/1 mask keeps
        # the usual meaning: an all-zero row is all padding.
        if (
            mask_tensor.dtype != torch.bool
            and mask_tensor.numel() > 0
            and torch.count_nonzero(mask_tensor).item() == 0
            and (mask_tensor.is_floating_point() or mask_tensor.ndim >= 3)
        ):
            mask_tensor = torch.ones_like(mask_tensor)

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

    def _pack_ids(ids_value: Any, mask_value: Any, idx: int) -> Dict[str, torch.Tensor]:
        ids_tensor = _to_2d_long_tensor(ids_value, "input_ids", idx)

        # A zero-ROW item (`torch.empty(0, N)`) carries no sample at all and is a
        # structural error. A zero-COLUMN row (`(1, 0)`, e.g. tokenizing `""`
        # with a tokenizer that adds no BOS) is merely too short: it is dropped
        # by the minimum-length filter below, exactly as before #4194.
        if ids_tensor.shape[0] == 0:
            raise ValueError(
                f"Quantize: calibration item {idx} has empty `input_ids` with shape "
                f"{tuple(ids_tensor.shape)}; every item needs at least one row of tokens."
            )

        if mask_value is None:
            mask_tensor = torch.ones_like(ids_tensor, dtype=torch.long)
        else:
            mask_tensor = _normalize_attention_mask(mask_value, ids_tensor, idx)

        return {
            "input_ids": ids_tensor.detach(),
            "attention_mask": mask_tensor.detach(),
        }

    def _tokenize_text_value(
        text_value: Any, idx: int, *, add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        _require_tokenizer("calibration data contains raw text")
        tokenized = tokenizer(  # type: ignore[call-arg]
            text_value,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
        return _pack_ids(input_ids, attention_mask, idx)

    def _tokenize_messages_value(
        messages_value: Any, idx: int, row: Any = None
    ) -> Dict[str, torch.Tensor]:
        nonlocal message_examples, message_template_name, dropped_template_inputs
        _require_tokenizer("calibration data uses the `messages` feature")
        apply_fn, template_name = _get_message_template()
        if apply_fn is None:
            raise ValueError(
                "tokenizer must expose `apply_template` or `apply_chat_template` to handle `messages` calibration data."
            )
        # Arrow unifies the `messages` struct schema across rows and turns, so
        # a user turn carries `tool_calls: None` and every tool call's
        # `arguments` mapping gains the other rows' argument names (#5082).
        # Render the conversation inference actually sends, before templating.
        messages_value = _normalize_chat_messages(messages_value)
        # A chat row carries more than `messages`: the standard HF layout puts
        # the tool schemas in `tools` (and RAG templates read `documents`).
        # Dropping them rendered the row as if no tools were declared, so the
        # schema block the model sees at inference never reached the Hessian
        # (#4858).
        template_inputs = {}
        if isinstance(row, Mapping):
            for key in _TEMPLATE_INPUT_COLUMNS:
                value = row.get(key)
                if value is not None:
                    # HF storage layouts (JSON string, Arrow-unified schema with
                    # injected `null` properties) are normalized back to the
                    # payload inference sends (#4934).
                    template_inputs[key] = _normalize_template_input_payload(value, key=key)
        try:
            if template_name == "apply_chat_template":
                templated = apply_fn(
                    messages_value, tokenize=False, add_generation_prompt=False, **template_inputs
                )
            else:
                templated = apply_fn(messages_value, tokenize=False, **template_inputs)
        except TypeError:
            templated = apply_fn(messages_value)
        except Exception as exc:
            # A tokenizer whose template does not use `tools`/`documents`
            # rejects them (Jinja raises). Keep the row rather than failing the
            # run, but say so once: the schema is then missing from the tokens.
            if not template_inputs:
                raise
            if not dropped_template_inputs:
                dropped_template_inputs = True
                log.warn(
                    "Quantize: this tokenizer's chat template rejected the calibration row's "
                    f"{sorted(template_inputs)}; rendering without them: {type(exc).__name__}: {exc}"
                )
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
                # A rendered chat template already contains the template's own
                # special tokens (e.g. a leading BOS on Llama/Gemma). Letting the
                # tokenizer add them again prepends a duplicate BOS to every row.
                return _tokenize_text_value(text_value, idx, add_special_tokens=False)

        if isinstance(templated, (list, tuple)):
            if len(templated) > 0 and isinstance(templated[0], int):
                return _pack_ids(list(templated), None, idx)
            raise ValueError(
                f"tokenizer.apply_template returned an unsupported sequence type for calibration item {idx}."
            )

        if torch.is_tensor(templated):
            return _pack_ids(templated, None, idx)

        if isinstance(templated, str):
            return _tokenize_text_value(templated, idx, add_special_tokens=False)

        raise ValueError(
            f"tokenizer.apply_template returned unsupported type {type(templated)} for calibration item {idx}."
        )

    processed_examples: List[Dict[str, torch.Tensor]] = []
    for idx, example in enumerate(raw_examples):
        # Tokenizer outputs are `BatchEncoding`, a `UserDict` rather than a
        # builtin dict. Accept every mapping so the AutoGPTQ-style
        # `[tokenizer(text), ...]` calibration input reaches the input_ids path.
        if isinstance(example, Mapping):
            # Explicit token ids win over every source column. A row carrying
            # `input_ids` next to `text` or `messages` is the normal output of
            # `Dataset.map(tokenizer, truncation=True, ...)` and of
            # `dataset[:]` / `list(dataset)`, so re-tokenizing from the source
            # silently discarded the caller's truncation, special-token, and
            # chat choices and fed documents far past the requested length
            # (#4635 for `text`, #4670 for `messages`).
            if example.get("input_ids") is not None:
                if example.get("messages") is not None or example.get("text") is not None:
                    ignored_source_examples += 1
                processed_examples.append(_pack_ids(example["input_ids"], example.get("attention_mask"), idx))
                continue
            if "messages" in example:
                apply_fn, _ = _get_message_template()
                if apply_fn is None:
                    if "text" in example:
                        message_text_fallback_examples += 1
                        processed_examples.append(_tokenize_text_value(example["text"], idx))
                        continue
                    raise ValueError(
                        "tokenizer must expose `apply_template` or `apply_chat_template`, or calibration data must "
                        "provide `text` when using `messages`."
                    )
                processed_examples.append(_tokenize_messages_value(example["messages"], idx, example))
                continue
            if "text" in example:
                processed_examples.append(_tokenize_text_value(example["text"], idx))
                continue
            raise ValueError(
                f"Quantize: unsupported calibration example structure at index {idx}: keys={list(example.keys())}"
            )

        if isinstance(example, str):
            processed_examples.append(_tokenize_text_value(example, idx))
            continue

        if isinstance(example, (list, tuple)):
            if _is_integer_sequence(example):
                processed_examples.append(_pack_ids(list(example), None, idx))
                continue
            raise ValueError(f"Quantize: list-based calibration example at index {idx} must contain only integers.")

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
            return tensor.cpu().tolist()
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
    # Padding and truncation are independent tokenizer settings. Trimming a
    # row to the model's position budget must follow `truncation_side`
    # (HF default: right), not the batch padding direction: left padding used
    # to front-truncate every overlong row and drop its BOS/attention sink.
    truncation_side = getattr(tokenizer, "truncation_side", "right") if tokenizer is not None else "right"
    if truncation_side not in ("left", "right"):
        raise ValueError(
            f"Unsupported tokenizer.truncation_side `{truncation_side}`. Expected `left` or `right`."
        )

    # Concatenation packs real tokens into windows of exactly this width, so a
    # window can only fit the model's position budget when the width does. The
    # per-row position trim below is then unnecessary: it used to cut a long
    # document to the budget *before* windowing, discarding everything past it
    # even though every resulting window was far below the budget (#4892).
    concat_mode = bool(calibration_dataset_concat_size)
    if concat_mode and max_positions is not None and calibration_dataset_concat_size > max_positions:
        raise ValueError(
            f"Quantize: calibration_dataset_concat_size={calibration_dataset_concat_size} exceeds "
            f"{max_positions_source}={max_positions}, so a packed window could never fit the model's "
            "position budget. Lower `concat_size`."
        )

    dropped_padded_row_count = 0
    too_short_row_count = 0
    too_short_window_count = 0

    for example_index, example in enumerate(calibration_dataset):
        input_ids = _convert_tensor_to_list(example["input_ids"])
        attention_mask = _convert_tensor_to_list(example["attention_mask"])

        if len(input_ids) == 0:
            raise ValueError(f"Quantize: calibration item {example_index} has no token rows.")
        if len(input_ids) != len(attention_mask):
            raise ValueError(
                f"Quantize: calibration item {example_index} has {len(input_ids)} input_ids "
                f"rows but {len(attention_mask)} attention_mask rows."
            )

        rows_changed = False
        prepared_rows = []

        for row_index, (row_ids, row_mask) in enumerate(zip(input_ids, attention_mask)):
            if len(row_ids) == 0:
                # An empty token row is a too-short sample, not a malformed one:
                # tokenizers without a BOS encode `""` as `(1, 0)`. Drop it
                # through the minimum-length filter below so one blank line in
                # the calibration data cannot abort the whole run.
                too_short_row_count += 1
                rows_changed = True
                continue
            if len(row_ids) != len(row_mask):
                raise ValueError(
                    f"Quantize: calibration item {example_index} row {row_index} has "
                    f"{len(row_ids)} tokens but an attention_mask of length {len(row_mask)}."
                )

            row_len = len(row_ids)
            if not concat_mode and max_positions is not None and row_len > max_positions:
                trimmed_row_count += 1
                longest_trimmed_row = max(longest_trimmed_row, row_len)
                rows_changed = True

                # Padding is not part of the sample. Strip it before spending the
                # position budget: slicing a left-padded row by width keeps the
                # leading pads and drops real tokens, and a row whose padding
                # alone fills the budget would come out as pure padding with an
                # all-zero mask.
                real_positions = [pos for pos, keep in enumerate(row_mask) if keep]
                row_ids = [row_ids[pos] for pos in real_positions]
                row_mask = [row_mask[pos] for pos in real_positions]

                if truncation_side == "left":
                    row_ids = row_ids[-max_positions:]
                    row_mask = row_mask[-max_positions:]
                else:
                    row_ids = row_ids[:max_positions]
                    row_mask = row_mask[:max_positions]
            else:
                # Padding is not part of the sample. A row below the position
                # budget may still be pre-padded by `tokenizer(texts,
                # padding=True)` / `padding="max_length"`; keeping the pads
                # makes every capture and per-layer replay forward run at the
                # padded width and turns the length sort into a no-op (#4889).
                # Strip the contiguous pad positions at either edge so the sort
                # and `collate_data` see real lengths and re-pad per batch.
                # Interior mask zeros describe a caller-built mask rather than
                # tokenizer padding, so such a row is left untouched.
                first_real = None
                last_real = None
                real_count = 0
                for pos, keep in enumerate(row_mask):
                    if keep:
                        if first_real is None:
                            first_real = pos
                        last_real = pos
                        real_count += 1
                if first_real is not None and real_count == last_real - first_real + 1:
                    if first_real != 0 or last_real != row_len - 1:
                        row_ids = row_ids[first_real : last_real + 1]
                        row_mask = row_mask[first_real : last_real + 1]
                        rows_changed = True

            # An all-padding row carries no activation statistics: keeping it
            # would only feed the Hessian zeros, so drop it whether or not the
            # row needed trimming.
            if not any(row_mask):
                dropped_padded_row_count += 1
                rows_changed = True
                continue

            # The minimum-length filter is per row and measures real tokens from
            # the attention mask, not the padded width. Otherwise a long first
            # row hides later rows that are mostly padding.
            #
            # Concatenation skips it: packing exists to combine short samples
            # into full windows, and filtering first discarded almost the whole
            # corpus on a dataset of short utterances (#4892). The filter is
            # applied to the packed windows instead.
            if not concat_mode and sum(1 for keep in row_mask if keep) <= calibration_data_min_length:
                too_short_row_count += 1
                rows_changed = True
                continue

            prepared_rows.append((row_ids, row_mask))

        if rows_changed:
            if not prepared_rows:
                too_short_calibration_data_count += 1
                continue

            # Stripping or dropping rows can leave an item ragged. The batched
            # path and the concatenation path consume rows one at a time, but
            # the non-batched path stacks an item into a single tensor, so only
            # that path needs the item made rectangular again.
            widths = {len(row_ids) for row_ids, _ in prepared_rows}
            if not support_batch_quantize and len(widths) > 1:
                row_width = max(widths)
                pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
                padded_rows = []
                for row_ids, row_mask in prepared_rows:
                    pad_len = row_width - len(row_ids)
                    if pad_len <= 0:
                        padded_rows.append((row_ids, row_mask))
                    elif padding_side == "left":
                        padded_rows.append(([pad_token_id] * pad_len + row_ids, [0] * pad_len + row_mask))
                    else:
                        padded_rows.append((row_ids + [pad_token_id] * pad_len, row_mask + [0] * pad_len))
                prepared_rows = padded_rows

            input_ids = [row_ids for row_ids, _ in prepared_rows]
            attention_mask = [row_mask for _, row_mask in prepared_rows]

        new_calibration_dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    if too_short_calibration_data_count > 0:
        log.warn(
            f"Quantize: {too_short_calibration_data_count} calibration items were removed because every row was "
            f"empty, all-padding, or <= {calibration_data_min_length} non-padding tokens. "
            f"Use CalibrationConfig(min_length=...) to set a custom minimum length."
        )

    if too_short_row_count > 0:
        log.warn(
            f"Quantize: {too_short_row_count} calibration rows with <= {calibration_data_min_length} "
            "non-padding tokens were removed. "
            f"Use CalibrationConfig(min_length=...) to set a custom minimum length."
        )

    if message_examples > 0 and message_template_name is not None:
        log.info(
            "Calibration: tokenized %s `messages` examples via tokenizer.%s",
            message_examples,
            message_template_name,
        )
    if ignored_source_examples > 0:
        log.info(
            "Calibration: used the provided `input_ids` for %s examples that also carried `text`/`messages`; "
            "the source column was not re-tokenized.",
            ignored_source_examples,
        )
    if message_text_fallback_examples > 0:
        log.warn(
            "Calibration: fell back to raw `text` for %s `messages` examples because the tokenizer has no message "
            "template configured.",
            message_text_fallback_examples,
        )

    if trimmed_row_count > 0:
        log.info(
            "Quantize: trimmed %s calibration rows above %s=%s (longest original length=%s, truncation_side=%s)",
            trimmed_row_count,
            max_positions_source,
            max_positions,
            longest_trimmed_row,
            truncation_side,
        )

    if dropped_padded_row_count > 0:
        log.warn(
            "Quantize: dropped %s calibration rows that had no non-padding token.",
            dropped_padded_row_count,
        )

    if calibration_dataset_concat_size:
        _require_tokenizer("`calibration_dataset_concat_size` is specified")
        concatenated_data = []
        input_ids_buff = []
        attention_mask_buff = []
        current_length = 0

        separator = calibration_concat_separator if calibration_concat_separator is not None else ""
        if separator:
            # The separator is inserted between already-tokenized documents, so
            # it must not contribute special tokens (a BOS here lands at every
            # document boundary).
            new_line = tokenizer(  # type: ignore[call-arg]
                separator,
                add_special_tokens=False,
                return_tensors="pt",
            )
            new_line_input_ids = _convert_tensor_to_list(new_line["input_ids"])[0]
            new_line_attention_mask = _convert_tensor_to_list(new_line["attention_mask"])[0]
        else:
            new_line_input_ids = []
            new_line_attention_mask = []
        new_line_input_ids_len = len(new_line_input_ids)

        def flush_buffer():
            nonlocal input_ids_buff, attention_mask_buff, current_length, too_short_window_count
            # `min_length` is measured on the packed window here, not on the
            # source rows: a full window carries `concat_size` real tokens, so
            # only a near-empty tail window is dropped.
            if sum(attention_mask_buff) <= calibration_data_min_length:
                too_short_window_count += 1
            else:
                concatenated_data.append(
                    {
                        "input_ids": [input_ids_buff],
                        "attention_mask": [attention_mask_buff],
                    }
                )
            input_ids_buff = []
            attention_mask_buff = []
            current_length = 0

        for row_ids, row_mask in _iter_calibration_rows(new_calibration_dataset):
            # Strip pad positions before packing. Rows that needed trimming were
            # already stripped above, but a padded row below the position budget
            # (e.g. from `tokenizer(texts, padding=True)`) still carries pad ids
            # that must never enter a concatenated window.
            if not all(row_mask):
                row_ids = [token for token, keep in zip(row_ids, row_mask) if keep]
                row_mask = [keep for keep in row_mask if keep]

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

        if input_ids_buff and sum(attention_mask_buff) <= calibration_data_min_length:
            # The tail window never reached a full block and is shorter than the
            # minimum, so it carries no usable activation statistics.
            too_short_window_count += 1
            input_ids_buff = []
            attention_mask_buff = []

        if input_ids_buff:
            padding_length = calibration_dataset_concat_size - len(input_ids_buff)
            if padding_length > 0:
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    # A tokenizer without a pad token still needs a value for
                    # the final window's tail. EOS is the conventional safe
                    # fallback, then 0; the zero mask keeps these slots out of
                    # the Hessian.
                    pad_id = getattr(tokenizer, "eos_token_id", None)
                if pad_id is None:
                    pad_id = 0
                if padding_side == "left":
                    input_ids_buff = ([pad_id] * padding_length) + input_ids_buff
                    attention_mask_buff = ([0] * padding_length) + attention_mask_buff
                else:
                    input_ids_buff.extend([pad_id] * padding_length)
                    attention_mask_buff.extend([0] * padding_length)
            concatenated_data.append(
                {
                    "input_ids": [input_ids_buff],
                    "attention_mask": [attention_mask_buff],
                }
            )

        if too_short_window_count > 0:
            log.warn(
                f"Quantize: {too_short_window_count} concatenated calibration windows with "
                f"<= {calibration_data_min_length} non-padding tokens were removed. "
                "`min_length` is applied to the packed windows when `concat_size` is set."
            )

        if not concatenated_data:
            raise ValueError(
                "Quantize: concatenation produced no calibration window with more than "
                f"{calibration_data_min_length} non-padding tokens. "
                "`calibration_dataset_concat_size` must be larger than `min_length`."
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
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
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
        new_calibration_dataset_batched = [
            {
                "input_ids": torch.tensor(block["input_ids"], dtype=torch.long),
                # Keep the mask even without batch collation: a padded tail block
                # (or a user row that already carried padding) must not let pad
                # tokens enter the Hessian as real activations.
                "attention_mask": torch.tensor(block["attention_mask"], dtype=torch.long),
            }
            for block in sorted_dataset
        ]

    return new_calibration_dataset_batched


#: A window shorter than this carries no usable activation statistics.
MIN_WINDOW_TOKENS = 8


def _record_text(record: Any) -> Optional[str]:
    """Pull the text out of one calibration record."""

    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in ("text", "content", "prompt", "question"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        joined = " ".join(str(v) for v in record.values() if isinstance(v, str))
        return joined or None
    return None


def load_calibration_records(
    source: Optional[str] = None, *, rows: int = 0
) -> List[str]:
    """Load calibration texts from the source the GPTQ test harness uses.

    Resolution order mirrors ``tests/models/model_test.py`` so the TT path and the
    GPTQ path calibrate on the same records: an explicit ``source``, then
    ``$GPTQMODEL_CALIBRATION_PARQUET``, then the local ``nm-calibration`` parquet,
    then the ``nm-calibration`` dataset repo. Parquet needs ``pyarrow``; the JSON
    and JSONL forms do not.
    """

    source = (
        source
        or os.environ.get("GPTQMODEL_CALIBRATION_PARQUET")
        or "/monster/data/model/dataset/nm-calibration/llm.parquet"
    )

    records: Any
    if os.path.isfile(source):
        if source.endswith(".gz"):
            import gzip
            import json

            with gzip.open(source, "rt") as handle:
                records = json.load(handle)
        elif source.endswith(".jsonl"):
            import json

            with open(source) as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        elif source.endswith(".json"):
            import json

            with open(source) as handle:
                records = json.load(handle)
        elif source.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    f"reading {source} needs pyarrow; install a numpy-compatible "
                    "build or export the records as JSONL"
                ) from exc
            records = pq.read_table(source).to_pylist()
        else:
            raise ValueError(f"unsupported calibration file type: {source}")
    else:
        if source.startswith(("/", "./", "../", "~")):
            raise FileNotFoundError(
                f"calibration file not found: {os.path.expanduser(source)}; pass a "
                "local file or a Hub dataset repo id instead"
            )
        # Treat it as a Hub dataset repo and fetch the JSON payload it ships.
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list(list_repo_files(source, repo_type="dataset"))
        preferred = [
            name for name in files
            if name.endswith((".json.gz", ".json", ".jsonl", ".parquet"))
        ]
        if not preferred:
            raise ValueError(f"{source} ships no JSON or parquet calibration file")
        local = hf_hub_download(
            source, preferred[0], repo_type="dataset"
        )
        return load_calibration_records(local, rows=rows)

    if isinstance(records, dict):
        records = list(records.values())
    texts = [text for text in (_record_text(r) for r in records) if text]
    if not texts:
        raise ValueError(f"no calibration text found in {source}")
    return texts[:rows] if rows else texts


def iter_calibration_windows(
    tokenizer,
    text: Union[str, Sequence[str]],
    *,
    seqlen: int = 256,
    nsamples: int = 32,
    min_tokens: int = MIN_WINDOW_TOKENS,
) -> Iterator[torch.Tensor]:
    """Yield ``nsamples`` distinct ``[1, seqlen]`` windows of ``text``.

    ``text`` may be one string or a sequence of calibration records. Records are
    tokenized individually -- as the GPTQ calibration loader does -- and the ids
    are concatenated before windowing, so a dataset-based run covers the same
    records the quantizer would see.

    GPTQ's error feedback pushes each layer's rounding error into directions where
    the calibration input has little curvature, so the Hessian decides *where* the
    error lands. If every sample is the same passage, "low curvature" means
    "rarely active in that passage" and the error is steered into directions that
    are ordinary for every other text.

    That happened here: the Tenstorrent block-float path collected its Hessian with
    ``ids[:, :seqlen]`` on every iteration, so ``nsamples`` multiplied a single
    passage. The resulting Llama-3.2-1B checkpoint measured perplexity 354 on
    held-out questions against 9.03 dense, while plain per-layer rounding of the
    same weights gave 10.21 -- and every per-layer metric looked correct. Keeping
    the windowing in one place stops a quantizer from reintroducing that shape, and
    reports a short calibration string instead of silently degrading into a single
    passage.

    Windows are spread across the whole stream, so the sample budget covers the
    corpus instead of only its beginning. With more than one sample the first
    window starts at the top, the last ends at the bottom, and the rest are
    evenly spaced between, so both endpoints of the legal start range
    ``0..total - seqlen`` are selected. A single sample cannot both start at the
    top and end at the bottom unless the corpus is exactly one window long, so
    the singleton policy anchors that window at the top of the stream. When the
    text is shorter than ``nsamples * seqlen`` the windows necessarily overlap,
    and the warning says so.
    """

    if seqlen < min_tokens:
        raise ValueError(f"seqlen must be >= {min_tokens}, got {seqlen}")
    if nsamples < 1:
        raise ValueError(f"nsamples must be positive, got {nsamples}")

    log = setup_logger()
    if isinstance(text, str):
        ids = tokenizer(text, return_tensors="pt").input_ids
    else:
        rows = [row for row in text if isinstance(row, str) and row.strip()]
        if not rows:
            raise ValueError("calibration records are empty")
        ids = torch.cat(
            [tokenizer(row, return_tensors="pt").input_ids for row in rows], dim=1
        )
    if ids.dim() != 2:
        ids = ids.unsqueeze(0)
    total = int(ids.shape[1])
    if total < min_tokens:
        raise ValueError(
            f"calibration text tokenized to {total} tokens, need at least "
            f"{min_tokens}; a Hessian built from this would be meaningless"
        )

    if total <= seqlen:
        log.warning(
            f"calibration text is {total} tokens but seqlen={seqlen}: all "
            f"{nsamples} samples are the same window, so the Hessian describes one "
            "passage. Pass longer calibration text."
        )
        for _ in range(nsamples):
            yield ids
        return

    last_start = total - seqlen
    budget = nsamples * seqlen
    if total < budget:
        log.warning(
            f"calibration text has {total} distinct tokens but {nsamples} windows "
            f"of {seqlen} need {budget}: the windows overlap, so the Hessian "
            f"describes only {total} tokens. Pass longer calibration text."
        )

    if nsamples == 1:
        # One window cannot reach both ends of a stream longer than ``seqlen``.
        # Anchor it at the beginning so the singleton case is explicit rather
        # than an accidental consequence of the spacing formula.
        starts = [0]
    else:
        # Spread the windows over the whole stream rather than packing them at
        # ``index * seqlen``. Packing only ever reaches ``nsamples * seqlen``
        # tokens, so a 32 x 256 budget read the first ~8k tokens of a
        # 100k-token corpus and silently ignored the other 92 %. Dividing by
        # ``nsamples - 1`` includes both endpoints of the legal start range, so
        # the final window ends exactly at the end of the stream instead of one
        # stride short of it.
        starts = [
            (index * last_start) // (nsamples - 1) for index in range(nsamples)
        ]

    for start in starts:
        yield ids[:, start : start + seqlen]


__all__ = [
    "MIN_WINDOW_TOKENS",
    "batched",
    "iter_calibration_windows",
    "load_calibration_records",
    "prepare_calibration_dataset",
]
